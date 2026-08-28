#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agentrunner.mini_gateway.executor import agent_status, resolve_workspace, run_agent


JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _now() -> float:
    return round(time.time(), 3)


def _update_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job.update(updates)
        job["updated_at"] = _now()


def _append_log(job_id: str, message: str) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        logs = list(job.get("logs", []))
        logs.append(str(message))
        # Keep the HTTP payload bounded while retaining the newest evidence.
        job["logs"] = logs[-250:]
        job["updated_at"] = _now()


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return (cleaned or "venture")[:80]


def _validated_github_repo(value: str) -> str:
    raw = value.strip()
    parsed = urlparse(raw)
    if parsed.scheme != "https" or parsed.hostname != "github.com":
        raise ValueError("Repository must be an https://github.com/<owner>/<repo> URL")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2:
        raise ValueError("Repository URL must identify exactly one GitHub repository")
    owner, repo = parts
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner) or not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
        raise ValueError("Repository owner/name contains unsupported characters")
    return f"https://github.com/{owner}/{repo}.git"


def _workspace_root() -> Path:
    configured = os.getenv(
        "MINI_REST_WORKSPACE_ROOT",
        str(Path.home() / "dawn-v4" / "venture-foundry-workspaces"),
    )
    root = Path(configured).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Reuse the Mini gateway's workspace allowlist. The bridge cannot widen it.
    return resolve_workspace(str(root))


def _run_checked(argv: list[str], *, cwd: Path | None = None, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
    )


def _tail_lines(value: str, limit: int = 120) -> list[str]:
    lines = [line for line in value.splitlines() if line.strip()]
    return lines[-limit:]


def _prepare_workspace(job_id: str, project_name: str, repository: str) -> Path:
    root = _workspace_root()
    destination = root / f"{_slug(project_name)}-{job_id[:8]}"
    if destination.exists():
        raise RuntimeError(f"Refusing to reuse existing workspace: {destination}")

    _run_checked(["git", "clone", repository, str(destination)], timeout=240)
    resolve_workspace(str(destination))
    _run_checked(["git", "checkout", "-b", f"venture-foundry/{job_id[:8]}"], cwd=destination)
    return destination


def _execute_job(job_id: str, payload: dict[str, Any]) -> None:
    project_name = str(payload.get("projectName") or "Venture")
    prompt = str(payload.get("prompt") or "").strip()
    specification = payload.get("specification") or {}
    if not isinstance(specification, dict):
        specification = {}
    repository = str(specification.get("repository") or "").strip()
    source_type = str(specification.get("source_type") or "").strip()
    worker = os.getenv("MINI_REST_BUILDER_AGENT", "goose_local").strip() or "goose_local"

    try:
        if not prompt:
            raise ValueError("Build prompt is required")
        repository = _validated_github_repo(repository)

        _update_job(job_id, status="Specifying", progress=5, agent=worker)
        _append_log(job_id, f"Preparing isolated workspace from {repository}")
        workspace = _prepare_workspace(job_id, project_name, repository)
        _update_job(job_id, workspace=str(workspace), status="Code Generation", progress=15)
        _append_log(job_id, f"Isolated git workspace ready: {workspace}")
        _append_log(job_id, f"Starting local worker: {worker}")

        role_instruction = (
            "Act as a bounded repository repair engineer inside this isolated git clone. "
            "You already have direct filesystem and shell access to the cloned repository at your current working directory. "
            "Do not ask the user to run commands for you. Inspect the repository yourself before editing, preserve the existing architecture, "
            "make the smallest safe change, run the repository's real validation commands yourself, and report exact evidence. "
            "Do not push, merge, deploy, publish, use destructive git clean/reset commands, or invent success."
        )
        result = asyncio.run(
            run_agent(
                worker,
                prompt,
                str(workspace),
                role_instruction=role_instruction,
            )
        )

        stdout = str(result.get("stdout") or "")
        stderr = str(result.get("stderr") or "")
        for line in _tail_lines(stdout):
            _append_log(job_id, line)
        for line in _tail_lines(stderr, 60):
            _append_log(job_id, f"stderr: {line}")

        git_status = ""
        diff_stat = ""
        try:
            git_status = _run_checked(["git", "status", "--short"], cwd=workspace, timeout=30).stdout.strip()
            diff_stat = _run_checked(["git", "diff", "--stat"], cwd=workspace, timeout=30).stdout.strip()
            _append_log(job_id, "git status --short: " + (git_status or "clean"))
            if diff_stat:
                _append_log(job_id, "git diff --stat:")
                for line in _tail_lines(diff_stat, 80):
                    _append_log(job_id, line)
        except Exception as git_error:
            _append_log(job_id, f"Post-run git evidence unavailable: {git_error}")

        _update_job(job_id, git_status=git_status, git_diff_stat=diff_stat)

        process_success = bool(result.get("success"))
        repair_requires_change = source_type in {"github_repo", "existing_codebase"}
        changed = bool(git_status.strip())

        if process_success and repair_requires_change and not changed:
            error = (
                "Worker exited successfully but produced no repository changes for an explicit repair/build job. "
                "Execution is therefore not accepted as a completed repair."
            )
            _append_log(job_id, error)
            _update_job(job_id, status="Failed", progress=100, error=error, result=result)
            return

        if process_success:
            _append_log(
                job_id,
                "Local worker process completed with repository evidence. This is execution completion, not deployment or acceptance-test PASS.",
            )
            _update_job(job_id, status="Completed", progress=100, result=result)
        else:
            error = str(result.get("error") or f"Worker exited with code {result.get('exit_code')}")
            _append_log(job_id, f"Local worker failed: {error}")
            _update_job(job_id, status="Failed", progress=100, error=error, result=result)
    except Exception as exc:
        _append_log(job_id, f"Bridge execution failed: {exc}")
        _update_job(job_id, status="Failed", progress=100, error=str(exc))


class Handler(BaseHTTPRequestHandler):
    server_version = "MiniRestBuilder/1.2"

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, status: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
        self.wfile.flush()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()
        self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            status = agent_status()
            worker = os.getenv("MINI_REST_BUILDER_AGENT", "goose_local").strip() or "goose_local"
            worker_status = status.get("agents", {}).get(worker, {})
            self._json(
                200,
                {
                    "status": "HEALTHY" if worker_status.get("ready") else "DEGRADED",
                    "worker": worker,
                    "worker_ready": bool(worker_status.get("ready")),
                    "worker_proven_on_mini": bool(worker_status.get("proven_on_mini")),
                    "workspace_root": str(_workspace_root()),
                    "bridge_version": "1.2",
                },
            )
            return

        match = re.fullmatch(r"/builds/([A-Za-z0-9-]+)", self.path)
        if not match:
            self._json(404, {"error": "Not found"})
            return
        job_id = match.group(1)
        with JOBS_LOCK:
            job = dict(JOBS.get(job_id, {}))
        if not job:
            self._json(404, {"error": "Unknown build id"})
            return
        self._json(200, job)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/builds":
            self._json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000:
                raise ValueError("Request body is missing or too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            prompt = str(payload.get("prompt") or "").strip()
            specification = payload.get("specification") or {}
            repository = specification.get("repository") if isinstance(specification, dict) else ""
            if not prompt:
                raise ValueError("prompt is required")
            _validated_github_repo(str(repository or ""))

            worker = os.getenv("MINI_REST_BUILDER_AGENT", "goose_local").strip() or "goose_local"
            worker_info = agent_status().get("agents", {}).get(worker)
            if not worker_info:
                raise ValueError(f"Unknown MINI_REST_BUILDER_AGENT: {worker}")
            if not worker_info.get("ready"):
                raise RuntimeError(f"Configured worker is not ready: {worker}")

            job_id = str(uuid.uuid4())
            now = _now()
            job = {
                "buildId": job_id,
                "status": "Queued",
                "progress": 0,
                "logs": [f"Accepted by loopback Mini REST builder; worker={worker}"],
                "projectName": str(payload.get("projectName") or "Venture"),
                "created_at": now,
                "updated_at": now,
            }
            with JOBS_LOCK:
                JOBS[job_id] = job

            # Snapshot the acceptance payload before the worker thread can mutate
            # the shared job object. This prevents the response race that could
            # leave browser fetches spinning even though the worker had started.
            response_job = dict(job)
            thread = threading.Thread(target=_execute_job, args=(job_id, payload), daemon=True)
            thread.start()
            self._json(202, response_job)
        except Exception as exc:
            self._json(400, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[mini-rest-builder] {self.address_string()} {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Loopback REST builder bridge for Venture Foundry")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("Refusing non-loopback bind; use 127.0.0.1 or localhost")

    root = _workspace_root()
    worker = os.getenv("MINI_REST_BUILDER_AGENT", "goose_local").strip() or "goose_local"
    info = agent_status().get("agents", {}).get(worker, {})
    print(f"Mini REST builder: http://{args.host}:{args.port}", flush=True)
    print(f"Worker: {worker} ready={info.get('ready')} proven_on_mini={info.get('proven_on_mini')}", flush=True)
    print(f"Workspace root: {root}", flush=True)
    print("Bridge version: 1.2 (race-safe acceptance response; repair jobs fail closed without repo changes)", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
