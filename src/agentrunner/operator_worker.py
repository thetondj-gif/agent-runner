"""Persistent Deus operator sidecar built on Agent Runner.

This module is deliberately isolated from the normal `agentrunner run/chat` paths.
It polls the existing Deus `operator-tasks` Edge Function, claims one approved
mission at a time, executes it inside an allowed workspace, checkpoints the
Agent Runner session, verifies the result, and retries recoverable failures.

Nothing runs unless DEUS_OPERATOR_ENABLED=1 is set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from agentrunner.core.config import AgentConfig
from agentrunner.core.factory import create_agent
from agentrunner.providers.base import ProviderConfig


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(frozen=True)
class OperatorConfig:
    endpoint: str
    token: str
    default_workspace: Path
    allowed_root: Path
    model: str
    base_url: str | None = None
    context_window: int = 32768
    poll_seconds: float = 5.0
    stale_seconds: int = 900
    max_attempts: int = 5
    max_rounds: int = 80
    tool_timeout_s: int = 300
    strict_commands: bool = False
    allowed_risks: tuple[str, ...] = ("low", "medium")
    enabled: bool = False

    @classmethod
    def from_env(cls) -> "OperatorConfig":
        supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
        endpoint = os.getenv("DEUS_OPERATOR_URL", "").strip()
        if not endpoint and supabase_url:
            endpoint = f"{supabase_url}/functions/v1/operator-tasks"

        default_workspace = (
            Path(os.getenv("DEUS_OPERATOR_DEFAULT_WORKSPACE", ".")).expanduser().resolve()
        )
        allowed_root = (
            Path(os.getenv("DEUS_OPERATOR_ALLOWED_ROOT", str(default_workspace)))
            .expanduser()
            .resolve()
        )

        risks = tuple(
            item.strip().lower()
            for item in os.getenv("DEUS_OPERATOR_ALLOWED_RISKS", "low,medium").split(",")
            if item.strip()
        )

        return cls(
            endpoint=endpoint,
            token=os.getenv("DEUS_OPERATOR_TOKEN", ""),
            default_workspace=default_workspace,
            allowed_root=allowed_root,
            model=os.getenv("DEUS_OPERATOR_MODEL", "qwen3.5-9b-mlx-4bit"),
            base_url=os.getenv("AGENTRUNNER_OPENAI_BASE_URL"),
            context_window=int(os.getenv("AGENTRUNNER_CONTEXT_WINDOW", "32768")),
            poll_seconds=float(os.getenv("DEUS_OPERATOR_POLL_SECONDS", "5")),
            stale_seconds=int(os.getenv("DEUS_OPERATOR_STALE_SECONDS", "900")),
            max_attempts=int(os.getenv("DEUS_OPERATOR_MAX_ATTEMPTS", "5")),
            max_rounds=int(os.getenv("DEUS_OPERATOR_MAX_ROUNDS", "80")),
            tool_timeout_s=int(os.getenv("DEUS_OPERATOR_TOOL_TIMEOUT", "300")),
            strict_commands=_as_bool(os.getenv("DEUS_OPERATOR_STRICT_COMMANDS"), False),
            allowed_risks=risks or ("low",),
            enabled=_as_bool(os.getenv("DEUS_OPERATOR_ENABLED"), False),
        )

    def validate(self) -> None:
        if not self.enabled:
            raise RuntimeError("DEUS_OPERATOR_ENABLED is not set to 1")
        if not self.endpoint:
            raise RuntimeError("DEUS_OPERATOR_URL or SUPABASE_URL is required")
        if not self.token:
            raise RuntimeError("DEUS_OPERATOR_TOKEN is required")
        if not self.allowed_root.exists() or not self.allowed_root.is_dir():
            raise RuntimeError(f"Allowed root does not exist: {self.allowed_root}")
        if not self.default_workspace.exists() or not self.default_workspace.is_dir():
            raise RuntimeError(f"Default workspace does not exist: {self.default_workspace}")


class OperatorTasksClient:
    def __init__(self, config: OperatorConfig, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self._client = client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _call(self, action: str, **payload: Any) -> dict[str, Any]:
        response = await self._client.post(
            self.config.endpoint,
            headers={"x-operator-token": self.config.token},
            json={"action": action, **payload},
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(str(data["error"]))
        return data

    async def list(self, limit: int = 50) -> list[dict[str, Any]]:
        data = await self._call("list", limit=limit)
        missions = data.get("missions", [])
        return missions if isinstance(missions, list) else []

    async def next(self) -> dict[str, Any] | None:
        data = await self._call("next")
        mission = data.get("mission")
        return mission if isinstance(mission, dict) else None

    async def update(self, mission_id: str, **payload: Any) -> dict[str, Any]:
        data = await self._call("update", id=mission_id, **payload)
        mission = data.get("mission")
        if not isinstance(mission, dict):
            raise RuntimeError(f"Mission update returned no mission: {mission_id}")
        return mission


class PersistentOperator:
    def __init__(self, config: OperatorConfig, tasks: OperatorTasksClient | None = None) -> None:
        self.config = config
        self.tasks = tasks or OperatorTasksClient(config)
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    def _runtime(self, mission: dict[str, Any]) -> dict[str, Any]:
        runtime = mission.get("runtime")
        return runtime if isinstance(runtime, dict) else {}

    def _resolve_workspace(self, mission: dict[str, Any]) -> Path:
        runtime = self._runtime(mission)
        requested = runtime.get("workspace_path")
        workspace = (
            Path(str(requested)).expanduser().resolve()
            if requested
            else self.config.default_workspace
        )
        try:
            workspace.relative_to(self.config.allowed_root)
        except ValueError as exc:
            raise RuntimeError(
                f"Workspace {workspace} is outside allowed root {self.config.allowed_root}"
            ) from exc
        if not workspace.exists() or not workspace.is_dir():
            raise RuntimeError(f"Workspace does not exist: {workspace}")
        return workspace

    async def recover_stale(self) -> int:
        recovered = 0
        now = _utcnow()
        for mission in await self.tasks.list(limit=50):
            if str(mission.get("runtime_stage", "")).upper() != "RUNNING":
                continue
            runtime = self._runtime(mission)
            updated = _parse_iso(str(runtime.get("updated_at") or mission.get("updated") or ""))
            if updated is None:
                continue
            age = (now - updated).total_seconds()
            if age < self.config.stale_seconds:
                continue
            await self.tasks.update(
                str(mission["id"]),
                stage="RETRY",
                last_error=f"Recovered stale RUNNING lease after {int(age)}s",
                event={
                    "type": "worker_recovery",
                    "status": "RETRY",
                    "message": "Recovered stale mission after worker/process interruption",
                },
            )
            recovered += 1
        return recovered

    def _build_prompt(self, mission: dict[str, Any]) -> str:
        runtime = self._runtime(mission)
        acceptance = runtime.get("acceptance_criteria")
        criteria = acceptance if isinstance(acceptance, list) else []
        criteria_text = (
            "\n".join(f"- {item}" for item in criteria)
            or "- Complete the objective safely and verifiably."
        )
        return (
            "Own this job until the requested outcome is actually complete. "
            "Do not stop at a plan, status report, or explanation. Reuse existing project capabilities first. "
            "When a tool or approach fails, inspect the failure, repair it or use the next safe route, then continue. "
            "Do not make destructive, financial, external-send, credential-rotation, or production-release actions "
            "unless the mission explicitly authorizes them.\n\n"
            f"OBJECTIVE:\n{mission.get('objective', '')}\n\n"
            f"ACCEPTANCE CRITERIA:\n{criteria_text}\n\n"
            "Before declaring completion, run the relevant tests/QA and collect concrete evidence. "
            "Finish with a concise result containing what changed, verification performed, and remaining blockers, if any."
        )

    def _build_verify_prompt(self, mission: dict[str, Any]) -> str:
        runtime = self._runtime(mission)
        acceptance = runtime.get("acceptance_criteria")
        criteria = acceptance if isinstance(acceptance, list) else []
        criteria_text = (
            "\n".join(f"- {item}" for item in criteria) or "- Objective is complete and working."
        )
        return (
            "Act as an independent verifier for the work just performed in this workspace. "
            "Do not rely on the previous agent's claims. Inspect the actual files/state and run the relevant tests or QA. "
            "Repair only small obvious verification blockers when safe; otherwise report failure.\n\n"
            f"OBJECTIVE:\n{mission.get('objective', '')}\n\n"
            f"ACCEPTANCE CRITERIA:\n{criteria_text}\n\n"
            "Your final line MUST be exactly 'VERDICT: PASS' only when the acceptance criteria are genuinely met; "
            "otherwise the final line MUST be exactly 'VERDICT: FAIL'. Include concrete evidence before the verdict."
        )

    async def _run_agent(self, mission: dict[str, Any], workspace: Path) -> tuple[str, str]:
        runtime = self._runtime(mission)
        extensions: dict[str, Any] = {"context_window": self.config.context_window}
        if self.config.base_url:
            extensions["base_url"] = self.config.base_url
            extensions["openai_compatible"] = True

        provider = ProviderConfig(
            model=self.config.model,
            temperature=0.2,
            max_tokens=4096,
            provider_extensions=extensions,
        )
        agent_cfg = AgentConfig(
            max_rounds=self.config.max_rounds,
            tool_timeout_s=self.config.tool_timeout_s,
        )

        # The mission itself is already token-authenticated and approval-gated.
        # CommandValidator still blocks blacklist/dangerous patterns even when
        # strict_commands=False; this simply permits normal project commands.
        agent = create_agent(
            workspace_path=str(workspace),
            provider_config=provider,
            agent_config=agent_cfg,
            strict_commands=self.config.strict_commands,
            require_confirmation=False,
        )

        session_id = f"deus-{mission['id']}"
        try:
            await agent.load_session(session_id)
        except FileNotFoundError:
            pass

        result = await agent.process_message(self._build_prompt(mission))
        await agent.save_session(session_id)

        verifier = create_agent(
            workspace_path=str(workspace),
            provider_config=provider,
            agent_config=agent_cfg,
            strict_commands=self.config.strict_commands,
            require_confirmation=False,
        )
        verify_result = await verifier.process_message(self._build_verify_prompt(mission))
        await verifier.save_session(f"{session_id}-verify")
        return result.content, verify_result.content

    async def execute_claimed(self, mission: dict[str, Any]) -> None:
        mission_id = str(mission["id"])
        runtime = self._runtime(mission)
        attempts = int(runtime.get("attempts") or 1)
        max_attempts = int(runtime.get("max_attempts") or self.config.max_attempts)
        risk = str(mission.get("risk") or "low").lower()

        if risk not in self.config.allowed_risks:
            await self.tasks.update(
                mission_id,
                stage="BLOCKED",
                approval="REQUIRED",
                last_error=f"Risk '{risk}' is outside autonomous worker allowance",
                event={
                    "type": "policy_block",
                    "status": "BLOCKED",
                    "message": f"Risk '{risk}' requires human approval",
                },
            )
            return

        try:
            workspace = self._resolve_workspace(mission)
            await self.tasks.update(
                mission_id,
                stage="RUNNING",
                event={
                    "type": "worker_claimed",
                    "status": "RUNNING",
                    "workspace": str(workspace),
                    "attempt": attempts,
                },
            )
            result_text, verify_text = await self._run_agent(mission, workspace)
            passed = verify_text.rstrip().endswith("VERDICT: PASS")
            evidence = [
                {"type": "agent_result", "text": result_text[-6000:]},
                {"type": "verification", "text": verify_text[-6000:]},
            ]
            if passed:
                await self.tasks.update(
                    mission_id,
                    stage="COMPLETE",
                    status="CLOSED",
                    closed=True,
                    last_error=None,
                    evidence=evidence,
                    event={
                        "type": "verification",
                        "status": "PASS",
                        "message": "Independent verification pass completed",
                    },
                )
                return

            if attempts >= max_attempts:
                await self.tasks.update(
                    mission_id,
                    stage="BLOCKED",
                    last_error="Verification failed and retry budget is exhausted",
                    evidence=evidence,
                    event={
                        "type": "verification",
                        "status": "FAIL",
                        "message": "Retry budget exhausted",
                    },
                )
            else:
                await self.tasks.update(
                    mission_id,
                    stage="RETRY",
                    last_error="Independent verification failed",
                    evidence=evidence,
                    event={
                        "type": "verification",
                        "status": "RETRY",
                        "message": "Verification failed; mission queued for another attempt",
                    },
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempts >= max_attempts:
                stage = "BLOCKED"
                message = "Execution failed and retry budget is exhausted"
            else:
                stage = "RETRY"
                message = "Execution failed; mission queued for another attempt"
            await self.tasks.update(
                mission_id,
                stage=stage,
                last_error=error[:4000],
                event={
                    "type": "worker_error",
                    "status": stage,
                    "message": message,
                    "error": error[:2000],
                },
            )

    async def run_once(self) -> bool:
        await self.recover_stale()
        mission = await self.tasks.next()
        if mission is None:
            return False
        await self.execute_claimed(mission)
        return True

    async def serve_forever(self) -> None:
        self.config.validate()
        while not self._stop.is_set():
            worked = await self.run_once()
            if not worked:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.config.poll_seconds)
                except TimeoutError:
                    pass

    async def close(self) -> None:
        await self.tasks.close()


async def _async_main(once: bool) -> int:
    config = OperatorConfig.from_env()
    config.validate()
    worker = PersistentOperator(config)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, worker.stop)
        except NotImplementedError:
            pass

    try:
        if once:
            await worker.run_once()
        else:
            await worker.serve_forever()
    finally:
        await worker.close()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent Deus operator sidecar")
    parser.add_argument("--once", action="store_true", help="Claim at most one mission and exit")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_async_main(args.once)))


if __name__ == "__main__":
    main()
