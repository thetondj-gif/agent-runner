from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Optional per-machine overrides. This file is created by the install helper and
# deliberately lives outside the repository so credentials/provider settings are
# never committed.
load_dotenv(Path.home() / ".config" / "agentrunner" / "mini-gateway.env")


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: str
    executable: str
    argv: tuple[str, ...]
    tier: str
    proven_on_mini: bool = False
    timeout_s: int = 900
    requires_env: tuple[str, ...] = ()


def _agent_specs() -> dict[str, AgentSpec]:
    goose_provider = os.getenv("GOOSE_MINI_PROVIDER", "custom_ollama2")
    goose_model = os.getenv("GOOSE_MINI_MODEL", "qwen2.5:7b")
    aider_model = os.getenv("AIDER_MINI_MODEL", "ollama/qwen2.5:7b")
    opencode_model = os.getenv("OPENCODE_MINI_MODEL", "ollama/qwen2.5:7b")
    aoe_session = os.getenv("AOE_SESSION", "")

    return {
        "goose_local": AgentSpec(
            name="goose_local",
            role="Primary free/local implementation and debugging worker",
            executable="goose",
            argv=(
                "goose",
                "run",
                "--provider",
                goose_provider,
                "--model",
                goose_model,
                "--with-builtin",
                "developer",
                "-t",
                "{task}",
            ),
            tier="free-local",
            proven_on_mini=True,
            timeout_s=1200,
        ),
        "aider_patch": AgentSpec(
            name="aider_patch",
            role="Focused patch/edit worker using Aider with local Ollama by default",
            executable="aider",
            argv=("aider", "--model", aider_model, "--message", "{task}"),
            tier="free-local-default",
        ),
        "opencode_engineer": AgentSpec(
            name="opencode_engineer",
            role="Repository coding worker using OpenCode with local Ollama by default",
            executable="opencode",
            argv=("opencode", "run", "--model", opencode_model, "{task}"),
            tier="free-local-default",
        ),
        "codex_engineer": AgentSpec(
            name="codex_engineer",
            role="Escalation coding worker for difficult implementation tasks",
            executable="codex",
            argv=("codex", "exec", "{task}"),
            tier="quota-or-paid",
        ),
        "claude_architect": AgentSpec(
            name="claude_architect",
            role="Architecture and difficult reasoning escalation worker",
            executable="claude",
            argv=("claude", "-p", "{task}"),
            tier="quota-or-paid",
        ),
        "gemini_reviewer": AgentSpec(
            name="gemini_reviewer",
            role="Large-context review and alternative implementation worker",
            executable="gemini",
            argv=("gemini", "-p", "{task}"),
            tier="configured-provider",
        ),
        "aoe_delegate": AgentSpec(
            name="aoe_delegate",
            role="Delegate a task into an existing Agent of Empires session",
            executable="aoe",
            argv=("aoe", "send", aoe_session, "{task}"),
            tier="multi-agent",
            requires_env=("AOE_SESSION",),
            timeout_s=1200,
        ),
    }


def _allowed_roots() -> list[Path]:
    raw = os.getenv("MINI_AGENT_WORKSPACE_ROOTS")
    if raw:
        values = [item for item in raw.split(os.pathsep) if item.strip()]
    else:
        # Conservative defaults for the Mini. Add other project roots in
        # ~/.config/agentrunner/mini-gateway.env rather than widening to $HOME.
        values = [str(Path.home() / "dawn-v4"), str(Path.home() / "goose-test")]
    return [Path(value).expanduser().resolve() for value in values]


def resolve_workspace(workspace: str) -> Path:
    path = Path(workspace).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Workspace does not exist or is not a directory: {path}")

    roots = _allowed_roots()
    if not any(path == root or path.is_relative_to(root) for root in roots):
        allowed = ", ".join(str(root) for root in roots)
        raise ValueError(f"Workspace {path} is outside allowed roots: {allowed}")
    return path


def _env_ready(spec: AgentSpec) -> tuple[bool, list[str]]:
    missing = [name for name in spec.requires_env if not os.getenv(name)]
    return not missing, missing


def agent_status() -> dict[str, Any]:
    agents: dict[str, Any] = {}
    for name, spec in _agent_specs().items():
        env_ready, missing_env = _env_ready(spec)
        binary = shutil.which(spec.executable)
        agents[name] = {
            **asdict(spec),
            "installed": binary is not None,
            "binary": binary,
            "ready": binary is not None and env_ready,
            "missing_env": missing_env,
            "argv": list(spec.argv),
        }

    return {
        "agents": agents,
        "allowed_workspace_roots": [str(p) for p in _allowed_roots()],
        "routing_policy": [
            "goose_local",
            "aider_patch",
            "opencode_engineer",
            "aoe_delegate",
            "codex_engineer",
            "claude_architect",
            "gemini_reviewer",
        ],
        "default": "goose_local",
    }


def _render_argv(spec: AgentSpec, task: str) -> list[str]:
    return [part.replace("{task}", task) for part in spec.argv]


def _trim(value: str) -> str:
    limit = int(os.getenv("MINI_AGENT_MAX_OUTPUT_CHARS", "60000"))
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated {len(value) - limit} chars]"


async def run_agent(
    agent: str,
    task: str,
    workspace: str,
    *,
    role_instruction: str | None = None,
) -> dict[str, Any]:
    specs = _agent_specs()
    if agent not in specs:
        raise ValueError(f"Unknown agent: {agent}. Available: {', '.join(sorted(specs))}")

    spec = specs[agent]
    binary = shutil.which(spec.executable)
    if binary is None:
        return {
            "success": False,
            "agent": agent,
            "error": f"Executable not installed/on PATH: {spec.executable}",
        }

    env_ready, missing_env = _env_ready(spec)
    if not env_ready:
        return {
            "success": False,
            "agent": agent,
            "error": f"Missing required environment variables: {', '.join(missing_env)}",
        }

    cwd = resolve_workspace(workspace)
    effective_task = task.strip()
    if role_instruction:
        effective_task = f"{role_instruction.strip()}\n\nTASK:\n{effective_task}"

    argv = _render_argv(spec, effective_task)
    started = time.monotonic()

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=spec.timeout_s)
        duration = time.monotonic() - started
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        return {
            "success": proc.returncode == 0,
            "agent": agent,
            "role": spec.role,
            "tier": spec.tier,
            "proven_on_mini": spec.proven_on_mini,
            "exit_code": proc.returncode,
            "duration_seconds": round(duration, 2),
            "workspace": str(cwd),
            "command": shlex.join([argv[0], *["<task>" if item == effective_task else item for item in argv[1:]]]),
            "stdout": _trim(stdout),
            "stderr": _trim(stderr),
        }
    except TimeoutError:
        duration = time.monotonic() - started
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        return {
            "success": False,
            "agent": agent,
            "error": f"Timed out after {spec.timeout_s}s",
            "duration_seconds": round(duration, 2),
            "workspace": str(cwd),
        }
