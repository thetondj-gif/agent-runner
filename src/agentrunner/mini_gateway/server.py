from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from agentrunner.mini_gateway.capabilities import capability_catalog, execute_capability, plan_route
from agentrunner.mini_gateway.executor import agent_status, run_agent

mcp = MCPServer("Mac Mini Specialist Agents")


@mcp.tool()
def mini_status() -> dict[str, Any]:
    """Return installed/ready specialist agents and the routing policy for this Mac Mini."""
    return agent_status()


@mcp.tool()
def dawn_capabilities() -> dict[str, dict[str, Any]]:
    """List DAWN-style capabilities, governance boundaries, and local-first execution routes."""
    return capability_catalog()


@mcp.tool()
def dawn_plan_route(capability: str, allow_paid: bool = False) -> dict[str, Any]:
    """Plan which specialist should execute a capability without making any changes."""
    return plan_route(capability, allow_paid=allow_paid)


@mcp.tool()
async def dawn_execute(
    capability: str,
    task: str,
    workspace: str,
    approved: bool = False,
    allow_paid: bool = False,
) -> dict[str, Any]:
    """Execute a DAWN capability through the local-first specialist router.

    Publishing/deployment capabilities require approved=true. Paid/quota workers
    are excluded unless allow_paid=true.
    """
    return await execute_capability(
        capability,
        task,
        workspace,
        approved=approved,
        allow_paid=allow_paid,
    )


# Compatibility tools remain available for callers that want an explicit worker.
@mcp.tool()
async def local_build(task: str, workspace: str) -> dict[str, Any]:
    """Use the proven free/local Goose + Ollama worker for implementation work."""
    return await run_agent(
        "goose_local",
        task,
        workspace,
        role_instruction=(
            "Act as a local implementation engineer. Inspect the repository before editing. "
            "Make the smallest coherent changes, run relevant tests, and report exactly what changed."
        ),
    )


@mcp.tool()
async def local_debug(task: str, workspace: str) -> dict[str, Any]:
    """Use the proven free/local Goose + Ollama worker to diagnose and fix a bug."""
    return await run_agent(
        "goose_local",
        task,
        workspace,
        role_instruction=(
            "Act as a debugging engineer. Reproduce or inspect the failure first, identify root cause, "
            "then make a minimal fix and run validation. Do not claim success without evidence."
        ),
    )


@mcp.tool()
async def local_analysis(task: str, workspace: str) -> dict[str, Any]:
    """Use the local Goose worker for repository analysis; request no file changes unless needed."""
    return await run_agent(
        "goose_local",
        task,
        workspace,
        role_instruction=(
            "Act as a repository analyst. Prefer inspection and evidence. Do not modify files unless the "
            "task explicitly requires implementation. Distinguish facts from recommendations."
        ),
    )


@mcp.tool()
async def quick_patch(task: str, workspace: str) -> dict[str, Any]:
    """Use Aider, defaulting to local Ollama, for a focused patch."""
    return await run_agent(
        "aider_patch",
        task,
        workspace,
        role_instruction="Make a focused patch only. Keep scope tight and verify the edited files.",
    )


@mcp.tool()
async def opencode_engineer(task: str, workspace: str) -> dict[str, Any]:
    """Delegate repository implementation to OpenCode using the configured local model by default."""
    return await run_agent("opencode_engineer", task, workspace)


@mcp.tool()
async def codex_engineer(task: str, workspace: str) -> dict[str, Any]:
    """Escalate a difficult coding task to the installed Codex CLI."""
    return await run_agent("codex_engineer", task, workspace)


@mcp.tool()
async def claude_architect(task: str, workspace: str) -> dict[str, Any]:
    """Escalate architecture/reasoning work to the installed Claude CLI."""
    return await run_agent("claude_architect", task, workspace)


@mcp.tool()
async def gemini_reviewer(task: str, workspace: str) -> dict[str, Any]:
    """Use the installed Gemini CLI for a large-context review or alternative solution."""
    return await run_agent("gemini_reviewer", task, workspace)


@mcp.tool()
async def aoe_delegate(task: str, workspace: str) -> dict[str, Any]:
    """Send a task to an explicitly configured existing Agent of Empires session."""
    return await run_agent("aoe_delegate", task, workspace)
