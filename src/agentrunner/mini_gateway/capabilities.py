from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from agentrunner.mini_gateway.executor import agent_status, run_agent


@dataclass(frozen=True)
class CapabilitySpec:
    code: str
    category: str
    description: str
    primary_agent: str
    fallback_agents: tuple[str, ...]
    role_instruction: str
    mutates_workspace: bool = False
    requires_approval: bool = False
    verification_required: bool = True


# Capability names mirror DAWN's capability-first architecture while execution
# remains deliberately local-first. The gateway selects an implementation shell;
# it does not bind business capabilities to a paid model vendor.
_CAPABILITIES: tuple[CapabilitySpec, ...] = (
    CapabilitySpec(
        "plan",
        "orchestration",
        "Decompose founder intent into ordered, testable work without changing files.",
        "goose_local",
        ("opencode_engineer", "gemini_reviewer"),
        "Act as DAWN Mission Planner. Inspect available project evidence first. Break the objective into ordered jobs, dependencies, acceptance criteria, risks, and the cheapest suitable execution route. Do not modify files.",
    ),
    CapabilitySpec(
        "build",
        "engineering",
        "Implement a coherent feature or application change.",
        "opencode_engineer",
        ("goose_local", "aider_patch", "codex_engineer"),
        "Act as implementation engineer. Read project instructions and relevant files before editing. Make the smallest coherent implementation, then run relevant validation.",
        mutates_workspace=True,
    ),
    CapabilitySpec(
        "debug",
        "engineering",
        "Reproduce, diagnose, and fix a defect.",
        "goose_local",
        ("opencode_engineer", "aider_patch", "codex_engineer"),
        "Act as debugger. Reproduce or inspect the failure first, identify root cause, make a minimal fix, and run a regression check.",
        mutates_workspace=True,
    ),
    CapabilitySpec(
        "patch",
        "engineering",
        "Make a small focused code or configuration edit.",
        "aider_patch",
        ("goose_local", "opencode_engineer"),
        "Make a focused patch only. Keep scope tight, preserve existing behaviour outside the request, and verify edited files.",
        mutates_workspace=True,
    ),
    CapabilitySpec(
        "integrate",
        "engineering",
        "Wire APIs, services, databases, repositories, or migrations together.",
        "opencode_engineer",
        ("goose_local", "codex_engineer", "claude_architect"),
        "Act as DAWN Integrator. Inspect both sides of every integration before editing. Validate contracts, environment assumptions, migrations, error paths, and end-to-end handoff. Never invent a connection that has not been proven.",
        mutates_workspace=True,
    ),
    CapabilitySpec(
        "test",
        "verification",
        "Run or improve tests and report evidence.",
        "opencode_engineer",
        ("goose_local",),
        "Act as test engineer. Prefer existing test commands and repository instructions. Add or change tests only when necessary. Report exact commands and failures; do not mask failing tests.",
        mutates_workspace=True,
    ),
    CapabilitySpec(
        "review",
        "verification",
        "Static review for correctness, regressions, maintainability, and style.",
        "opencode_engineer",
        ("goose_local", "gemini_reviewer"),
        "Act as DAWN Reviewer. Review before editing. Prioritise correctness, regressions, maintainability, and evidence. Do not modify files unless the task explicitly asks for fixes.",
    ),
    CapabilitySpec(
        "security_review",
        "verification",
        "Inspect secrets, permissions, unsafe execution, dependency and application risks.",
        "opencode_engineer",
        ("goose_local", "claude_architect"),
        "Act as security reviewer. Inspect trust boundaries, secrets handling, command execution, authentication, authorisation, dependency risk, and external exposure. Prefer concrete evidence and severity-ranked findings. Do not weaken security controls to make a test pass.",
    ),
    CapabilitySpec(
        "research",
        "intelligence",
        "Evidence-led repository or domain research using available project tools.",
        "goose_local",
        ("opencode_engineer", "gemini_reviewer"),
        "Act as evidence-led research analyst. Separate observed facts, assumptions, and recommendations. Use available project data/tools and cite file or command evidence in the result.",
    ),
    CapabilitySpec(
        "market_scout",
        "commercial-intelligence",
        "DAWN-style market signal and opportunity analysis.",
        "goose_local",
        ("opencode_engineer", "gemini_reviewer"),
        "Act as DAWN Market Scout. Analyse available market inputs for material signals, customer pain, competitors, timing, routes to revenue, and next actions. Do not fabricate live market data; identify missing sources explicitly.",
    ),
    CapabilitySpec(
        "tender_scout",
        "commercial-intelligence",
        "Tender/RFP discovery, qualification, and bid-readiness analysis using available data sources.",
        "goose_local",
        ("opencode_engineer", "gemini_reviewer"),
        "Act as DAWN Tender Scout. Use only available tender/RFP data or project integrations. Extract buyer, scope, value, deadlines, eligibility, likely delivery burden, fit, risks, and recommended next action. Never invent a tender.",
    ),
    CapabilitySpec(
        "financial_analysis",
        "commercial-intelligence",
        "Financial, market, unit-economics, or risk analysis without executing financial transactions.",
        "goose_local",
        ("opencode_engineer", "gemini_reviewer"),
        "Act as DAWN Financial Analyst. Analyse supplied or available financial data, assumptions, scenarios, risks, and confidence. Do not execute trades, transfers, purchases, or other financial transactions.",
    ),
    CapabilitySpec(
        "venture_analysis",
        "commercial-intelligence",
        "Venture Foundry-style idea scoring, validation, positioning, and launch analysis.",
        "goose_local",
        ("opencode_engineer", "gemini_reviewer"),
        "Act as DAWN Venture Analyst. Test the idea against user pain, differentiation, buildability, distribution, monetisation, evidence, defensibility, cost, and fastest validation path. Prefer a concrete go/no-go or next experiment.",
    ),
    CapabilitySpec(
        "content_create",
        "creative-ops",
        "Create or prepare copy, scripts, campaign assets, metadata, or content packages using available tools.",
        "goose_local",
        ("opencode_engineer", "gemini_reviewer"),
        "Act as DAWN Content Ops creator. Follow project brand instructions and existing content systems. Create production-ready outputs and distinguish generated assets from placeholders.",
        mutates_workspace=True,
    ),
    CapabilitySpec(
        "asset_generate",
        "creative-ops",
        "Invoke available project image/video/media generation tooling and package outputs.",
        "goose_local",
        ("opencode_engineer",),
        "Act as DAWN Asset Factory operator. Use only generation tools actually available in the workspace. Do not claim an image, video, audio file, or deployment exists unless the tool produced a verifiable artifact.",
        mutates_workspace=True,
    ),
    CapabilitySpec(
        "content_publish",
        "external-action",
        "Publish or schedule approved content through an available project integration.",
        "opencode_engineer",
        ("goose_local",),
        "Act as DAWN Content Publisher. Confirm the target account/channel, final approved asset/copy, scheduling intent, and integration status before making any external write. Record evidence of the external action.",
        mutates_workspace=True,
        requires_approval=True,
    ),
    CapabilitySpec(
        "deploy",
        "external-action",
        "Deploy an approved build through an available deployment integration.",
        "opencode_engineer",
        ("goose_local", "codex_engineer"),
        "Act as deployment engineer. Inspect the repository deployment instructions and target first. Run pre-deploy validation, make the approved deployment, then verify the live result. Do not change unrelated infrastructure.",
        mutates_workspace=True,
        requires_approval=True,
    ),
    CapabilitySpec(
        "verify",
        "verification",
        "Independently verify another agent's claimed completion using tests, files, endpoints, or artifacts.",
        "goose_local",
        ("opencode_engineer",),
        "Act as DAWN Verification Agent. Treat previous success claims as untrusted until proven. Inspect the resulting state, run relevant checks, and return PASS, PARTIAL, or FAIL with exact evidence and remaining blockers. Do not modify files unless verification itself requires a harmless fixture.",
    ),
)


def capability_catalog() -> dict[str, dict[str, Any]]:
    return {spec.code: asdict(spec) for spec in _CAPABILITIES}


def _get_capability(code: str) -> CapabilitySpec:
    for spec in _CAPABILITIES:
        if spec.code == code:
            return spec
    raise ValueError(f"Unknown capability: {code}. Available: {', '.join(spec.code for spec in _CAPABILITIES)}")


def plan_route(code: str, *, allow_paid: bool = False) -> dict[str, Any]:
    spec = _get_capability(code)
    status = agent_status()["agents"]
    candidates = (spec.primary_agent, *spec.fallback_agents)
    considered: list[dict[str, Any]] = []
    selected: str | None = None

    for name in candidates:
        agent = status.get(name, {})
        tier = str(agent.get("tier", ""))
        paid = tier == "quota-or-paid"
        usable = bool(agent.get("ready")) and (allow_paid or not paid)
        considered.append({
            "agent": name,
            "ready": bool(agent.get("ready")),
            "tier": tier,
            "eligible": usable,
        })
        if selected is None and usable:
            selected = name

    return {
        "capability": spec.code,
        "category": spec.category,
        "description": spec.description,
        "selected_agent": selected,
        "candidates": considered,
        "local_first": True,
        "allow_paid": allow_paid,
        "requires_approval": spec.requires_approval,
        "mutates_workspace": spec.mutates_workspace,
        "verification_required": spec.verification_required,
    }


async def execute_capability(
    code: str,
    task: str,
    workspace: str,
    *,
    approved: bool = False,
    allow_paid: bool = False,
) -> dict[str, Any]:
    spec = _get_capability(code)
    route = plan_route(code, allow_paid=allow_paid)

    if spec.requires_approval and not approved:
        return {
            "success": False,
            "status": "approval_required",
            "capability": code,
            "route": route,
            "message": "This DAWN capability is an external-action boundary. Re-run with approved=true only after the founder has approved the exact action.",
        }

    selected = route["selected_agent"]
    if not selected:
        return {
            "success": False,
            "status": "no_ready_agent",
            "capability": code,
            "route": route,
        }

    verification_contract = (
        "\n\nVERIFICATION CONTRACT:\n"
        "Do not claim completion without evidence. Report files changed, commands/tests run, outputs or artifacts created, and any remaining uncertainty. "
        "If an external service or data source is unavailable, say so rather than simulating success."
    )
    execution = await run_agent(
        selected,
        task,
        workspace,
        role_instruction=spec.role_instruction + verification_contract,
    )
    return {
        "success": bool(execution.get("success")),
        "status": "completed" if execution.get("success") else "execution_failed",
        "capability": code,
        "route": route,
        "governance": {
            "approval_required": spec.requires_approval,
            "approved": approved,
            "external_action": spec.category == "external-action",
        },
        "verification": {
            "required": spec.verification_required,
            "independent_verify_capability": "verify" if code != "verify" else None,
        },
        "execution": execution,
    }
