from pathlib import Path

import pytest

from agentrunner.mini_gateway import executor
from agentrunner.mini_gateway.capabilities import capability_catalog, plan_route


def test_status_has_proven_goose_route() -> None:
    status = executor.agent_status()
    assert status["default"] == "goose_local"
    assert "goose_local" in status["agents"]
    assert status["agents"]["goose_local"]["proven_on_mini"] is True
    assert status["agents"]["goose_local"]["tier"] == "free-local"


def test_workspace_allowlist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    child = allowed / "repo"
    outside = tmp_path / "outside"
    child.mkdir(parents=True)
    outside.mkdir()
    monkeypatch.setenv("MINI_AGENT_WORKSPACE_ROOTS", str(allowed))

    assert executor.resolve_workspace(str(child)) == child.resolve()
    with pytest.raises(ValueError, match="outside allowed roots"):
        executor.resolve_workspace(str(outside))


def test_goose_task_is_single_argv_element(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOSE_MINI_PROVIDER", "custom_ollama2")
    monkeypatch.setenv("GOOSE_MINI_MODEL", "qwen2.5:7b")
    spec = executor._agent_specs()["goose_local"]
    task = "create file; rm -rf / should remain task text"
    argv = executor._render_argv(spec, task)

    assert argv[-1] == task
    assert argv[0] == "goose"
    assert "rm -rf /" in argv[-1]


def test_opencode_defaults_to_local_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCODE_MINI_MODEL", raising=False)
    spec = executor._agent_specs()["opencode_engineer"]
    assert "ollama/qwen2.5:7b" in spec.argv
    assert spec.tier == "free-local-default"


def test_dawn_capability_catalog_covers_core_domains() -> None:
    catalog = capability_catalog()
    expected = {
        "plan",
        "build",
        "debug",
        "integrate",
        "test",
        "review",
        "security_review",
        "research",
        "market_scout",
        "tender_scout",
        "financial_analysis",
        "venture_analysis",
        "content_create",
        "asset_generate",
        "content_publish",
        "deploy",
        "verify",
    }
    assert expected.issubset(catalog)
    assert catalog["content_publish"]["requires_approval"] is True
    assert catalog["deploy"]["requires_approval"] is True


def test_route_does_not_select_paid_worker_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentrunner.mini_gateway.capabilities.agent_status",
        lambda: {
            "agents": {
                "opencode_engineer": {"ready": False, "tier": "free-local-default"},
                "goose_local": {"ready": False, "tier": "free-local"},
                "aider_patch": {"ready": False, "tier": "free-local-default"},
                "codex_engineer": {"ready": True, "tier": "quota-or-paid"},
            }
        },
    )
    route = plan_route("build")
    assert route["selected_agent"] is None
    paid_route = plan_route("build", allow_paid=True)
    assert paid_route["selected_agent"] == "codex_engineer"
