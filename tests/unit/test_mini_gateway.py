from pathlib import Path

import pytest

from agentrunner.mini_gateway import executor


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
