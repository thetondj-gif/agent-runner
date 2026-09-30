from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agentrunner.operator_worker import OperatorConfig, PersistentOperator


def config(tmp_path: Path) -> OperatorConfig:
    return OperatorConfig(
        endpoint="https://example.invalid/operator-tasks",
        token="test-token",
        default_workspace=tmp_path,
        allowed_root=tmp_path,
        model="local-model",
        enabled=True,
        stale_seconds=60,
        max_attempts=3,
    )


def test_workspace_must_stay_inside_allowed_root(tmp_path: Path) -> None:
    worker = PersistentOperator(config(tmp_path), tasks=AsyncMock())
    mission = {"id": "OP-1", "runtime": {"workspace_path": str(tmp_path.parent)}}
    with pytest.raises(RuntimeError, match="outside allowed root"):
        worker._resolve_workspace(mission)


@pytest.mark.asyncio
async def test_stale_running_mission_is_requeued(tmp_path: Path) -> None:
    tasks = AsyncMock()
    old = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    tasks.list.return_value = [
        {
            "id": "OP-STALE",
            "runtime_stage": "RUNNING",
            "runtime": {"updated_at": old},
        }
    ]
    worker = PersistentOperator(config(tmp_path), tasks=tasks)

    recovered = await worker.recover_stale()

    assert recovered == 1
    tasks.update.assert_awaited_once()
    assert tasks.update.await_args.kwargs["stage"] == "RETRY"


@pytest.mark.asyncio
async def test_disallowed_risk_blocks_without_execution(tmp_path: Path) -> None:
    tasks = AsyncMock()
    worker = PersistentOperator(config(tmp_path), tasks=tasks)
    worker._run_agent = AsyncMock()  # type: ignore[method-assign]
    mission = {
        "id": "OP-HIGH",
        "risk": "high",
        "runtime": {"attempts": 1},
    }

    await worker.execute_claimed(mission)

    worker._run_agent.assert_not_awaited()
    assert tasks.update.await_args.kwargs["stage"] == "BLOCKED"
    assert tasks.update.await_args.kwargs["approval"] == "REQUIRED"


@pytest.mark.asyncio
async def test_failed_verification_retries(tmp_path: Path) -> None:
    tasks = AsyncMock()
    worker = PersistentOperator(config(tmp_path), tasks=tasks)
    worker._run_agent = AsyncMock(return_value=("work done", "evidence\nVERDICT: FAIL"))  # type: ignore[method-assign]
    mission = {
        "id": "OP-RETRY",
        "risk": "low",
        "objective": "test objective",
        "runtime": {"attempts": 1, "workspace_path": str(tmp_path)},
    }

    await worker.execute_claimed(mission)

    stages = [call.kwargs.get("stage") for call in tasks.update.await_args_list]
    assert stages[-1] == "RETRY"


@pytest.mark.asyncio
async def test_verified_mission_closes(tmp_path: Path) -> None:
    tasks = AsyncMock()
    worker = PersistentOperator(config(tmp_path), tasks=tasks)
    worker._run_agent = AsyncMock(return_value=("work done", "real checks ran\nVERDICT: PASS"))  # type: ignore[method-assign]
    mission = {
        "id": "OP-DONE",
        "risk": "low",
        "objective": "test objective",
        "runtime": {"attempts": 1, "workspace_path": str(tmp_path)},
    }

    await worker.execute_claimed(mission)

    last = tasks.update.await_args_list[-1].kwargs
    assert last["stage"] == "COMPLETE"
    assert last["status"] == "CLOSED"
    assert last["closed"] is True
