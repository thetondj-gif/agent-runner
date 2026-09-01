import json

import pytest

from agentrunner.core.execution_contract import (
    CommandExecutionAdapter,
    IndependentVerifier,
    LiveMachineExecutor,
    promote_configured,
)


def _contract(tmp_path, expected="CANARY_OK"):
    return {
        "compiled_manifest": {
            "executable": "python",
            "version_command": ["python3", "--version"],
            "apply_command": ["python3", "-c", "print('applied')"],
            "artifact_paths": [],
        },
        "target_environment_profile": {
            "working_directory": str(tmp_path),
            "attach_command": ["python3", "-c", "print('running')"],
        },
        "connection_requirements": [
            {
                "connection_handle": "local-loopback",
                "auth_required": False,
                "state_command": ["python3", "-c", "print('connected')"],
            }
        ],
        "startup_tests": [
            {
                "name": "startup",
                "command": ["python3", "-c", "print('ready')"],
                "expected": {"contains": "ready"},
            }
        ],
        "capability_tests": [
            {
                "name": "canary",
                "command": ["python3", "-c", "print('CANARY_OK')"],
                "expected": {"contains": expected},
            }
        ],
        "rollback_plan": {"command": ["python3", "-c", "print('rollback')"]},
    }


def test_receipt_gates_configured_to_verified(tmp_path):
    contract = _contract(tmp_path)
    adapter = CommandExecutionAdapter(contract)
    path = tmp_path / "receipt.json"

    receipt = LiveMachineExecutor().execute(contract, adapter, path)
    assert receipt["final_status"] == "EXECUTED"
    with pytest.raises(ValueError, match="independently observed"):
        promote_configured(receipt)

    receipt = IndependentVerifier().verify(contract, receipt, adapter, path)
    assert promote_configured(receipt) == "VERIFIED"
    assert json.loads(path.read_text())["receipt_hash"] == receipt["receipt_hash"]


def test_failed_canary_rolls_back_and_cannot_promote(tmp_path):
    contract = _contract(tmp_path, expected="NOT_PRESENT")
    receipt = LiveMachineExecutor().execute(
        contract, CommandExecutionAdapter(contract), tmp_path / "failed.json"
    )

    assert receipt["final_status"] == "ROLLED_BACK"
    assert receipt["rollback_state"]["state"] == "rolled_back"
    with pytest.raises(ValueError, match="independently observed"):
        promote_configured(receipt)


def test_raw_secret_is_rejected(tmp_path):
    contract = _contract(tmp_path)
    contract["connection_requirements"][0]["api_key"] = "raw-secret"

    with pytest.raises(ValueError, match="Raw credential"):
        CommandExecutionAdapter(contract)


def test_no_receipt_is_not_verified():
    with pytest.raises(ValueError, match="NO EXECUTION RECEIPT"):
        promote_configured(None)
