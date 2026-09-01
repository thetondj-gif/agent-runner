"""Thin execution contract between compiled configuration and live verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Universal Execution Contract",
    "type": "object",
    "required": [
        "compiled_manifest",
        "target_environment_profile",
        "connection_requirements",
        "startup_tests",
        "capability_tests",
        "rollback_plan",
    ],
    "properties": {
        "compiled_manifest": {"type": "object"},
        "target_environment_profile": {"type": "object"},
        "connection_requirements": {"type": "array", "items": {"type": "object"}},
        "startup_tests": {"type": "array", "items": {"$ref": "#/$defs/test"}},
        "capability_tests": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/test"},
        },
        "rollback_plan": {"type": "object"},
    },
    "additionalProperties": False,
    "$defs": {
        "test": {
            "type": "object",
            "required": ["name", "command", "expected"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "command": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "expected": {
                    "type": "object",
                    "properties": {
                        "exit_code": {"type": "integer"},
                        "contains": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                "timeout_s": {"type": "number", "exclusiveMinimum": 0},
            },
            "additionalProperties": False,
        }
    },
}

_REQUIRED_FIELDS = tuple(CONTRACT_SCHEMA["required"])
_SECRET_KEY = re.compile(r"(^|_)(secret|password|token|api_key|private_key)($|_)", re.I)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _receipt_hash(receipt: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    return content_hash(unsigned)


def _assert_no_raw_secrets(value: Any, path: str = "connection_requirements") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _SECRET_KEY.search(key) and not key.endswith("_handle"):
                raise ValueError(f"Raw credential field is forbidden at {path}.{key}")
            _assert_no_raw_secrets(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_raw_secrets(item, f"{path}[{index}]")


def validate_contract(contract: dict[str, Any]) -> None:
    missing = [field for field in _REQUIRED_FIELDS if field not in contract]
    if missing:
        raise ValueError(f"Execution contract is missing: {', '.join(missing)}")
    if not contract["capability_tests"]:
        raise ValueError("Execution contract requires at least one capability test")
    _assert_no_raw_secrets(contract["connection_requirements"])
    _assert_no_raw_secrets(
        contract["target_environment_profile"].get("environment", {}), "environment"
    )


class CommandExecutionAdapter:
    """Runs compiled argv commands while leaving credential resolution external."""

    def __init__(self, contract: dict[str, Any]) -> None:
        validate_contract(contract)
        self.contract = contract
        profile = contract["target_environment_profile"]
        self.cwd = profile.get("working_directory")
        self.env = os.environ.copy()
        self.env.update(profile.get("environment", {}))
        self._process: subprocess.Popen[str] | None = None

    def _run(self, command: list[str], timeout_s: float = 30) -> dict[str, Any]:
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=self.cwd,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-20_000:],
            "stderr": completed.stderr[-4_000:],
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    @staticmethod
    def _passed(observed: dict[str, Any], expected: dict[str, Any]) -> bool:
        if observed["exit_code"] != expected.get("exit_code", 0):
            return False
        contains = expected.get("contains")
        return contains is None or contains in observed["stdout"]

    def apply_configuration(self) -> dict[str, Any]:
        command = self.contract["compiled_manifest"].get("apply_command")
        if not command:
            return {"state": "already_configured", "passed": True}
        observed = self._run(command, self.contract["compiled_manifest"].get("timeout_s", 30))
        observed["state"] = "configured" if observed["exit_code"] == 0 else "apply_failed"
        observed["passed"] = observed["exit_code"] == 0
        return observed

    def start_or_attach(self) -> dict[str, Any]:
        profile = self.contract["target_environment_profile"]
        attach = profile.get("attach_command")
        if attach:
            observed = self._run(attach, profile.get("attach_timeout_s", 10))
            if observed["exit_code"] == 0:
                observed.update({"state": "attached", "passed": True})
                return observed
        start = profile.get("start_command")
        if not start:
            return {"state": "attach_failed", "passed": False}
        self._process = subprocess.Popen(
            start,
            cwd=self.cwd,
            env=self.env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return {"state": "started", "pid": self._process.pid, "passed": True}

    def connection_state(self) -> list[dict[str, Any]]:
        states = []
        for requirement in self.contract["connection_requirements"]:
            observed = self._run(requirement["state_command"], requirement.get("timeout_s", 10))
            connected = observed["exit_code"] == 0
            states.append(
                {
                    "handle": requirement["connection_handle"],
                    "authenticated": connected
                    if requirement.get("auth_required", True)
                    else "not_required",
                    "connected": connected,
                    "latency_ms": observed["latency_ms"],
                }
            )
        return states

    def run_tests(self, tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for test in tests:
            observed = self._run(test["command"], test.get("timeout_s", 30))
            observed.update(
                {
                    "name": test["name"],
                    "expected": test["expected"],
                    "passed": self._passed(observed, test["expected"]),
                }
            )
            results.append(observed)
        return results

    def executable_version(self) -> dict[str, str]:
        manifest = self.contract["compiled_manifest"]
        observed = self._run(manifest["version_command"], manifest.get("timeout_s", 30))
        return {
            "executable": manifest["executable"],
            "version": observed["stdout"].strip().splitlines()[-1],
        }

    def artifacts(self) -> list[dict[str, str]]:
        artifacts = []
        for value in self.contract["compiled_manifest"].get("artifact_paths", []):
            path = Path(value)
            if path.is_file():
                artifacts.append(
                    {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                )
        return artifacts

    def rollback(self) -> dict[str, Any]:
        command = self.contract["rollback_plan"].get("command")
        if command:
            observed = self._run(command, self.contract["rollback_plan"].get("timeout_s", 30))
            observed["state"] = "rolled_back" if observed["exit_code"] == 0 else "rollback_failed"
        else:
            observed = {"state": "not_available"}
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=10)
            observed["started_process_stopped"] = True
        return observed


def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    text = "\n".join(item.get("stdout", "") for item in results)
    prompt = re.findall(r'"prompt_tokens"\s*:\s*(\d+)', text)
    completion = re.findall(r'"completion_tokens"\s*:\s*(\d+)', text)
    cost = re.findall(r"x-omniroute-response-cost:\s*([0-9.]+)", text, re.I)
    return {
        "runtime_ms": round(sum(item.get("latency_ms", 0) for item in results), 3),
        "tokens": {
            "input": int(prompt[-1]) if prompt else None,
            "output": int(completion[-1]) if completion else None,
        },
        "cost": float(cost[-1]) if cost else None,
    }


def write_receipt(path: str | Path, receipt: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


class LiveMachineExecutor:
    def execute(
        self, contract: dict[str, Any], adapter: CommandExecutionAdapter, receipt_path: str | Path
    ) -> dict[str, Any]:
        validate_contract(contract)
        applied = adapter.apply_configuration()
        target = (
            adapter.start_or_attach()
            if applied["passed"]
            else {"state": "not_started", "passed": False}
        )
        connections = adapter.connection_state() if target["passed"] else []
        connected = all(item["connected"] for item in connections)
        startup = adapter.run_tests(contract["startup_tests"]) if connected else []
        startup_passed = all(item["passed"] for item in startup)
        capability = adapter.run_tests(contract["capability_tests"]) if startup_passed else []
        passed = (
            applied["passed"]
            and target["passed"]
            and connected
            and startup_passed
            and all(item["passed"] for item in capability)
        )
        rollback = {"state": "available_not_used"} if passed else adapter.rollback()
        receipt = {
            "contract_version": "1.0",
            "manifest_hash": content_hash(contract["compiled_manifest"]),
            "environment_hash": content_hash(contract["target_environment_profile"]),
            **adapter.executable_version(),
            "timestamp": datetime.now(UTC).isoformat(),
            "auth_connectivity_state": connections,
            "exact_canary": [test["command"] for test in contract["capability_tests"]],
            "expected_result": [test["expected"] for test in contract["capability_tests"]],
            "observed_result": capability,
            "latency_ms": round(sum(item.get("latency_ms", 0) for item in startup + capability), 3),
            "runtime_tokens_cost": _metrics(capability),
            "artifacts": adapter.artifacts(),
            "apply_state": applied,
            "target_state": target,
            "startup_results": startup,
            "rollback_state": rollback,
            "final_status": "EXECUTED"
            if passed
            else "ROLLED_BACK"
            if rollback.get("state") == "rolled_back"
            else "FAILED",
        }
        receipt["receipt_hash"] = _receipt_hash(receipt)
        write_receipt(receipt_path, receipt)
        return receipt


class IndependentVerifier:
    """Observes the canary again; it does not rebuild or apply configuration."""

    def verify(
        self,
        contract: dict[str, Any],
        receipt: dict[str, Any],
        adapter: CommandExecutionAdapter,
        receipt_path: str | Path,
    ) -> dict[str, Any]:
        if receipt.get("receipt_hash") != _receipt_hash(receipt):
            raise ValueError("Execution receipt hash is invalid")
        if receipt.get("final_status") != "EXECUTED":
            return receipt
        observed = adapter.run_tests(contract["capability_tests"])
        verified = all(item["passed"] for item in observed)
        receipt = dict(receipt)
        receipt["independent_observation"] = {
            "verifier": type(self).__name__,
            "timestamp": datetime.now(UTC).isoformat(),
            "observed_result": observed,
            "passed": verified,
        }
        receipt["final_status"] = "VERIFIED" if verified else "NOT_VERIFIED"
        receipt["receipt_hash"] = _receipt_hash(receipt)
        write_receipt(receipt_path, receipt)
        return receipt


def promote_configured(receipt: dict[str, Any] | None) -> str:
    if not receipt:
        raise ValueError("NO EXECUTION RECEIPT = NOT VERIFIED")
    if receipt.get("receipt_hash") != _receipt_hash(receipt):
        raise ValueError("Execution receipt hash is invalid")
    observation = receipt.get("independent_observation", {})
    if receipt.get("final_status") != "VERIFIED" or observation.get("passed") is not True:
        raise ValueError("Only an independently observed successful receipt can promote CONFIGURED")
    return "VERIFIED"


def run_fast_path(contract: dict[str, Any], receipt_path: str | Path) -> dict[str, Any]:
    adapter = CommandExecutionAdapter(contract)
    receipt = LiveMachineExecutor().execute(contract, adapter, receipt_path)
    receipt = IndependentVerifier().verify(contract, receipt, adapter, receipt_path)
    if receipt["final_status"] == "VERIFIED":
        receipt["promotion_state"] = promote_configured(receipt)
        receipt["receipt_hash"] = _receipt_hash(receipt)
        write_receipt(receipt_path, receipt)
    return receipt
