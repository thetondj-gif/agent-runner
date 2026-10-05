#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import signal
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import agent_dm_mcp as dm

HEARTBEAT = dm.OPERATOR_HEARTBEAT
LOG = Path.home() / ".agent-dm" / "operator.log"
EXECUTORS = [
    "workspace:01-delivery-director",
    "workspace:deus-intus-founder-os",
    "workspace:02-ai-automation-and-integration",
]
VERIFIER = "workspace:04-product-engineering-and-qa"
POLL_SECONDS = 5
MAX_RECEIPT_AGE_SECONDS = 1200
RESTART_GRACE_SECONDS = 60
INSTANCE_ID = uuid.uuid4().hex[:12]
RUNNING = True

# The persistent worker is asynchronous, so a long local-model turn must not
# block the control loop. This only affects this worker process.
dm.CHAT_TIMEOUT = max(dm.CHAT_TIMEOUT, 900)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{utc_now()} {message}\n")


def parse_marker(text: str, marker: str) -> dict[str, Any] | None:
    """Extract the first JSON object after marker, tolerating markdown wrappers."""
    if not isinstance(text, str):
        return None
    pos = text.rfind(marker)
    if pos < 0:
        return None
    tail = text[pos + len(marker):]
    start = tail.find("{")
    end = tail.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(tail[start:end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def executor_for_attempt(attempt: int) -> str:
    idx = max(0, min(int(attempt or 1) - 1, len(EXECUTORS) - 1))
    return EXECUTORS[idx]


def _update(mission_id: str, **payload: Any) -> dict[str, Any]:
    return dm._operator_request("update", {"id": mission_id, **payload}, timeout=30)


def _events(mission: dict[str, Any]) -> list[dict[str, Any]]:
    runtime = mission.get("runtime") or {}
    rows = runtime.get("progress") or []
    return [x for x in rows if isinstance(x, dict)]


def _latest_event(mission: dict[str, Any], event_type: str, pass_no: int | None = None) -> dict[str, Any] | None:
    for event in reversed(_events(mission)):
        if event.get("type") != event_type:
            continue
        if pass_no is not None and int(event.get("pass") or 0) != pass_no:
            continue
        return event
    return None


def _age_seconds(timestamp: str | None) -> float:
    if not timestamp:
        return 10**9
    try:
        dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except Exception:
        return 10**9


def _executor_prompt(mission: dict[str, Any]) -> str:
    runtime = mission.get("runtime") or {}
    criteria = runtime.get("acceptance_criteria") or []
    criteria_text = "\n".join("- " + str(c) for c in criteria) if criteria else "- Satisfy the stated objective with concrete evidence."
    return f"""You are the execution lead inside the Deus Intus persistent operator.

MISSION ID: {mission['id']}
MISSION APPROVAL: {mission.get('approval', 'GRANTED')} — this low-risk mission has already passed the operator approval gate.
OBJECTIVE:
{mission['objective']}

ACCEPTANCE CRITERIA:
{criteria_text}

OPERATING RULES:
- Do the work now with the tools/capabilities already available to you. Do not return a plan instead of execution.
- Reuse existing workflows, MCP tools, repos, skills and services before creating anything new.
- Keep changes bounded to this mission.
- Do not ask the user to approve ordinary read-only or reversible in-scope work again when MISSION APPROVAL is GRANTED.
- Do not perform an external/public send, financial transaction, destructive live-data action, account/permission change, or irreversible action. If one becomes necessary, stop and return NEEDS_APPROVAL.
- Verify concrete outputs as you work.
- If blocked, identify the exact blocker and whether another existing route can be tried.
- Your final line MUST be exactly:
OPERATOR_RESULT {{"status":"COMPLETE|RETRY|NEEDS_APPROVAL","summary":"short factual summary","evidence":["specific evidence"],"next":"optional next action or blocker"}}
"""


def _verifier_prompt(mission: dict[str, Any], execution: dict[str, Any], pass_no: int) -> str:
    return f"""You are independent QA pass {pass_no} for a persistent Deus Intus mission.

MISSION ID: {mission['id']}
OBJECTIVE:
{mission['objective']}

EXECUTOR SUMMARY:
{execution.get('summary','')}

CLAIMED EVIDENCE:
{json.dumps(execution.get('evidence') or [], ensure_ascii=False)}

VERIFY ONLY. Do not modify files or state. Use available read/test/browser/QA capabilities where appropriate.
Pass only if the objective is materially complete and the evidence is real.
Your final line MUST be exactly:
VERIFIER_RESULT {{"pass":true,"summary":"short factual verdict","evidence":["checks performed"]}}
or
VERIFIER_RESULT {{"pass":false,"summary":"what failed","evidence":["checks performed"]}}
"""


def _dispatch(mission: dict[str, Any]) -> None:
    runtime = mission.get("runtime") or {}
    attempt = int(runtime.get("attempts") or 1)
    executor = executor_for_attempt(attempt)
    result = dm._do_send(executor, _executor_prompt(mission), agent=True, wait=False)
    if result.get("status") != "accepted" or not result.get("receipt_id"):
        _retry_or_block(mission, f"dispatch failed via {executor}: {result.get('error') or result.get('status')}")
        return
    receipt_id = result["receipt_id"]
    submitted_at = utc_now()
    _update(
        mission["id"],
        stage="RUNNING",
        event={
            "type": "EXECUTION_DISPATCHED",
            "receipt_id": receipt_id,
            "executor": executor,
            "attempt": attempt,
            "submitted_at": submitted_at,
            "worker_instance": INSTANCE_ID,
        },
    )
    log(f"dispatch {mission['id']} {executor} receipt={receipt_id}")


def _dispatch_verifier(mission: dict[str, Any], execution: dict[str, Any], pass_no: int) -> None:
    result = dm._do_send(VERIFIER, _verifier_prompt(mission, execution, pass_no), agent=True, wait=False)
    if result.get("status") != "accepted" or not result.get("receipt_id"):
        _retry_or_block(mission, f"verifier dispatch {pass_no} failed: {result.get('error') or result.get('status')}", execution.get("evidence") or [])
        return
    stage = f"VERIFYING_{pass_no}"
    _update(
        mission["id"],
        stage=stage,
        event={
            "type": "VERIFIER_DISPATCHED",
            "pass": pass_no,
            "receipt_id": result["receipt_id"],
            "submitted_at": utc_now(),
            "worker_instance": INSTANCE_ID,
            "execution_summary": execution.get("summary"),
            "execution_evidence": execution.get("evidence") or [],
        },
    )
    log(f"verify-dispatch {mission['id']} pass={pass_no} receipt={result['receipt_id']}")


def _retry_or_block(mission: dict[str, Any], reason: str, evidence: list[Any] | None = None) -> None:
    attempts = int((mission.get("runtime") or {}).get("attempts") or 0)
    stage = "BLOCKED" if attempts >= len(EXECUTORS) else "RETRY"
    _update(
        mission["id"],
        stage=stage,
        last_error=reason,
        evidence=evidence or [],
        event={
            "type": "EXECUTION_BLOCKED" if stage == "BLOCKED" else "EXECUTION_RETRY",
            "message": reason,
        },
    )
    log(f"{stage.lower()} {mission['id']} {reason}")


def _receipt_text(receipt_id: str) -> tuple[str | None, dict[str, Any]]:
    receipt = dm._do_receipt(receipt_id)
    stored = receipt.get("result") if isinstance(receipt, dict) else None
    text = stored.get("text") if isinstance(stored, dict) else None
    return (text if isinstance(text, str) and text.strip() else None, receipt)


def _poll_execution(mission: dict[str, Any]) -> None:
    event = _latest_event(mission, "EXECUTION_DISPATCHED")
    if not event:
        _retry_or_block(mission, "RUNNING mission has no persisted execution receipt")
        return
    receipt_id = str(event.get("receipt_id") or "")
    text, receipt = _receipt_text(receipt_id)
    if text:
        execution = parse_marker(text, "OPERATOR_RESULT")
        if not execution:
            _retry_or_block(mission, "executor finished without a valid OPERATOR_RESULT marker")
            return
        status = str(execution.get("status") or "").upper()
        evidence = execution.get("evidence") if isinstance(execution.get("evidence"), list) else []
        if status not in {"COMPLETE", "RETRY", "NEEDS_APPROVAL"}:
            detail = " ".join(
                str(execution.get(k) or "") for k in ("status", "summary", "next")
            ).lower()
            status = "NEEDS_APPROVAL" if "approval" in detail else "RETRY"
        if status == "NEEDS_APPROVAL":
            _update(
                mission["id"],
                stage="AWAITING_APPROVAL",
                approval="PENDING",
                evidence=evidence,
                event={
                    "type": "APPROVAL_REQUIRED",
                    "message": execution.get("next") or execution.get("summary") or "Approval required",
                },
            )
            log(f"approval {mission['id']}")
            return
        if status == "COMPLETE":
            _dispatch_verifier(mission, execution, 1)
            return
        _retry_or_block(mission, execution.get("next") or execution.get("summary") or "executor requested retry", evidence)
        return

    status = str(receipt.get("status") or "")
    age = _age_seconds(event.get("submitted_at"))
    if (
        status == "running_or_lost"
        and event.get("worker_instance") != INSTANCE_ID
        and age > RESTART_GRACE_SECONDS
    ):
        _retry_or_block(
            mission,
            f"worker restarted before execution receipt {receipt_id} completed; requeueing durable mission",
        )
        return
    if status in {"accepted", "running_or_lost"} and age <= MAX_RECEIPT_AGE_SECONDS:
        return
    if age <= MAX_RECEIPT_AGE_SECONDS and status == "not_found":
        return
    _retry_or_block(mission, f"execution receipt {receipt_id} ended as {status or 'unknown'} after {int(age)}s")


def _poll_verifier(mission: dict[str, Any], pass_no: int) -> None:
    event = _latest_event(mission, "VERIFIER_DISPATCHED", pass_no)
    if not event:
        _retry_or_block(mission, f"VERIFYING_{pass_no} mission has no verifier receipt")
        return
    receipt_id = str(event.get("receipt_id") or "")
    text, receipt = _receipt_text(receipt_id)
    execution = {
        "summary": event.get("execution_summary") or "",
        "evidence": event.get("execution_evidence") or [],
    }
    if text:
        verdict = parse_marker(text, "VERIFIER_RESULT")
        if not verdict or verdict.get("pass") is not True:
            reason = (verdict or {}).get("summary") or f"verifier pass {pass_no} failed"
            evidence = list(execution["evidence"])
            if isinstance((verdict or {}).get("evidence"), list):
                evidence.extend(verdict["evidence"])
            _retry_or_block(mission, reason, evidence)
            return
        evidence = list(execution["evidence"])
        if isinstance(verdict.get("evidence"), list):
            evidence.extend(verdict["evidence"])
        _update(
            mission["id"],
            stage=f"VERIFIED_{pass_no}",
            evidence=evidence,
            event={"type": "VERIFICATION_PASS", "pass": pass_no, "summary": verdict.get("summary")},
        )
        if pass_no == 1:
            execution["evidence"] = evidence
            _dispatch_verifier(mission, execution, 2)
        else:
            _update(
                mission["id"],
                stage="CLOSED",
                status="CLOSED",
                closed=True,
                evidence=evidence,
                last_error=None,
                event={"type": "MISSION_CLOSED", "summary": execution.get("summary")},
            )
            log(f"closed {mission['id']}")
        return

    status = str(receipt.get("status") or "")
    age = _age_seconds(event.get("submitted_at"))
    if (
        status == "running_or_lost"
        and event.get("worker_instance") != INSTANCE_ID
        and age > RESTART_GRACE_SECONDS
    ):
        _retry_or_block(
            mission,
            f"worker restarted before verifier receipt {receipt_id} completed; requeueing durable mission",
            list(execution["evidence"]),
        )
        return
    if status in {"accepted", "running_or_lost"} and age <= MAX_RECEIPT_AGE_SECONDS:
        return
    if age <= MAX_RECEIPT_AGE_SECONDS and status == "not_found":
        return
    _retry_or_block(mission, f"verifier receipt {receipt_id} ended as {status or 'unknown'} after {int(age)}s", list(execution["evidence"]))


def _active_mission(missions: list[dict[str, Any]]) -> dict[str, Any] | None:
    active_stages = {"RUNNING", "VERIFYING_1", "VERIFYING_2"}
    for mission in missions:
        if mission.get("status") == "OPEN" and mission.get("approval") == "GRANTED" and mission.get("runtime_stage") in active_stages:
            return mission
    return None


def run_once() -> bool:
    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT.write_text(utc_now(), encoding="utf-8")

    listed = dm._operator_request("list", {"limit": 50}, timeout=20)
    if not listed.get("ok"):
        log(f"api list error: {listed.get('error')}")
        return False

    active = _active_mission(listed.get("missions") or [])
    if active:
        stage = active.get("runtime_stage")
        if stage == "RUNNING":
            _poll_execution(active)
        elif stage == "VERIFYING_1":
            _poll_verifier(active, 1)
        elif stage == "VERIFYING_2":
            _poll_verifier(active, 2)
        return True

    claimed = dm._operator_request("next", {}, timeout=20)
    if not claimed.get("ok"):
        log(f"api next error: {claimed.get('error')}")
        return False
    mission = claimed.get("mission")
    if not isinstance(mission, dict):
        return False
    _dispatch(mission)
    return True


def _stop(*_: Any) -> None:
    global RUNNING
    RUNNING = False


def main() -> None:
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    log("operator-v2 started")
    while RUNNING:
        try:
            worked = run_once()
        except Exception as exc:
            log(f"loop error: {type(exc).__name__}: {exc}")
            worked = False
        time.sleep(1 if worked else POLL_SECONDS)
    log("operator-v2 stopped")


if __name__ == "__main__":
    main()
