# Deus Agent DM Persistent Operator

This directory snapshots the **canonical Studio production operator** that owns durable Deus missions and delegates execution to existing Agent DM specialists.

It deliberately does not replace Agent DM, Supabase, n8n, MCP, existing agents, or project repositories.

## Runtime contract

The worker uses the existing Supabase `operator-tasks` Edge Function and `public.missions` state machine.

Normal flow:

```text
QUEUED / RETRY
  -> RUNNING
  -> VERIFYING_1
  -> VERIFYING_2
  -> CLOSED
```

Recoverable executor or verifier failures return the mission to `RETRY`. The next claim increments the attempt count and rotates to the next configured executor. When the executor budget is exhausted, the mission becomes `BLOCKED` rather than looping forever.

External/public sends, financial actions, destructive live-data changes, account/permission changes, and irreversible actions are required to stop at `AWAITING_APPROVAL`.

## Crash / restart behaviour

Every dispatch stores its receipt ID in the mission's durable runtime progress before the worker continues.

The launchd service uses `RunAtLoad` and `KeepAlive`, so the worker restarts after process failure or login/boot. Each process has a `worker_instance` ID. If the process restarts while an in-process asynchronous Agent DM receipt is still `running_or_lost`, the replacement worker waits a short grace period and returns the mission to `RETRY` instead of abandoning it until the full receipt timeout.

The verifier runs twice and must produce structured PASS evidence before the mission can close.

## Studio deployment

Canonical paths currently used by the Studio:

- runtime: `~/studio-agent-dm/persistent_operator.py`
- environment: `~/.agent-dm/.env`
- Python environment: `~/agent-dm-venv`
- launchd label: `com.deusintus.persistent-operator`
- operator log: `~/.agent-dm/operator.log`

The example launchd file in this directory contains placeholders rather than secrets or account-specific paths.

## Important: one production consumer

Do **not** simultaneously enable the generic `deus-operator` sidecar from `src/agentrunner/operator_worker.py` against the same production `operator-tasks` queue while this Agent DM operator is active.

The Agent DM worker is the canonical production route because it delegates to the estate's existing specialist agents and performs two independent QA passes. The generic Agent Runner sidecar is retained as a local-model/fallback implementation and testable reference path.

A multi-worker/HA deployment would need explicit worker leases and idempotency semantics for external side effects before enabling more than one production consumer.

## Verification

On the Studio:

```bash
python -m py_compile persistent_operator.py
python test_persistent_operator_v2.py
launchctl print gui/$(id -u)/com.deusintus.persistent-operator
```

A process restart test must preserve the mission in Supabase and either resume a completed stored receipt or move an orphaned in-flight receipt to `RETRY`.
