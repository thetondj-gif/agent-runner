# Agent Runner Mini Gateway — Agent Instructions

These instructions apply to Devin and every coding agent working in this repository.

## Read before changing code

1. Read `README.md`, `MINI_GATEWAY.md`, this file, and the relevant implementation/tests before editing.
2. Inspect existing behaviour before replacing or redesigning it.
3. Prefer the smallest coherent change that preserves working behaviour.
4. Never claim a service, integration, model, endpoint, file, deployment, test, or external action works without evidence.
5. Do not introduce fake, simulated, placeholder, or hard-coded production data as if it were live data.

## Architecture

Devin is the top-level software orchestrator. The Mac Mini gateway is a specialist worker pool.

Use DAWN-style **capabilities**, not vendor/model names, as the public interface:

- `plan`
- `build`
- `debug`
- `patch`
- `integrate`
- `test`
- `review`
- `security_review`
- `research`
- `market_scout`
- `tender_scout`
- `financial_analysis`
- `venture_analysis`
- `content_create`
- `asset_generate`
- `content_publish`
- `deploy`
- `verify`

The canonical registry is `src/agentrunner/mini_gateway/capabilities.py`.

## Routing policy

Default to free/local execution:

1. Goose + Ollama
2. Aider + Ollama
3. OpenCode + Ollama
4. Agent of Empires coordination when a multi-agent session is intentionally configured
5. Codex / Claude / Gemini only as explicit escalation

Do not silently use a paid/quota agent. `allow_paid` must be explicit.

## Governance

- Workspace access is restricted by `MINI_AGENT_WORKSPACE_ROOTS`.
- `content_publish` and `deploy` cross an external-action boundary and require explicit approval.
- Financial analysis is analysis only; this gateway must not execute trades, transfers, purchases, or financial transactions.
- Do not expose unrestricted shell access as a public MCP tool.
- Do not widen workspace roots to the entire home directory merely to bypass an access error.

## Verification

Every mutating task should report:

- files changed;
- commands/tests run;
- exit status or observed result;
- artifacts/endpoints created or changed;
- remaining uncertainty or blockers.

Use the `verify` capability for an independent completion check when the result matters.

## DAWN relationship

DAWN contributes capability definitions, mission-planning behaviour, integration/reviewer roles, governance boundaries, and evidence-first verification. It does not need to own every executor. The gateway intentionally keeps model/provider selection underneath the capability layer.
