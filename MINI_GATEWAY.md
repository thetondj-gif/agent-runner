# Mac Mini Agent Gateway

This branch adds a controlled specialist-agent gateway to the Design Arena Agent Runner fork.
It is intended to let Devin or another MCP client delegate work to agents installed on the Mac Mini instead of using one paid cloud agent for every task.

## Current routing

| MCP tool | Backend | Default cost tier | Mini proof state |
|---|---|---|---|
| `local_build` | Goose + Ollama `qwen2.5:7b` | free/local | proven write execution |
| `local_debug` | Goose + Ollama `qwen2.5:7b` | free/local | backend proven |
| `local_analysis` | Goose + Ollama `qwen2.5:7b` | free/local | backend proven |
| `quick_patch` | Aider + Ollama | free/local default | adapter needs Mini validation |
| `opencode_engineer` | OpenCode | configured provider | adapter needs Mini validation |
| `aoe_delegate` | Agent of Empires | depends on AoE session | needs `AOE_SESSION` |
| `codex_engineer` | Codex CLI | quota/paid | adapter needs Mini validation |
| `claude_architect` | Claude CLI | quota/paid | adapter needs Mini validation |
| `gemini_reviewer` | Gemini CLI | configured provider | adapter needs Mini validation |

The gateway does **not** claim untested adapters are working. `mini_status` reports installed, ready and `proven_on_mini` separately.

## Security model

- Commands are executed with `asyncio.create_subprocess_exec`; no shell string is evaluated.
- Workspaces are restricted to `MINI_AGENT_WORKSPACE_ROOTS`.
- The default allowed roots are only `~/dawn-v4` and `~/goose-test`.
- The MCP server should bind locally. For Devin access, put an authenticated tunnel/access layer in front of it rather than exposing a Mac shell directly.
- Provider secrets stay in the Mini's normal agent/keychain configuration; the repository does not contain them.

## Install on the Mini

```bash
git clone -b feature/mini-agent-gateway https://github.com/thetondj-gif/agent-runner.git ~/agent-runner-mini
cd ~/agent-runner-mini
bash scripts/install_mini_gateway.sh
```

If the repo already exists locally:

```bash
cd ~/agent-runner-mini
git fetch origin
git switch feature/mini-agent-gateway
git pull
bash scripts/install_mini_gateway.sh
```

## Doctor

```bash
source ~/agent-runner-mini/.venv-mini-gateway/bin/activate
python ~/agent-runner-mini/scripts/mini_gateway_doctor.py
```

Expected proven local route on this Mini:

- `goose` installed
- custom Goose provider `custom_ollama2`
- model `qwen2.5:7b`
- developer tool execution proven by creating `~/goose-test/hello.txt`

## Start MCP locally

The gateway uses the official MCP Python SDK v2 and Streamable HTTP.

```bash
cd ~/agent-runner-mini
source .venv-mini-gateway/bin/activate
mcp run src/agentrunner/mini_gateway/server.py --transport streamable-http
```

The standard local endpoint is:

```text
http://localhost:8000/mcp
```

Test `mini_status` with MCP Inspector before exposing the endpoint remotely.

## Machine configuration

The install helper creates:

```text
~/.config/agentrunner/mini-gateway.env
```

Default contents include:

```bash
GOOSE_MINI_PROVIDER=custom_ollama2
GOOSE_MINI_MODEL=qwen2.5:7b
AIDER_MINI_MODEL=ollama/qwen2.5:7b
MINI_AGENT_WORKSPACE_ROOTS=$HOME/dawn-v4:$HOME/goose-test
```

Add additional repository roots deliberately rather than setting the entire home directory as writable.

To enable AoE delegation, set an existing session name, for example:

```bash
AOE_SESSION=creator-studio-canary
```

Only do this after confirming that session exists with `aoe ps`/`aoe agents`.

## Devin routing instruction

Recommended policy for Devin:

1. Call `mini_status` first when agent availability matters.
2. Prefer `local_build`, `local_debug`, or `local_analysis` for normal work.
3. Use `quick_patch`, OpenCode or AoE after their Mini adapters have been validated.
4. Escalate to Codex/Claude/Gemini only when local execution is unsuitable or has failed validation.
5. Devin remains responsible for final review/integration; a worker's success message is not proof by itself.
