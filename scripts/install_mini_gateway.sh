#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${MINI_GATEWAY_VENV:-$ROOT/.venv-mini-gateway}"

"$PYTHON_BIN" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e '.[gateway]'

CONFIG_DIR="$HOME/.config/agentrunner"
ENV_FILE="$CONFIG_DIR/mini-gateway.env"
mkdir -p "$CONFIG_DIR"

if [[ ! -f "$ENV_FILE" ]]; then
  cat > "$ENV_FILE" <<EOF
# Proven local Goose route on AnthonsMini615
GOOSE_MINI_PROVIDER=custom_ollama2
GOOSE_MINI_MODEL=qwen2.5:7b
AIDER_MINI_MODEL=ollama/qwen2.5:7b

# Restrict remote agents to explicit project roots. Add more roots separated by ':' on macOS.
MINI_AGENT_WORKSPACE_ROOTS=$HOME/dawn-v4:$HOME/goose-test
MINI_AGENT_MAX_OUTPUT_CHARS=60000

# Optional: set this only after choosing an existing AoE session.
# AOE_SESSION=creator-studio-canary
EOF
  chmod 600 "$ENV_FILE"
fi

echo
echo "Mini Agent Gateway installed."
echo "Virtualenv: $VENV"
echo "Config:     $ENV_FILE"
echo
echo "Run doctor:"
echo "  source '$VENV/bin/activate' && python scripts/mini_gateway_doctor.py"
echo
echo "Start local MCP server:"
echo "  source '$VENV/bin/activate' && mcp run src/agentrunner/mini_gateway/server.py --transport streamable-http"
