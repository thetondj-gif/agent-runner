#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from agentrunner.mini_gateway.executor import agent_status


def ollama_models() -> list[str]:
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) <= 1:
        return []
    return [line.split()[0] for line in lines[1:]]


def main() -> None:
    status = agent_status()
    status["ollama_models"] = ollama_models()
    status["gateway_config"] = str(Path.home() / ".config" / "agentrunner" / "mini-gateway.env")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
