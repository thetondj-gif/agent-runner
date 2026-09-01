#!/usr/bin/env python3
"""Execute and independently verify one Universal Execution Contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentrunner.core.execution_contract import run_fast_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract")
    parser.add_argument("receipt")
    args = parser.parse_args()
    contract = json.loads(Path(args.contract).read_text())
    receipt = run_fast_path(contract, args.receipt)
    print(
        json.dumps(
            {"final_status": receipt["final_status"], "receipt_hash": receipt["receipt_hash"]}
        )
    )
    return 0 if receipt["final_status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
