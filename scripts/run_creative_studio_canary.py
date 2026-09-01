#!/usr/bin/env python3
"""Run a small real Creative Studio canary against its local ComfyUI provider."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:8188"
OUTPUT_ROOT = Path("/Users/alinton/Documents/ComfyUI/output")
PREFIX = "uec/creative-studio-canary"


def request_json(path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{BASE_URL}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def health() -> None:
    request_json("/system_stats")
    print("CREATIVE_STUDIO_CONNECTED")


def version() -> None:
    stats = request_json("/system_stats")
    print(stats["system"]["comfyui_version"])


def canary() -> None:
    workflow = {
        "1": {
            "class_type": "EmptyImage",
            "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 2102840},
        },
        "2": {
            "class_type": "SaveImage",
            "inputs": {"images": ["1", 0], "filename_prefix": PREFIX},
        },
    }
    prompt_id = request_json("/prompt", {"prompt": workflow})["prompt_id"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        history = request_json(f"/history/{prompt_id}").get(prompt_id)
        if history:
            status = history.get("status", {})
            if status.get("status_str") != "success":
                raise RuntimeError(f"Creative Studio canary failed: {status}")
            image = history["outputs"]["2"]["images"][0]
            artifact = OUTPUT_ROOT / image.get("subfolder", "") / image["filename"]
            result = {
                "marker": "UEC_CREATIVE_STUDIO_OK",
                "prompt_id": prompt_id,
                "artifact": str(artifact),
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "width": 64,
                "height": 64,
            }
            print(json.dumps(result, sort_keys=True))
            return
        time.sleep(0.1)
    raise TimeoutError(f"Creative Studio canary timed out: {prompt_id}")


def rollback() -> None:
    directory = OUTPUT_ROOT / "uec"
    removed = 0
    if directory.is_dir():
        for artifact in directory.glob("creative-studio-canary_*.png"):
            artifact.unlink()
            removed += 1
    print(json.dumps({"marker": "CREATIVE_STUDIO_ROLLED_BACK", "removed": removed}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("health", "version", "canary", "rollback"))
    action = parser.parse_args().action
    {"health": health, "version": version, "canary": canary, "rollback": rollback}[action]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
