#!/usr/bin/env python3
"""Start an existing workload at an absolute time without modifying its repo."""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def wait_until(epoch_s: float) -> None:
    while time.time() < epoch_s:
        time.sleep(min(0.2, epoch_s - time.time()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-epoch-s", type=float, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("an external command is required after --")
    if not args.workdir.is_dir():
        parser.error(f"workdir does not exist: {args.workdir}")

    wait_until(args.start_epoch_s)
    started = time.time()
    completed = subprocess.run(command, cwd=args.workdir)
    finished = time.time()
    payload = {
        "command": command,
        "workdir": str(args.workdir),
        "started_epoch_s": started,
        "finished_epoch_s": finished,
        "start_epoch_s": started,
        "end_epoch_s": finished,
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_utc": datetime.fromtimestamp(finished, timezone.utc).isoformat(),
        "return_code": completed.returncode,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
