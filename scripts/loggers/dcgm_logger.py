#!/usr/bin/env python3
"""1 Hz DCGM logger (tier 0, alongside NVML).

Wraps `dcgmi dmon` and stamps every received line with
CLOCK_MONOTONIC_RAW receipt time plus one wall-clock UTC anchor, matching
the field discipline of nvml_logger.py. DCGM's own hostengine timestamps
are not exposed by dmon, so receipt time is the alignment reference; the
supervisor measures the actual offset against the load marker.

Default field set (DCGM field ids):
  155 power usage (W)        150 GPU temp (C)
  100 SM clock (MHz)         101 memory clock (MHz)
  203 GPU utilization (%)    204 memory-copy utilization (%)
  252 framebuffer used (MiB) 1009 PCIe TX (KB/s)   1010 PCIe RX (KB/s)
"""

import argparse
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_FIELDS = "155,150,100,101,203,204,252,1009,1010"


def raw_now() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="-1", help="comma-separated GPU ids for dcgmi -i, -1 = all")
    parser.add_argument("--fields", default=DEFAULT_FIELDS)
    parser.add_argument("--interval-ms", type=int, default=1000)
    parser.add_argument("--duration-s", type=float, default=0, help="0 = run until signalled")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["dcgmi", "dmon", "-i", args.gpus, "-e", args.fields, "-d", str(args.interval_ms)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def on_signal(signum, frame):
        proc.terminate()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    t0 = raw_now()
    lines = 0
    with args.output.open("w") as fh:
        fh.write(f"# anchor t_raw_s={t0} utc={datetime.now(timezone.utc).isoformat()}\n")
        fh.write(f"# cmd={' '.join(cmd)}\n")
        for line in proc.stdout:
            fh.write(f"{raw_now():.6f}\t{line.rstrip()}\n")
            fh.flush()
            lines += 1
            if args.duration_s and raw_now() - t0 >= args.duration_s:
                proc.terminate()
                break
    proc.wait(timeout=10)
    print(f"dcgm_logger: {lines} lines, output {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
