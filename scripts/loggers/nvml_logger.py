#!/usr/bin/env python3
"""1 Hz NVML logger for the SensorGuard baseline tier.

Logs the nine baseline signals per GPU with dual timestamps
(CLOCK_MONOTONIC_RAW target tick + receipt time) plus a wall-clock UTC
anchor so traces from different hosts can be cross-referenced. Rows are
written incrementally; SIGINT/SIGTERM close the file cleanly. Sample-loss
accounting: every scheduled tick either produces a row per GPU or one
row with status=missed.
"""

import argparse
import csv
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pynvml

FIELDS = [
    "tick_index", "t_target_raw_s", "t_receipt_raw_s", "utc_anchor",
    "gpu_index", "gpu_uuid", "status",
    "util_gpu_pct", "util_mem_pct", "mem_used_mib", "power_w",
    "temp_c", "clock_sm_mhz", "clock_mem_mhz", "pcie_tx_kbps", "pcie_rx_kbps",
]

stop = False


def on_signal(signum, frame):
    global stop
    stop = True


def raw_now() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="all", help="comma-separated GPU indices, or 'all'")
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--duration-s", type=float, default=0, help="0 = run until signalled")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    indices = list(range(count)) if args.gpus == "all" else [int(i) for i in args.gpus.split(",")]
    handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices}
    uuids = {i: pynvml.nvmlDeviceGetUUID(h) for i, h in handles.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = raw_now()
    ticks = 0
    missed = 0
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        # Anchor row: maps CLOCK_MONOTONIC_RAW to wall clock once, at start.
        writer.writerow({"tick_index": -1, "t_target_raw_s": t0, "t_receipt_raw_s": raw_now(),
                         "utc_anchor": datetime.now(timezone.utc).isoformat(),
                         "gpu_index": -1, "gpu_uuid": "anchor", "status": "anchor"})
        fh.flush()
        while not stop:
            target = t0 + ticks * args.interval_s
            now = raw_now()
            if now < target:
                time.sleep(min(target - now, args.interval_s))
                continue
            late_by = now - target
            if late_by > args.interval_s:
                # The whole tick window was missed (host stall); account for it.
                skipped = int(late_by / args.interval_s)
                for _ in range(skipped):
                    writer.writerow({"tick_index": ticks, "t_target_raw_s": t0 + ticks * args.interval_s,
                                     "t_receipt_raw_s": now, "utc_anchor": "",
                                     "gpu_index": -1, "gpu_uuid": "", "status": "missed"})
                    ticks += 1
                    missed += 1
            for i in indices:
                h = handles[i]
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    row = {
                        "util_gpu_pct": util.gpu, "util_mem_pct": util.memory,
                        "mem_used_mib": mem.used // (1024 * 1024),
                        "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
                        "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                        "clock_sm_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                        "clock_mem_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
                        "pcie_tx_kbps": pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES),
                        "pcie_rx_kbps": pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_RX_BYTES),
                        "status": "ok",
                    }
                except pynvml.NVMLError as err:
                    row = {"status": f"error:{err}"}
                    missed += 1
                row.update({"tick_index": ticks, "t_target_raw_s": target,
                            "t_receipt_raw_s": raw_now(), "utc_anchor": "",
                            "gpu_index": i, "gpu_uuid": uuids[i]})
                writer.writerow(row)
            fh.flush()
            ticks += 1
            if args.duration_s and raw_now() - t0 >= args.duration_s:
                break
    pynvml.nvmlShutdown()
    print(f"nvml_logger: {ticks} ticks, {missed} missed/error samples, output {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
