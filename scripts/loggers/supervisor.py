#!/usr/bin/env python3
"""Run supervisor: one synchronized, audited SensorGuard run.

Sequence: start loggers -> load marker -> workload -> load marker ->
stop loggers -> per-channel health + measured alignment error -> filled
run manifest with artifact checksums. A run whose channels fail health
is flagged, never silently kept.

Only NVML and DCGM channels are implemented so far; electrical joins via
pico_logger once probe wiring has safety sign-off. The marker/alignment
machinery is channel-agnostic.
"""

import argparse
import csv
import hashlib
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def raw_now() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_marker_lines(text: str):
    bursts = []
    for line in text.splitlines():
        if line.startswith("marker_burst"):
            parts = dict(p.split("=") for p in line.split()[2:] if "=" in p)
            bursts.append((float(parts["start_raw_s"]), float(parts["end_raw_s"])))
    return bursts


def nvml_health(trace: Path, gpu_index: int, bursts, interval_s=1.0):
    """Missingness, clock alignment, and edge-detection latency for one NVML channel.

    NVML samples and the load marker share CLOCK_MONOTONIC_RAW on the same
    host, so channel alignment is the scheduling/receipt jitter of the
    sampling loop (subject to the preregistered alignment limit). The power
    edge is a separate diagnostic: its latency includes the GPU power
    sensor's own lag plus 1 Hz quantization, so it gets a physical-latency
    budget rather than the clock-alignment limit. For external channels
    (scope, camera) with no shared clock, the edge IS the alignment.
    """
    rows = [r for r in csv.DictReader(trace.open()) if r["status"] in ("ok", "missed")]
    gpu_rows = [r for r in rows if r["status"] == "ok" and int(r["gpu_index"]) == gpu_index]
    missed = sum(1 for r in rows if r["status"] == "missed")
    total_ticks = max((int(r["tick_index"]) for r in rows), default=-1) + 1
    missing_fraction = missed / total_ticks if total_ticks else 1.0

    # Clock alignment: worst receipt-vs-target jitter across the run.
    jitter = [abs(float(r["t_receipt_raw_s"]) - float(r["t_target_raw_s"])) for r in gpu_rows]
    alignment_error_s = max(jitter) if jitter else None

    # Edge detection: earliest sample inside each marker burst whose power
    # rises >=50 W above the pre-burst floor.
    latencies = []
    times = [float(r["t_target_raw_s"]) for r in gpu_rows]
    powers = [float(r["power_w"]) for r in gpu_rows]
    for start, end in bursts:
        pre = [p for t, p in zip(times, powers) if start - 10 <= t < start]
        floor = min(pre) if pre else 30.0
        hits = [t for t, p in zip(times, powers) if start <= t <= end + interval_s and p >= floor + 50]
        if hits:
            latencies.append(abs(hits[0] - start))
    return {
        "samples": len(gpu_rows),
        "missing_fraction": round(missing_fraction, 5),
        "alignment_error_ms": None if alignment_error_s is None else round(alignment_error_s * 1000, 1),
        "edge_latency_ms": None if not latencies else round(max(latencies) * 1000, 1),
        "bursts_detected": len(latencies),
        "bursts_expected": len(bursts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workload-cmd", required=True, help="shell command for the workload")
    parser.add_argument("--workload-family", default="non_ml")
    parser.add_argument("--workload-name", default="")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU under test (marker + health)")
    parser.add_argument("--gpus", default="all", help="GPU set passed to loggers")
    parser.add_argument("--sensors", default="nvml,dcgm")
    parser.add_argument("--out-root", type=Path, default=Path("data/runs"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-missing-fraction", type=float, default=0.01)
    parser.add_argument("--max-alignment-error-ms", type=float, default=100.0)
    parser.add_argument("--max-edge-latency-ms", type=float, default=3000.0,
                        help="power-edge detection budget: sensor lag + 1 Hz quantization")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profiled", action="store_true",
                        help="mark this run as a profiled characterization pass")
    args = parser.parse_args()

    sensors = [s.strip() for s in args.sensors.split(",") if s.strip()]
    out = args.out_root / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    start_utc = datetime.now(timezone.utc).isoformat()

    loggers = {}
    if "nvml" in sensors:
        loggers["nvml"] = subprocess.Popen(
            [args.python, str(HERE / "nvml_logger.py"), "--output", str(out / "nvml.csv"),
             "--gpus", args.gpus], stderr=subprocess.DEVNULL)
    if "dcgm" in sensors:
        loggers["dcgm"] = subprocess.Popen(
            [args.python, str(HERE / "dcgm_logger.py"), "--output", str(out / "dcgm.tsv"),
             "--gpus", "-1" if args.gpus == "all" else args.gpus], stderr=subprocess.DEVNULL)
    time.sleep(3)  # loggers reach steady state before the first marker

    def marker() -> list:
        res = subprocess.run(
            [args.python, str(HERE / "load_marker.py"), "--device", f"cuda:{args.gpu_index}"],
            capture_output=True, text=True, timeout=120)
        if res.returncode != 0:
            print(res.stdout + res.stderr, file=sys.stderr)
            raise RuntimeError("load marker failed")
        return parse_marker_lines(res.stdout)

    status = "completed"
    workload_rc = None
    try:
        bursts_pre = marker()
        t_wl_start = raw_now()
        workload_rc = subprocess.run(args.workload_cmd, shell=True).returncode
        t_wl_end = raw_now()
        bursts_post = marker()
    except Exception as err:
        status = f"failed:{err}"
        bursts_pre, bursts_post, t_wl_start, t_wl_end = [], [], None, None
    finally:
        time.sleep(2)
        for proc in loggers.values():
            proc.send_signal(signal.SIGTERM)
        for proc in loggers.values():
            proc.wait(timeout=15)

    channels = []
    health_pass = True
    if "nvml" in sensors and (out / "nvml.csv").exists():
        h = nvml_health(out / "nvml.csv", args.gpu_index, bursts_pre + bursts_post)
        ok = (h["missing_fraction"] <= args.max_missing_fraction
              and h["alignment_error_ms"] is not None
              and h["alignment_error_ms"] <= args.max_alignment_error_ms
              and h["edge_latency_ms"] is not None
              and h["edge_latency_ms"] <= args.max_edge_latency_ms
              and h["bursts_detected"] == h["bursts_expected"])
        health_pass &= ok
        channels.append({"channel_id": f"nvml.gpu{args.gpu_index}", "sample_rate_hz": 1.0,
                         "clock_source": "CLOCK_MONOTONIC_RAW", "health": "pass" if ok else "fail", **h})
    if "dcgm" in sensors and (out / "dcgm.tsv").exists():
        n_lines = sum(1 for line in (out / "dcgm.tsv").open() if not line.startswith("#"))
        ok = n_lines > 0
        health_pass &= ok
        channels.append({"channel_id": "dcgm.all", "sample_rate_hz": 1.0,
                         "clock_source": "CLOCK_MONOTONIC_RAW(receipt)",
                         "health": "pass" if ok else "fail", "samples": n_lines})

    if workload_rc not in (0, None):
        status = f"workload_exit_{workload_rc}"
    if not health_pass and status == "completed":
        status = "flagged_channel_health"

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    manifest = {
        "run_id": args.run_id,
        "git_commit": git_commit,
        "operator": "supervisor",
        "start_utc": start_utc,
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "workload": {"family": args.workload_family, "name": args.workload_name,
                     "command": args.workload_cmd, "seed": args.seed,
                     "duration_s": None if t_wl_end is None else round(t_wl_end - t_wl_start, 3)},
        "hardware": {"gpu_index_under_test": args.gpu_index, "gpu_set": args.gpus},
        "sensors": {s: (s in sensors) for s in
                    ["nvml", "dcgm", "electrical", "thermal_camera", "contact_temperature",
                     "ultrasound", "rf_sdr", "network_mirror"]},
        "sensor_channels": channels,
        "marker_bursts": {"pre": bursts_pre, "post": bursts_post},
        "profiled": bool(args.profiled),
        "artifact_checksums": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
        "status": status,
    }
    manifest_path = out / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"supervisor: status={status} manifest={manifest_path}")
    for ch in channels:
        print(f"  {ch['channel_id']}: health={ch['health']} "
              f"missing={ch.get('missing_fraction')} align_ms={ch.get('alignment_error_ms')} "
              f"edge_ms={ch.get('edge_latency_ms')}")
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
