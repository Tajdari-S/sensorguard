#!/usr/bin/env python3
"""Per-sensor monitoring-overhead measurement (E5/E8 requirement).

For each logger, run it against a fixed steady workload and measure:
  - CPU: logger process %CPU (mean, max) via /proc sampling
  - storage: bytes written per GPU-hour (trace size / wall hours)
  - latency: per-sample handling latency (receipt - target) distribution
  - throughput: samples per second actually captured vs nominal
  - useful-work penalty: workload throughput with vs without the logger

Physical-sensor loggers (pico) are included only when --sensors names them
and a unit is attached; otherwise skipped and reported as unavailable.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def raw_now() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def proc_cpu_sampler(pid: int, stop_at: float, out: list):
    """Sample a process's cumulative CPU time; return %CPU series."""
    clk = 100.0  # USER_HZ, standard on Linux
    prev_t, prev_cpu = raw_now(), _proc_cpu_seconds(pid, clk)
    while raw_now() < stop_at and prev_cpu is not None:
        time.sleep(0.5)
        now, cpu = raw_now(), _proc_cpu_seconds(pid, clk)
        if cpu is None:
            break
        out.append(100.0 * (cpu - prev_cpu) / (now - prev_t))
        prev_t, prev_cpu = now, cpu


def _proc_cpu_seconds(pid: int, clk: float):
    try:
        parts = Path(f"/proc/{pid}/stat").read_text().split()
        return (int(parts[13]) + int(parts[14])) / clk  # utime + stime
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        return None


def nvml_latency_throughput(trace: Path):
    import csv
    rows = [r for r in csv.DictReader(trace.open()) if r["status"] == "ok"]
    if not rows:
        return {}
    lat_ms = [1000 * (float(r["t_receipt_raw_s"]) - float(r["t_target_raw_s"])) for r in rows]
    ticks = sorted({int(r["tick_index"]) for r in rows})
    span = float(rows[-1]["t_target_raw_s"]) - float(rows[0]["t_target_raw_s"])
    return {
        "latency_ms_mean": round(float(np.mean(lat_ms)), 2),
        "latency_ms_p95": round(float(np.percentile(lat_ms, 95)), 2),
        "latency_ms_max": round(float(np.max(lat_ms)), 2),
        "throughput_ticks_per_s": round(len(ticks) / span, 3) if span else None,
        "nominal_hz": 1.0,
    }


def run_logger(sensor: str, out_dir: Path, duration_s: float, gpus: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    if sensor == "nvml":
        trace = out_dir / "nvml.csv"
        cmd = [sys.executable, str(HERE / "nvml_logger.py"), "--output", str(trace), "--gpus", gpus]
    elif sensor == "dcgm":
        trace = out_dir / "dcgm.tsv"
        cmd = [sys.executable, str(HERE / "dcgm_logger.py"), "--output", str(trace),
               "--gpus", "-1" if gpus == "all" else gpus]
    else:
        return {"sensor": sensor, "available": False, "reason": "no logger / not attached"}

    proc = subprocess.Popen(cmd, stderr=subprocess.DEVNULL)
    cpu = []
    import threading
    t = threading.Thread(target=proc_cpu_sampler, args=(proc.pid, raw_now() + duration_s, cpu))
    t.start()
    time.sleep(duration_s)
    proc.terminate()
    proc.wait(timeout=10)
    t.join()

    size = trace.stat().st_size if trace.exists() else 0
    res = {
        "sensor": sensor, "available": True,
        "cpu_pct_mean": round(float(np.mean(cpu)), 2) if cpu else None,
        "cpu_pct_max": round(float(np.max(cpu)), 2) if cpu else None,
        "bytes_per_gpu_hour": round(size / (duration_s / 3600), 0),
        "trace_bytes": size,
    }
    if sensor == "nvml":
        res.update(nvml_latency_throughput(trace))
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensors", default="nvml,dcgm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", default="all")
    parser.add_argument("--duration-s", type=float, default=120)
    parser.add_argument("--output", type=Path, default=Path("results/sensor_overhead.json"))
    args = parser.parse_args()

    import torch
    dev = torch.device(args.device)
    a = torch.randn(8192, 8192, device=dev)
    b = torch.randn(8192, 8192, device=dev)

    def steady_gemm_throughput(seconds):
        end = raw_now() + seconds
        n = 0
        while raw_now() < end:
            a @ b
            torch.cuda.synchronize(dev)
            n += 1
        return n / seconds

    # Baseline useful-work rate with no logger running.
    base_rate = steady_gemm_throughput(20)

    results = {"config": {"duration_s": args.duration_s, "device": args.device, "gpus": args.gpus},
               "baseline_gemm_per_s": round(base_rate, 3), "sensors": []}
    for sensor in [s.strip() for s in args.sensors.split(",") if s.strip()]:
        # useful-work penalty: throughput while this logger runs
        proc = None
        if sensor in ("nvml", "dcgm"):
            entry = run_logger(sensor, args.output.parent / f"overhead_{sensor}", args.duration_s, args.gpus)
            # separate short penalty measurement with logger live
            if sensor == "nvml":
                p = subprocess.Popen([sys.executable, str(HERE / "nvml_logger.py"),
                                      "--output", "/dev/null", "--gpus", args.gpus],
                                     stderr=subprocess.DEVNULL)
            else:
                p = subprocess.Popen([sys.executable, str(HERE / "dcgm_logger.py"),
                                      "--output", "/tmp/_oh_dcgm.tsv",
                                      "--gpus", "-1" if args.gpus == "all" else args.gpus],
                                     stderr=subprocess.DEVNULL)
            time.sleep(2)
            rate = steady_gemm_throughput(20)
            p.terminate(); p.wait(timeout=10)
            entry["useful_work_penalty_pct"] = round(100 * (base_rate - rate) / base_rate, 3)
        else:
            entry = {"sensor": sensor, "available": False, "reason": "physical sensor not attached"}
        results["sensors"].append(entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
