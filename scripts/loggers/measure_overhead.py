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
    elif sensor == "pico":
        trace = out_dir / "pico"
        cmd = [
            run_logger.pico_python,
            str(HERE / "pico_logger.py"),
            "--parallel",
            "--duration-s", str(duration_s),
            "--sample-interval-us", str(run_logger.pico_interval_us),
            "--output-prefix", str(trace),
        ]
    else:
        return {"sensor": sensor, "available": False, "reason": "no logger / not attached"}

    proc = subprocess.Popen(cmd, stderr=subprocess.DEVNULL)
    cpu = []
    import threading
    t = threading.Thread(target=proc_cpu_sampler, args=(proc.pid, raw_now() + duration_s, cpu))
    t.start()
    if sensor == "pico":
        try:
            proc.wait(timeout=duration_s + 30)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=10)
            raise RuntimeError("PicoScope logger did not finish within its acquisition budget")
        if proc.returncode:
            return {"sensor": sensor, "available": False,
                    "reason": f"pico logger exited {proc.returncode}"}
    else:
        time.sleep(duration_s)
        proc.terminate()
        proc.wait(timeout=10)
    t.join()

    if sensor == "pico":
        artifacts = list(out_dir.glob("pico_u*"))
        size = sum(path.stat().st_size for path in artifacts)
        metas = [json.loads(path.read_text()) for path in out_dir.glob("pico_u*_meta.json")]
    else:
        size = trace.stat().st_size if trace.exists() else 0
        metas = []
    res = {
        "sensor": sensor, "available": True,
        "cpu_pct_mean": round(float(np.mean(cpu)), 2) if cpu else None,
        "cpu_pct_max": round(float(np.max(cpu)), 2) if cpu else None,
        "bytes_per_gpu_hour": round(size / (duration_s / 3600), 0),
        "trace_bytes": size,
    }
    if sensor == "nvml":
        res.update(nvml_latency_throughput(trace))
    elif sensor == "pico":
        res.update({
            "units": len(metas),
            "channels": 2 * len(metas),
            "sample_interval_us": run_logger.pico_interval_us,
            "samples_per_channel_mean": round(float(np.mean([m["samples"] for m in metas])), 1)
                if metas else 0,
            "overflow_units": sum(bool(m["overflow_flags"]) for m in metas),
            "clipped_units_channel_a": sum(
                (m["clipping_fraction_a"] or 0) > 0.01 for m in metas
            ),
        })
    return res


def start_penalty_logger(sensor: str, output_dir: Path, gpus: str,
                         pico_python: str, pico_interval_us: int):
    """Start the logger used during the paired useful-work interval."""
    if sensor == "nvml":
        return subprocess.Popen(
            [sys.executable, str(HERE / "nvml_logger.py"),
             "--output", "/dev/null", "--gpus", gpus],
            stderr=subprocess.DEVNULL,
        )
    if sensor == "dcgm":
        return subprocess.Popen(
            [sys.executable, str(HERE / "dcgm_logger.py"),
             "--output", "/tmp/_oh_dcgm.tsv", "--gpus",
             "-1" if gpus == "all" else gpus],
            stderr=subprocess.DEVNULL,
        )
    if sensor == "pico":
        output_dir.mkdir(parents=True, exist_ok=True)
        return subprocess.Popen(
            [pico_python, str(HERE / "pico_logger.py"), "--parallel",
             "--duration-s", "25", "--sample-interval-us", str(pico_interval_us),
             "--output-prefix", str(output_dir / "pico")],
            stderr=subprocess.DEVNULL,
        )
    raise ValueError(f"unsupported sensor: {sensor}")


def stop_penalty_logger(proc: subprocess.Popen, sensor: str) -> None:
    if sensor == "pico":
        proc.wait(timeout=15)
    else:
        proc.terminate()
        proc.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensors", default="nvml,dcgm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", default="all")
    parser.add_argument("--duration-s", type=float, default=120)
    parser.add_argument("--pico-python", default=sys.executable,
                        help="Python interpreter containing picosdk")
    parser.add_argument("--pico-sample-interval-us", type=int, default=100)
    parser.add_argument("--penalty-order", choices=["baseline-first", "logger-first"],
                        default="baseline-first",
                        help="order of the paired 20 s useful-work measurements")
    parser.add_argument("--output", type=Path, default=Path("results/sensor_overhead.json"))
    args = parser.parse_args()
    run_logger.pico_python = args.pico_python
    run_logger.pico_interval_us = args.pico_sample_interval_us

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

    # Warm the GPU before the paired measurement. The repetition driver
    # alternates condition order to control remaining thermal/time drift.
    steady_gemm_throughput(10)

    results = {"config": {"duration_s": args.duration_s, "device": args.device, "gpus": args.gpus},
               "penalty_order": args.penalty_order, "sensors": []}
    for sensor in [s.strip() for s in args.sensors.split(",") if s.strip()]:
        if sensor in ("nvml", "dcgm", "pico"):
            entry = run_logger(sensor, args.output.parent / f"overhead_{sensor}", args.duration_s, args.gpus)
            def measured_rate():
                proc = start_penalty_logger(
                    sensor, args.output.parent / "overhead_pico_penalty", args.gpus,
                    args.pico_python, args.pico_sample_interval_us,
                )
                time.sleep(2)
                rate = steady_gemm_throughput(20)
                stop_penalty_logger(proc, sensor)
                return rate

            if args.penalty_order == "baseline-first":
                base_rate = steady_gemm_throughput(20)
                monitored_rate = measured_rate()
            else:
                monitored_rate = measured_rate()
                base_rate = steady_gemm_throughput(20)
            entry["baseline_gemm_per_s"] = round(base_rate, 3)
            entry["monitored_gemm_per_s"] = round(monitored_rate, 3)
            entry["useful_work_penalty_pct"] = round(
                100 * (base_rate - monitored_rate) / base_rate, 3
            )
        else:
            entry = {"sensor": sensor, "available": False, "reason": "physical sensor not attached"}
        results["sensors"].append(entry)

    available = [entry for entry in results["sensors"] if entry.get("available")]
    if len(available) == 1 and "baseline_gemm_per_s" in available[0]:
        results["baseline_gemm_per_s"] = available[0]["baseline_gemm_per_s"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
