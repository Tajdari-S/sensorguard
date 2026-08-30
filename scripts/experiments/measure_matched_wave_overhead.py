#!/usr/bin/env python3
"""Measure SensorGuard base-logger overhead on WAVE's exact six workloads.

This experiment deliberately reuses the model configurations, inference entry
point, Python environment, GPU, repetition count, and whole-process wall-time
definition from WAVE's RTX 3090 overhead reproduction.  The only changed
condition is whether the SensorGuard NVML and DCGM loggers run concurrently.
Physical-sensor overhead is outside this test and must remain separately
labelled as pending.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


CONFIGS = [
    ("gpt2", 6, 1024, 4096, 16, 512, 2, 4, 1),
    ("gpt2", 8, 3072, 12288, 48, 1024, 2, 4, 1),
    ("llama", 6, 1024, 4096, 8, 512, 2, 4, 1),
    ("llama", 8, 4096, 16384, 32, 1024, 2, 4, 1),
    ("qwen", 6, 1024, 4096, 8, 512, 2, 4, 1),
    ("qwen", 8, 4096, 16384, 32, 1024, 2, 4, 1),
]


def model_id(config: tuple[object, ...]) -> str:
    model, layers, hidden, ffn, heads, positions, tokens, batch, prompt = config
    return (
        f"{model}_layer{layers}_embd{hidden}_ffn{ffn}_head{heads}_"
        f"pos{positions}_max{tokens}_batch{batch}_prompt{prompt}"
    )


def workload_command(wave_root: Path, config: tuple[object, ...]) -> list[str]:
    model, layers, hidden, ffn, heads, positions, tokens, batch, prompt = config
    return [
        str(wave_root / ".venv/bin/python3"),
        str(wave_root / "eval/lower_bound/inference.py"),
        "--model-type", str(model),
        "--n_layer", str(layers),
        "--hidden-dim", str(hidden),
        "--ffn-dim", str(ffn),
        "--n_head", str(heads),
        "--n_positions", str(positions),
        "--max_new_tokens", str(tokens),
        "--batch-size", str(batch),
        "--prompt-len", str(prompt),
    ]


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def assert_gpu_idle(index: int) -> dict[str, object]:
    deadline = time.monotonic() + 30.0
    while True:
        query = subprocess.check_output(
            [
                "nvidia-smi", f"--id={index}",
                "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total,power.limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        uuid, name, util, used, total, power_limit = [part.strip() for part in query.split(",")]
        apps = subprocess.run(
            [
                "nvidia-smi", f"--id={index}",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout.strip()
        if util in {"0", "[N/A]", "N/A"} and float(used) <= 128 and not apps:
            return {
                "index": index,
                "uuid": uuid,
                "name": name,
                "memory_total_mib": float(total),
                "power_limit_w": float(power_limit),
            }
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU {index} did not become idle: {query}; apps={apps}")
        time.sleep(2.0)


def run_workload(command: list[str], environment: dict[str, str]) -> float:
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=Path(command[1]).resolve().parents[2],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=900,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if completed.returncode:
        raise RuntimeError(
            f"workload failed with exit {completed.returncode}: {completed.stderr[-2000:]}"
        )
    if "Generated Outputs" not in completed.stdout:
        raise RuntimeError(
            "workload exited without a success marker: "
            f"stdout={completed.stdout[-1000:]} stderr={completed.stderr[-1000:]}"
        )
    return elapsed


def start_loggers(
    sensor_root: Path,
    logger_python: Path,
    gpu_index: int,
    trace_dir: Path,
) -> list[subprocess.Popen]:
    trace_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        [
            str(logger_python), str(sensor_root / "scripts/loggers/nvml_logger.py"),
            "--output", str(trace_dir / "nvml.csv"), "--gpus", str(gpu_index),
        ],
        [
            str(logger_python), str(sensor_root / "scripts/loggers/dcgm_logger.py"),
            "--output", str(trace_dir / "dcgm.tsv"), "--gpus", str(gpu_index),
        ],
    ]
    processes = [
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        for command in commands
    ]
    time.sleep(1.25)
    for process, command in zip(processes, commands):
        if process.poll() is not None:
            error = process.stderr.read() if process.stderr else ""
            raise RuntimeError(f"logger failed to start: {' '.join(command)}\n{error}")
    return processes


def stop_loggers(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def trace_health(trace_dir: Path) -> dict[str, object]:
    nvml = trace_dir / "nvml.csv"
    dcgm = trace_dir / "dcgm.tsv"
    with nvml.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ok_rows = sum(row.get("status") == "ok" for row in rows)
    dcgm_data_rows = sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in dcgm.read_text().splitlines()
    )
    if ok_rows < 2 or dcgm_data_rows < 2:
        raise RuntimeError(
            f"logger health failure in {trace_dir}: NVML ok={ok_rows}, DCGM lines={dcgm_data_rows}"
        )
    return {"nvml_ok_rows": ok_rows, "dcgm_data_rows": dcgm_data_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave-root", type=Path, default=Path("/home/felkru/Wave"))
    parser.add_argument("--sensor-root", type=Path, default=Path("/home/felkru/sensorguard"))
    parser.add_argument("--logger-python", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu = assert_gpu_idle(args.gpu_index)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    environment["TOKENIZERS_PARALLELISM"] = "false"

    result: dict[str, object] = {
        "experiment": "matched_wave_sensor_overhead",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "whole-process wall time; NVML+DCGM base loggers only",
        "physical_sensor_overhead": "not measured",
        "repetitions": args.repetitions,
        "gpu": gpu,
        "wave_commit": git_commit(args.wave_root),
        "sensorguard_commit": git_commit(args.sensor_root),
        "configurations": [],
    }

    for config in CONFIGS:
        name = model_id(config)
        command = workload_command(args.wave_root, config)
        print(f"[{name}] warmup", flush=True)
        run_workload(command, environment)
        direct: list[float] = []
        monitored: list[float] = []
        health: list[dict[str, object]] = []
        for repetition in range(1, args.repetitions + 1):
            order = ("direct", "monitored") if repetition % 2 else ("monitored", "direct")
            for condition in order:
                assert_gpu_idle(args.gpu_index)
                if condition == "direct":
                    elapsed = run_workload(command, environment)
                    direct.append(elapsed)
                else:
                    trace_dir = args.output_dir / "traces" / name / f"r{repetition:02d}"
                    loggers = start_loggers(
                        args.sensor_root, args.logger_python, args.gpu_index, trace_dir
                    )
                    try:
                        elapsed = run_workload(command, environment)
                    finally:
                        stop_loggers(loggers)
                    monitored.append(elapsed)
                    health.append(trace_health(trace_dir))
                print(f"[{name}] r{repetition} {condition}: {elapsed:.3f}s", flush=True)

        direct_mean = statistics.mean(direct)
        monitored_mean = statistics.mean(monitored)
        entry = {
            "model_id": name,
            "model_family": config[0],
            "direct_seconds": direct,
            "monitored_seconds": monitored,
            "direct_mean_seconds": direct_mean,
            "monitored_mean_seconds": monitored_mean,
            "runtime_multiplier": monitored_mean / direct_mean,
            "overhead_percent": 100.0 * (monitored_mean - direct_mean) / direct_mean,
            "trace_health": health,
            "command": command,
        }
        result["configurations"].append(entry)
        (args.output_dir / "matched-overhead.json").write_text(json.dumps(result, indent=2))

    configurations = result["configurations"]
    result["summary"] = {
        "configuration_count": len(configurations),
        "mean_runtime_multiplier": statistics.mean(
            entry["runtime_multiplier"] for entry in configurations
        ),
        "minimum_runtime_multiplier": min(
            entry["runtime_multiplier"] for entry in configurations
        ),
        "maximum_runtime_multiplier": max(
            entry["runtime_multiplier"] for entry in configurations
        ),
        "mean_overhead_percent": statistics.mean(
            entry["overhead_percent"] for entry in configurations
        ),
    }
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "matched-overhead.json").write_text(json.dumps(result, indent=2))

    with (args.output_dir / "matched-overhead.csv").open("w", newline="") as handle:
        fields = [
            "model_id", "model_family", "direct_mean_seconds",
            "monitored_mean_seconds", "runtime_multiplier", "overhead_percent",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for entry in configurations:
            writer.writerow({field: entry[field] for field in fields})

    print(json.dumps(result["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
