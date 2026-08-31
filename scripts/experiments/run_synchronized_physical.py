#!/usr/bin/env python3
"""Execute one side of an absolute-time synchronized NVML/Pico campaign."""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def wait_until(epoch_s: float) -> None:
    while time.time() < epoch_s:
        time.sleep(min(0.2, epoch_s - time.time()))


def node_command(run: dict, python: str, output: Path) -> list[str]:
    common = ["--device", "cuda:1", "--duration-s", str(run["duration_s"]),
              "--start-epoch-s", str(run["start_epoch_s"]), "--seed", str(run["seed"]),
              "--output", str(output)]
    if run["kind"] == "development":
        return [python, "scripts/workloads/development_evasion_workload.py", "--mode", run["mode"],
                "--secondary-device", "cuda:2", *common]
    if run["kind"] == "control":
        return [python, "scripts/workloads/scheduled_control_workload.py", "--mode", run["mode"], *common]
    if run["kind"] == "fused":
        # This branch is used only after a frozen manifest authorizes the sealed family.
        common_no_output = common[:-2]
        command = [python, "scripts/workloads/fused_update_workload.py", "--mode", run["mode"],
                   *common_no_output]
        for plan_key, cli_flag in (
            ("batch_size", "--batch-size"), ("size", "--size"),
            ("depth", "--depth"), ("dtype", "--dtype"),
            ("learning_rate", "--learning-rate"),
        ):
            if plan_key in run:
                command.extend([cli_flag, str(run[plan_key])])
        return command
    raise ValueError(run["kind"])


def write_status(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=["node", "verifier"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--scope-python", default="/home/felkru/picoenv/bin/python")
    parser.add_argument("--scope-serial", default="12789/2929")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    rows = []
    for run in plan["runs"]:
        run_dir = args.out_root / run["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        if time.time() > run["start_epoch_s"] + 2:
            rows.append({"run_id": run["run_id"], "role": args.role, "return_code": 98,
                         "started_epoch_s": time.time(), "finished_epoch_s": time.time()})
            write_status(args.out_root / f"status_{args.role}.csv", rows)
            continue
        if args.role == "verifier":
            # Opening/enumerating all six legacy units takes about 13 s on the
            # verifier.  Start early enough to retain a measured pre-run baseline.
            wait_until(run["start_epoch_s"] - 25)
            command = [args.scope_python, "scripts/loggers/pico_logger.py",
                       "--serial", args.scope_serial, "--duration-s", str(run["duration_s"] + 25),
                       "--sample-interval-us", str(plan.get("sample_interval_us", 100)),
                       "--output-prefix", str(run_dir / "pico")]
            started = time.time()
            code = subprocess.run(command, stdout=(run_dir / "pico.stdout").open("w"),
                                  stderr=(run_dir / "pico.stderr").open("w")).returncode
        else:
            wait_until(run["start_epoch_s"] - 18)
            nvml = subprocess.Popen(
                [args.python, "scripts/loggers/nvml_logger.py", "--gpus", "1", "--interval-s", "1",
                 "--duration-s", str(run["duration_s"] + 30), "--output", str(run_dir / "nvml.csv")],
                stdout=(run_dir / "nvml.stdout").open("w"), stderr=(run_dir / "nvml.stderr").open("w"))
            command = node_command(run, args.python, run_dir / "workload.json")
            started = time.time()
            code = subprocess.run(command, stdout=(run_dir / "workload.stdout").open("w"),
                                  stderr=(run_dir / "workload.stderr").open("w")).returncode
            nvml_code = nvml.wait()
            code = code or nvml_code
        rows.append({"run_id": run["run_id"], "role": args.role, "return_code": code,
                     "started_epoch_s": started, "finished_epoch_s": time.time()})
        write_status(args.out_root / f"status_{args.role}.csv", rows)
    return 0 if rows and all(row["return_code"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
