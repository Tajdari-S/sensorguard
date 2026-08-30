#!/usr/bin/env python3
"""Download-free development workloads for timing/interleaving experiments.

Every training mode performs real SGD updates on a fixed synthetic regression
problem and emits a machine-readable useful-work record.  These workloads are
for detector development only; they do not open the sealed fused-update family.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn


def raw_now() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def make_problem(batch_size: int, size: int, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(batch_size, size, generator=generator)
    # A deterministic nonlinear target avoids a dataset/model download while
    # still requiring genuine gradient updates to reduce the objective.
    target = torch.tanh(0.65 * x + 0.25 * x.roll(1, dims=1))
    return x, target


def make_model(size: int, depth: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    layers = []
    for index in range(depth):
        layers.append(nn.Linear(size, size, bias=False))
        if index + 1 < depth:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def loss_value(model, x, target, device) -> float:
    with torch.inference_mode():
        value = torch.nn.functional.mse_loss(model(x), target)
    synchronize(device)
    return float(value)


def train_step(model, optimizer, x, target, device) -> float:
    optimizer.zero_grad(set_to_none=True)
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    synchronize(device)
    return float(loss.detach())


def inference_step(model, x, device) -> None:
    with torch.inference_mode():
        model(x)
    synchronize(device)


def move_optimizer_state(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def run(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    devices = [torch.device(args.device)]
    if args.mode == "migration":
        devices.append(torch.device(args.secondary_device))
    for device in devices:
        torch.empty(1, device=device)

    cpu_x, cpu_target = make_problem(args.batch_size, args.size, args.seed)
    model = make_model(args.size, args.depth, args.seed).to(devices[0])
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    initial_weights = [parameter.detach().cpu().clone() for parameter in model.parameters()]
    current_device = devices[0]
    x = cpu_x.to(current_device)
    target = cpu_target.to(current_device)
    initial_loss = loss_value(model, x, target, current_device)

    start = raw_now()
    deadline = start + args.duration_s
    next_migration = start + args.migration_period_s
    steps = 0
    inference_steps = 0
    migrations = 0
    active_training_s = 0.0
    active_inference_s = 0.0

    while raw_now() < deadline or steps < args.min_steps:
        if args.mode == "duty_shaping":
            cycle_start = raw_now()
            train_deadline = min(deadline, cycle_start + args.cycle_s * args.training_fraction)
            while raw_now() < train_deadline:
                step_start = raw_now()
                train_step(model, optimizer, x, target, current_device)
                active_training_s += raw_now() - step_start
                steps += 1
            sleep_s = min(deadline - raw_now(), cycle_start + args.cycle_s - raw_now())
            if sleep_s > 0:
                time.sleep(sleep_s)
        elif args.mode == "interleaving":
            cycle_start = raw_now()
            train_deadline = min(deadline, cycle_start + args.cycle_s * args.training_fraction)
            while raw_now() < train_deadline:
                step_start = raw_now()
                train_step(model, optimizer, x, target, current_device)
                active_training_s += raw_now() - step_start
                steps += 1
            cycle_deadline = min(deadline, cycle_start + args.cycle_s)
            while raw_now() < cycle_deadline:
                step_start = raw_now()
                inference_step(model, x, current_device)
                active_inference_s += raw_now() - step_start
                inference_steps += 1
        else:
            if args.mode == "migration" and raw_now() >= next_migration:
                current_device = devices[(migrations + 1) % len(devices)]
                model.to(current_device)
                move_optimizer_state(optimizer, current_device)
                x = cpu_x.to(current_device)
                target = cpu_target.to(current_device)
                synchronize(current_device)
                migrations += 1
                next_migration = raw_now() + args.migration_period_s

            if args.mode == "inference_control":
                step_start = raw_now()
                inference_step(model, x, current_device)
                active_inference_s += raw_now() - step_start
                inference_steps += 1
            else:
                if args.mode == "memory_minimal":
                    micro = min(args.microbatch_size, x.shape[0])
                    offset = (steps * micro) % x.shape[0]
                    if offset + micro <= x.shape[0]:
                        step_x = x[offset : offset + micro]
                        step_target = target[offset : offset + micro]
                    else:
                        step_x = x[:micro]
                        step_target = target[:micro]
                else:
                    step_x, step_target = x, target
                step_start = raw_now()
                train_step(model, optimizer, step_x, step_target, current_device)
                active_training_s += raw_now() - step_start
                steps += 1

    end = raw_now()
    elapsed = end - start
    # Evaluate on the complete fixed problem after the last migration.
    x = cpu_x.to(current_device)
    target = cpu_target.to(current_device)
    final_loss = loss_value(model, x, target, current_device)
    final_weights = [parameter.detach().cpu() for parameter in model.parameters()]
    max_weight_change = max(
        (float((after - before).abs().max()) for before, after in zip(initial_weights, final_weights)),
        default=0.0,
    )
    reduction = (initial_loss - final_loss) / initial_loss if initial_loss else 0.0
    training_mode = args.mode != "inference_control"
    meaningful = bool(
        training_mode
        and steps > 0
        and math.isfinite(final_loss)
        and reduction > 0
        and max_weight_change > 0
    )
    result = {
        "mode": args.mode,
        "devices": [str(device) for device in devices],
        "batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size if args.mode == "memory_minimal" else None,
        "matrix_size": args.size,
        "depth": args.depth,
        "seed": args.seed,
        "steps": steps,
        "inference_steps": inference_steps,
        "migrations": migrations,
        "elapsed_s": round(elapsed, 6),
        "start_raw_s": start,
        "end_raw_s": end,
        "active_training_s": round(active_training_s, 6),
        "active_inference_s": round(active_inference_s, 6),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "relative_loss_reduction": reduction,
        "max_weight_change": max_weight_change,
        "meaningful_optimization_progress": meaningful,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "ordinary_training",
            "duty_shaping",
            "interleaving",
            "memory_minimal",
            "migration",
            "inference_control",
        ],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--secondary-device", default="cuda:1")
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--cycle-s", type=float, default=4.0)
    parser.add_argument("--training-fraction", type=float, default=0.35)
    parser.add_argument("--migration-period-s", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=5100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not 0 < args.training_fraction <= 1:
        parser.error("--training-fraction must be in (0, 1]")
    if args.mode == "migration" and args.device == args.secondary_device:
        parser.error("migration requires two different visible CUDA devices")

    result = run(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("useful_work " + json.dumps(result, sort_keys=True), flush=True)
    if args.mode != "inference_control" and not result["meaningful_optimization_progress"]:
        print("ERROR: workload made no measurable optimization progress", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
