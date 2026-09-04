#!/usr/bin/env python3
"""Matched training evasions for synchronized NVML/current acquisition.

Every positive mode performs a real parameter update and records optimization
progress.  The negative control uses the same model, batch, and forward path
without modifying weights.  No model or dataset download is required.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch import nn


def raw_now() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def wait_for_epoch(target: float | None) -> None:
    if target is None:
        return
    if time.time() > target + 2.0:
        raise RuntimeError(f"scheduled start {target} is already more than 2 s late")
    while time.time() < target:
        time.sleep(min(0.05, target - time.time()))


class LoRALinear(nn.Module):
    """Frozen dense projection plus trainable low-rank adapters."""

    def __init__(self, size: int, rank: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.register_buffer(
            "base",
            torch.randn(size, size, device=device, dtype=dtype) / math.sqrt(size),
        )
        self.a = nn.Parameter(torch.randn(rank, size, device=device, dtype=dtype) * 0.01)
        self.b = nn.Parameter(torch.zeros(size, rank, device=device, dtype=dtype))
        self.scale = 1.0 / rank

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.matmul(self.base.t()) + self.scale * value.matmul(self.a.t()).matmul(self.b.t())


def dense_network(size: int, depth: int, device: torch.device, dtype: torch.dtype) -> nn.Module:
    layers: list[nn.Module] = []
    for index in range(depth):
        layers.append(nn.Linear(size, size, bias=False, device=device, dtype=dtype))
        if index + 1 < depth:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def lora_network(
    size: int,
    depth: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    layers: list[nn.Module] = []
    for index in range(depth):
        layers.append(LoRALinear(size, rank, device, dtype))
        if index + 1 < depth:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def chunked_sgd(parameters: list[nn.Parameter], learning_rate: float, chunks: int) -> None:
    """Apply one SGD update as several short, separated vector operations."""

    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is None:
                continue
            flat_parameter = parameter.view(-1)
            flat_gradient = parameter.grad.view(-1)
            for parameter_chunk, gradient_chunk in zip(
                flat_parameter.tensor_split(chunks), flat_gradient.tensor_split(chunks)
            ):
                parameter_chunk.add_(gradient_chunk, alpha=-learning_rate)


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    inputs = torch.randn(
        args.batch_size, args.size, generator=generator, device=device, dtype=dtype
    )
    teacher = torch.randn(
        args.size, args.size, generator=generator, device=device, dtype=dtype
    ) / math.sqrt(args.size)
    targets = torch.tanh(inputs.matmul(teacher.t()))

    if args.mode == "lora_dilution":
        model = lora_network(args.size, args.depth, args.lora_rank, device, dtype)
    else:
        model = dense_network(args.size, args.depth, device, dtype)
    parameters = list(model.parameters())
    optimizer = None if args.mode == "chunked_optimizer" else torch.optim.AdamW(
        parameters, lr=args.learning_rate
    )
    initial_weights = [parameter.detach().clone() for parameter in parameters]

    wait_for_epoch(args.start_epoch_s)
    start_epoch_s = time.time()
    start = raw_now()
    deadline = start + args.duration_s
    training_steps = 0
    inference_passes = 0
    initial_loss = None
    final_loss = None

    def needs_minimum_work() -> bool:
        if args.mode == "inference_control":
            return inference_passes < args.min_steps
        return training_steps < args.min_steps

    while raw_now() < deadline or needs_minimum_work():
        if args.mode == "inference_control":
            with torch.inference_mode():
                output = model(inputs)
                loss = torch.nn.functional.mse_loss(output.float(), targets.float())
            inference_passes += 1
        else:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            else:
                for parameter in parameters:
                    parameter.grad = None
            output = model(inputs)
            loss = torch.nn.functional.mse_loss(output.float(), targets.float())
            loss.backward()
            if args.mode == "chunked_optimizer":
                chunked_sgd(parameters, args.learning_rate, args.optimizer_chunks)
            else:
                optimizer.step()
            training_steps += 1

            dilution = args.dilution if args.mode in {"inference_dilution", "lora_dilution"} else 0
            with torch.inference_mode():
                for _ in range(dilution):
                    model(inputs)
                    inference_passes += 1
            if args.mode == "throttled" and args.throttle_s > 0:
                time.sleep(args.throttle_s)

        synchronize(device)
        value = float(loss.detach())
        initial_loss = value if initial_loss is None else initial_loss
        final_loss = value

    elapsed = raw_now() - start
    max_weight_change = max(
        (
            float((after.detach() - before).abs().max())
            for before, after in zip(initial_weights, parameters)
        ),
        default=0.0,
    )
    loss_reduction = (
        0.0
        if not initial_loss
        else float((initial_loss - final_loss) / initial_loss)
    )
    is_training = args.mode != "inference_control"
    progress_valid = bool(
        is_training
        and math.isfinite(float(final_loss))
        and max_weight_change > 0
    )
    return {
        "mode": args.mode,
        "target": "training" if is_training else "not_training",
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "matrix_size": args.size,
        "depth": args.depth,
        "lora_rank": args.lora_rank if args.mode == "lora_dilution" else None,
        "dilution": args.dilution if args.mode in {"inference_dilution", "lora_dilution"} else 0,
        "optimizer_chunks": args.optimizer_chunks if args.mode == "chunked_optimizer" else 1,
        "throttle_s": args.throttle_s if args.mode == "throttled" else 0.0,
        "seed": args.seed,
        "training_steps": training_steps,
        "inference_passes": inference_passes,
        "elapsed_s": round(elapsed, 6),
        "scheduled_start_epoch_s": args.start_epoch_s,
        "start_epoch_s": start_epoch_s,
        "end_epoch_s": time.time(),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "relative_loss_reduction": loss_reduction,
        "max_weight_change": max_weight_change,
        "meaningful_optimization_progress": progress_valid,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "honest_full",
            "chunked_optimizer",
            "throttled",
            "inference_dilution",
            "lora_dilution",
            "inference_control",
        ],
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--duration-s", type=float, default=300)
    parser.add_argument("--start-epoch-s", type=float)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer-chunks", type=int, default=8)
    parser.add_argument("--dilution", type=int, default=20)
    parser.add_argument("--throttle-s", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.optimizer_chunks < 1 or args.dilution < 0 or args.lora_rank < 1:
        parser.error("optimizer chunks/rank must be positive and dilution non-negative")

    result = run(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("useful_work " + json.dumps(result, sort_keys=True))
    if args.mode != "inference_control" and not result["meaningful_optimization_progress"]:
        print("ERROR: training mode did not make measurable optimization progress")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
