#!/usr/bin/env python3
"""Matched fused-update experiment with measurable optimization progress.

The four modes use identical matrix shapes:

* ``forward``: one inference-like forward pass.
* ``forward3``: three forwards, the FLOP-matched negative control.
* ``dummy_write``: forward, dX, and dW written to scratch, but no learning.
* ``fused_update``: forward, dX, and an in-place SGD weight update in dW.
* ``adamw``: ordinary autograd plus AdamW, the positive training control.

The fused mode is inspired by Robi Rahman's symmetric-GEMM experiment in
robirahman/GPU-monitoring@101027906b63067ded909c272e43df19e56c75c1,
but is a small SensorGuard-native adapter that has no model download.  It is
intended for acquisition-pipeline validation on RTX 3090 nodes.  The existing
TinyLlama implementation and its prior measurements remain separate evidence.
"""

import argparse
import json
import math
import time

import torch
from torch import nn


def raw_now() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


class MatchedLinearFn(torch.autograd.Function):
    """Linear backward that either discards dW or updates W in-place."""

    @staticmethod
    def forward(ctx, x, weight, mode, learning_rate, scratch):
        ctx.weight = weight
        ctx.mode = mode
        ctx.learning_rate = learning_rate
        ctx.scratch = scratch
        ctx.save_for_backward(x)
        return x.matmul(weight.t())

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        weight = ctx.weight
        g2 = grad_output.reshape(-1, grad_output.shape[-1])
        x2 = x.reshape(-1, x.shape[-1])

        # dX must use the old weight.  The fused path mutates it below.
        dx = grad_output.matmul(weight)
        if ctx.mode == "dummy_write":
            torch.matmul(g2.t(), x2, out=ctx.scratch)
        elif ctx.mode == "fused_update":
            weight.addmm_(g2.t(), x2, beta=1.0, alpha=-ctx.learning_rate)
        return dx, None, None, None, None


class MatchedLinear(nn.Module):
    def __init__(self, size: int, mode: str, learning_rate: float, generator):
        super().__init__()
        scale = 1.0 / math.sqrt(size)
        self.register_buffer("weight", torch.randn(size, size, generator=generator) * scale)
        self.register_buffer("scratch", torch.empty(size, size))
        self.mode = mode
        self.learning_rate = learning_rate

    def forward(self, x):
        return MatchedLinearFn.apply(
            x, self.weight, self.mode, self.learning_rate, self.scratch
        )


class FusedNetwork(nn.Module):
    def __init__(self, size: int, depth: int, mode: str, learning_rate: float, generator):
        super().__init__()
        self.layers = nn.ModuleList(
            MatchedLinear(size, mode, learning_rate, generator) for _ in range(depth)
        )

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index + 1 < len(self.layers):
                x = torch.nn.functional.gelu(x)
        return x


def make_batch(batch_size: int, size: int, device, dtype, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(batch_size, size, generator=generator, dtype=torch.float32)
    teacher = torch.randn(size, size, generator=generator, dtype=torch.float32) / math.sqrt(size)
    y = torch.tanh(x.matmul(teacher.t()))
    return x.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype), generator


def synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(args) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    x, target, generator = make_batch(args.batch_size, args.size, device, dtype, args.seed)

    if args.mode == "adamw":
        layers = []
        for index in range(args.depth):
            layers.append(nn.Linear(args.size, args.size, bias=False))
            if index + 1 < args.depth:
                layers.append(nn.GELU())
        model = nn.Sequential(*layers).to(device=device, dtype=dtype)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        model = FusedNetwork(
            args.size, args.depth, args.mode, args.learning_rate, generator
        ).to(device=device, dtype=dtype)
        optimizer = None

    start = raw_now()
    deadline = start + args.duration_s
    steps = 0
    initial_loss = None
    final_loss = None
    weight_tensors = (
        [p for name, p in model.named_parameters() if name.endswith("weight")]
        if optimizer is not None
        else [p for name, p in model.named_buffers() if name.endswith("weight")]
    )
    initial_weights = [p.detach().clone() for p in weight_tensors]

    while raw_now() < deadline or steps < args.min_steps:
        if args.mode in {"forward", "forward3"}:
            repeats = 1 if args.mode == "forward" else 3
            with torch.inference_mode():
                output = None
                for _ in range(repeats):
                    output = model(x)
            loss = torch.nn.functional.mse_loss(output.float(), target.float())
        else:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            # The manual fused layers need an input gradient to root autograd.
            batch = x.detach().requires_grad_(optimizer is None)
            output = model(batch)
            loss = torch.nn.functional.mse_loss(output.float(), target.float())
            loss.backward()
            if optimizer is not None:
                optimizer.step()

        synchronize(device)
        value = float(loss.detach())
        initial_loss = value if initial_loss is None else initial_loss
        final_loss = value
        steps += 1

    elapsed = raw_now() - start
    max_weight_change = max(
        (float((after - before).abs().max()) for before, after in zip(initial_weights, weight_tensors)),
        default=0.0,
    )
    loss_reduction = 0.0 if not initial_loss else (initial_loss - final_loss) / initial_loss
    learns = args.mode in {"fused_update", "adamw"}
    progress_valid = bool(
        learns and math.isfinite(final_loss) and loss_reduction > 0 and max_weight_change > 0
    )
    return {
        "mode": args.mode,
        "device": str(device),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "matrix_size": args.size,
        "depth": args.depth,
        "seed": args.seed,
        "steps": steps,
        "elapsed_s": round(elapsed, 6),
        "examples": steps * args.batch_size,
        "examples_per_s": round(steps * args.batch_size / elapsed, 3),
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
        choices=["forward", "forward3", "dummy_write", "fused_update", "adamw"],
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--duration-s", type=float, default=600)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run(args)
    print("useful_work " + json.dumps(result, sort_keys=True))
    if args.mode in {"fused_update", "adamw"} and not result["meaningful_optimization_progress"]:
        print("ERROR: training mode did not make measurable optimization progress")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
