#!/usr/bin/env python3
"""Bounded synthetic application steps for cross-GPU roofline collection.

The same code, shapes, seed, precision, and iteration count run on RTX 3090
and H200. No dataset or model download is required. A CUDA profiler range
allows NCU to capture only the measured steps, excluding model construction.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def dtype_from_name(torch, name):
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[name]


def make_resnet(torch, args, device, dtype):
    from torchvision.models import resnet50

    model = resnet50(weights=None, num_classes=1000).to(device=device, dtype=dtype)
    images = torch.randn(args.batch_size, 3, 224, 224, device=device, dtype=dtype)
    labels = torch.randint(0, 1000, (args.batch_size,), device=device)
    if args.mode == "resnet50_train":
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        def step():
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits.float(), labels)
            loss.backward()
            optimizer.step()
            return loss
    else:
        model.eval()

        def step():
            with torch.inference_mode():
                return model(images).float().mean()
    return step, model


def make_gpt2(torch, args, device, dtype):
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=50257,
        n_positions=max(args.seq_len + args.decode_tokens + 8, 512),
        n_ctx=max(args.seq_len + args.decode_tokens + 8, 512),
        n_embd=768,
        n_layer=12,
        n_head=12,
        use_cache=True,
    )
    model = GPT2LMHeadModel(config).to(device=device, dtype=dtype)
    tokens = torch.randint(0, config.vocab_size,
                           (args.batch_size, args.seq_len), device=device)

    training_modes = {"gpt2_train", "gpt2_shaped_train", "gpt2_memory_minimized"}
    if args.mode in training_modes:
        model.train()
        if args.mode == "gpt2_memory_minimized":
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=tokens, labels=tokens).loss
            loss.backward()
            optimizer.step()
            return loss
        return step, model

    model.eval()
    if args.mode == "gpt2_prefill":
        def step():
            with torch.inference_mode():
                return model(input_ids=tokens, use_cache=False).logits.float().mean()
        return step, model

    if args.mode == "gpt2_decode":
        def step():
            with torch.inference_mode():
                output = model(input_ids=tokens, use_cache=True)
                past = output.past_key_values
                next_token = output.logits[:, -1:].argmax(dim=-1)
                for _ in range(args.decode_tokens - 1):
                    output = model(input_ids=next_token, past_key_values=past, use_cache=True)
                    past = output.past_key_values
                    next_token = output.logits[:, -1:].argmax(dim=-1)
                return output.logits.float().mean()
        return step, model

    raise ValueError(f"unsupported GPT-2 mode: {args.mode}")


def profiler_flops(torch, step) -> float:
    """Return PyTorch's operator FLOP estimate for one complete application step."""
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, with_flops=True) as prof:
        step()
    torch.cuda.synchronize()
    return float(sum(event.flops for event in prof.key_averages() if event.flops))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=[
        "resnet50_train", "resnet50_infer", "gpt2_train", "gpt2_prefill",
        "gpt2_decode", "gpt2_shaped_train", "gpt2_memory_minimized",
    ])
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--physical-gpu-uuid", default="",
                        help="NVML UUID selected by the outer runner for audit")
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"],
                        default="float16")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--gap-ms", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile-range", action="store_true")
    parser.add_argument("--skip-flop-profiler", action="store_true",
                        help="skip torch.profiler in the NCU pass to avoid CUPTI conflicts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.batch_size < 1:
        parser.error("iterations and batch size must be positive; warmup must be nonnegative")
    if args.decode_tokens < 1 or args.seq_len < 1 or args.gap_ms < 0:
        parser.error("sequence/decode lengths must be positive and gap must be nonnegative")

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA PyTorch is required")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    dtype = dtype_from_name(torch, args.dtype)
    if args.mode.startswith("resnet50"):
        step, model = make_resnet(torch, args, device, dtype)
    else:
        step, model = make_gpt2(torch, args, device, dtype)

    flops_per_iteration = None
    if not args.skip_flop_profiler:
        flops_per_iteration = profiler_flops(torch, step)
        if not math.isfinite(flops_per_iteration) or flops_per_iteration <= 0:
            raise SystemExit("PyTorch profiler produced no usable FLOP estimate")
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize(device)

    events = []
    final_value = None
    if args.profile_range:
        torch.cuda.profiler.start()
    wall_start = time.perf_counter()
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        final_value = step()
        end.record()
        events.append((start, end))
        if args.gap_ms:
            time.sleep(args.gap_ms / 1000.0)
    torch.cuda.synchronize(device)
    wall_elapsed_s = time.perf_counter() - wall_start
    if args.profile_range:
        torch.cuda.profiler.stop()
    active_cuda_s = sum(start.elapsed_time(end) for start, end in events) / 1000.0
    total_flops = (None if flops_per_iteration is None
                   else flops_per_iteration * args.iterations)
    result = {
        "schema_version": 1,
        "case_id": args.case_id,
        "suite": args.suite,
        "platform": args.platform,
        "physical_gpu_uuid": args.physical_gpu_uuid,
        "repetition": args.repetition,
        "gpu_name": torch.cuda.get_device_name(device),
        "mode": args.mode,
        "device": str(device),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "decode_tokens": args.decode_tokens,
        "gap_ms": args.gap_ms,
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "flops_source": "torch.profiler.with_flops",
        "flops_per_iteration": flops_per_iteration,
        "total_flops": total_flops,
        "active_cuda_s": active_cuda_s,
        "wall_elapsed_s": wall_elapsed_s,
        "active_tflops": (None if total_flops is None
                           else total_flops / active_cuda_s / 1e12),
        "wall_tflops": (None if total_flops is None
                         else total_flops / wall_elapsed_s / 1e12),
        "final_scalar": None if final_value is None else float(final_value.detach()),
        "profile_range": bool(args.profile_range),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("useful_work " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
