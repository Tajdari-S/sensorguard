#!/usr/bin/env python3
"""P0 diffusion inference adapter: DDPM image generation
(google/ddpm-cifar10-32 — small pinned model, heavy iterative UNet load)."""

import argparse
import json
import time

import torch


def raw_now() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--duration-s", type=float, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="google/ddpm-cifar10-32")
    args = parser.parse_args()

    from diffusers import DDPMPipeline

    device = torch.device(args.device)
    pipe = DDPMPipeline.from_pretrained(args.model).to(device)
    pipe.set_progress_bar_config(disable=True)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    end = raw_now() + args.duration_s
    t0 = raw_now()
    images = 0
    while raw_now() < end:
        pipe(batch_size=args.batch_size, num_inference_steps=args.steps,
             generator=gen, output_type="np")
        images += args.batch_size
    torch.cuda.synchronize(device)
    print("useful_work " + json.dumps({
        "mode": "infer_diffusion_ddpm", "images": images,
        "images_per_s": round(images / (raw_now() - t0), 3),
        "steps": args.steps}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
