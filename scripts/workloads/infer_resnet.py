#!/usr/bin/env python3
"""P0 inference adapter: ResNet-50 image classification (matched control for
train_resnet_cifar10 — same model/precision/batch, no backward pass)."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def raw_now() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--duration-s", type=float, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default=str(Path.home() / "datasets/cifar10"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    ds = datasets.CIFAR10(args.data_root, train=False, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    model = models.resnet50(num_classes=10).to(device).eval()
    end = raw_now() + args.duration_s
    samples = 0
    t0 = raw_now()
    with torch.inference_mode():
        while raw_now() < end:
            for x, _ in loader:
                model(x.to(device, non_blocking=True))
                samples += x.size(0)
                if raw_now() >= end:
                    break
    torch.cuda.synchronize(device)
    elapsed = raw_now() - t0
    print("useful_work " + json.dumps({
        "mode": "infer_resnet50", "samples": samples,
        "samples_per_s": round(samples / elapsed, 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
