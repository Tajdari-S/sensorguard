#!/usr/bin/env python3
"""P0 training adapter: ResNet-50 on CIFAR-10 (supervised, single GPU).

Deterministic seed, duration- or epoch-bounded, reports useful work
(samples/s and final loss) for the manifest. Dataset downloads to
~/datasets/cifar10 on first use (pinned by torchvision version).
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    ds = datasets.CIFAR10(args.data_root, train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    model = models.resnet50(num_classes=10).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    model.train()

    end = raw_now() + args.duration_s
    samples = 0
    loss_val = float("nan")
    t0 = raw_now()
    while raw_now() < end:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            samples += x.size(0)
            loss_val = float(loss.detach())
            if raw_now() >= end:
                break
    torch.cuda.synchronize(device)
    elapsed = raw_now() - t0
    print("useful_work " + json.dumps({
        "mode": "train_resnet50_cifar10", "samples": samples,
        "samples_per_s": round(samples / elapsed, 2), "final_loss": round(loss_val, 4)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
