#!/usr/bin/env python3
"""Fill GPU UUID and collection day from committed inventory/manifests.

The operation is deterministic and preserves the label and family columns.
It exists to repair early campaign CSVs that left ``gpu_uuid`` empty.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


INVENTORIES = {
    "node2": Path("results/inventory-node2/gpus.csv"),
    "verifier": Path("results/inventory-verifier/gpus.csv"),
    "testbed-node1": Path("results/inventory-testbed-node1/gpus.csv"),
}


def host_from_run_id(run_id: str) -> str:
    for host in sorted(INVENTORIES, key=len, reverse=True):
        if f"_{host}-gpu" in run_id:
            return host
    raise ValueError(f"cannot infer inventory host from run_id={run_id!r}")


def inventory_maps() -> dict[str, dict[int, str]]:
    maps = {}
    for host, path in INVENTORIES.items():
        frame = pd.read_csv(path, skipinitialspace=True)
        frame.columns = [column.strip() for column in frame.columns]
        maps[host] = {
            int(row["index"]): str(row["uuid"]).strip()
            for _, row in frame.iterrows()
        }
    return maps


def repair(path: Path) -> None:
    runs = pd.read_csv(path)
    mappings = inventory_maps()
    uuids = []
    days = []
    for _, row in runs.iterrows():
        run_id = str(row["run_id"])
        host = host_from_run_id(run_id)
        gpu_index = int(row["gpu_index"])
        uuids.append(mappings[host][gpu_index])
        manifest_path = Path("data/runs") / run_id / "manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text())
        days.append(str(manifest["start_utc"])[:10])
    runs["gpu_uuid"] = uuids
    runs["collection_day"] = days
    runs.to_csv(path, index=False)
    print(f"Repaired {len(runs)} rows in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.labels:
        repair(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
