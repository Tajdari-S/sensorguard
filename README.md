# SensorGuard: ASPLOS 2027 submission sprint

Public collaboration repository for **Beyond NVML: Sensor-Fused Hardware Telemetry for Robust AI Compute Governance**.

## Deadline

ASPLOS 2027 has **no separate abstract deadline** for the September cycle. The full paper is due **September 9, 2026 (Anywhere on Earth)**. We target a frozen, upload-ready PDF by **September 8**.

Official CFP: https://www.asplos-conference.org/asplos2027/cfp/

## Submission claim

Determine the smallest trusted sensor configuration that improves held-out training-evasion detection over a frozen NVML baseline at a fixed run-level false-alert budget, while meeting deployment overhead and privacy constraints.

## Four role lanes

Collaborators should put their names beside one primary lane in `docs/OWNERS.md`. Roles are deliberately unassigned until the team confirms them.

1. Paper and integration
2. Telemetry and physical sensors
3. Workloads and adversarial evaluation
4. Analysis, statistics, and reproducibility

## Critical path

1. Freeze the protocol, inventory, data schema, splits, false-alert budget, and sensor-retention rule.
2. Reproduce the NeurIPS NVML baseline on the RTX 3090 system.
3. Establish a small WAVE reproduction on one RTX 3090 and quantify profiling overhead.
4. Collect synchronized NVML plus electrical traces on matched training/non-training/evasion workloads.
5. Screen each additional sensor on validation only; drop weak modalities at the gate.
6. Freeze the random-forest fusion model and fixed 3-of-5 rule, then evaluate once on held-out GPU/day/application/attack groups.
7. Derive minimum sampling-rate and bit-depth requirements and confirm them in hardware.
8. Freeze claims, figures, and the paper before the final upload day.

The dated plan is in [`docs/TIMELINE.md`](docs/TIMELINE.md). The exact tests and acceptance criteria are in [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md). The complete named application/evasion registry and its feasibility status are in [`docs/WORKLOAD_COVERAGE.md`](docs/WORKLOAD_COVERAGE.md); daily commands and gates are in [`docs/DAILY_RUNBOOK.md`](docs/DAILY_RUNBOOK.md); executable roofline tests are in [`docs/ROOFLINE_RUNBOOK.md`](docs/ROOFLINE_RUNBOOK.md).

The live, machine-readable queue of unfinished evidence is in
[`remaining_tests/`](remaining_tests/). Update that tracker with each result so
diagnostic, validation, and untouched-test evidence cannot be conflated.

## Repository map

```text
configs/     Frozen experiment and manifest templates
data/        Local mount points only; raw traces are never committed
docs/        Timeline, test protocol, system inventory, decisions, ownership
next_paper/  Deferred CUSUM and optional extension roadmap
paper/       Working LaTeX draft
results/     Small derived tables/figures only
scripts/     Inventory and validation helpers
```

## First-day checklist

- [ ] Fill `docs/OWNERS.md` with the four collaborators.
- [ ] Complete every `TBD` in `docs/SYSTEM_SPEC.md`.
- [ ] Run `bash scripts/capture_inventory.sh results/inventory`.
- [ ] Fill and freeze `configs/preregistration.yaml`.
- [ ] Validate it with `python3 scripts/validate_preregistration.py configs/preregistration.yaml`.
- [ ] Validate application coverage with `python3 scripts/check_workload_coverage.py`.
- [ ] Run `make roofline-test`; after CUDA/PyTorch setup, run `make roofline-smoke`.
- [ ] Review every `blocked_exact_name` and `external_hardware` workload before freezing scope.
- [ ] Record a single synchronization pulse across all active sensor loggers.
- [ ] Run one idle, GEMM, memory-copy, and burst calibration trace.
- [ ] Confirm the raw-data storage location, quota, backups, and deletion policy.

## Non-negotiable rules

- Split by **run**, never by windows from the same run.
- Tune thresholds and sensor selection on development/validation data only.
- Evaluate the held-out test split exactly once after the model and decision rule are frozen.
- Run Nsight/WAVE characterization separately from unprofiled physical-sensor collection.
- Keep failed and excluded runs in an audit ledger with a preregistered reason.
- Telemetry, scope, and profiler traces ARE committed, with binary formats via git LFS (decision 2026-08-25, following github.com/robirahman/GPU-monitoring conventions). Camera and network captures are still never committed.
- Do not claim a sensor helps unless its paired run-bootstrap interval excludes zero and it passes the retention rule.

## What the scripts currently do

The repository contains runnable inventory, preregistration validation, workload-coverage validation, CSV queue generation, and roofline microbenchmark/plot tools. The roofline microbenchmarks execute when CUDA PyTorch is installed; the Nsight wrapper executes when `ncu` is available. Full ML/HPC application adapters still need to be ported, pinned, smoke-tested, and connected to synchronized sensor logging. `scripts/run_matrix.py` is deliberately a non-executing planner so a placeholder can never be mistaken for a completed experiment.

## Reproduce the current evidence snapshot

Install the pinned analysis dependencies, repair/audit early label metadata,
run all tests, and regenerate the figures and tables:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-analysis.txt
make data-audit test validate baseline-audit figures
```

Generated publication assets live under `results/figures/` and
`results/tables/`. `docs/FIGURE_DATA_CONTRACT.md` defines the schemas for the
headline electrical, matched-roofline, operating-curve, time-to-alert,
ablation, and hardware-sweep figures. Plotting code must reject missing inputs;
it must never replace uncollected measurements with simulated values.

The current `e2_baseline_combined.json` is a run-grouped pipeline diagnostic,
not a held-out transfer result. New baseline runs should save out-of-fold
window/run predictions and use `--group-by family`, `--group-by gpu_uuid`, and
`--group-by collection_day` as separate generalization audits before the
frozen untouched-test evaluation.

The committed protocol audit exposes the present limitation instead of hiding
it: run-grouped and leave-GPU-out evaluation each detect 22/23 training runs
with 0/95 false-positive runs, whereas leave-family-out detects 0/23 training
runs and raises 10/95 false-positive runs. One direction of the two-day audit
is non-estimable because the first day contains no training runs. These are
pipeline/generalization audits, not final sensor-fusion results.
