# Remaining tests for the current ASPLOS paper

This directory is the submission-critical execution queue for the current
paper. It contains only evidence required to support the paper's final claims.
CUSUM, HMM/EWMA comparisons, optional sensor expansion, and broader platform
transfer are deferred to [`../next_paper/`](../next_paper/).

The current decision pipeline is:

```text
NVML/retained-sensor features -> random-forest window probability
                              -> fixed 3-of-5 rule at 0.75
                              -> run-level alert
```

The machine-readable tracker is [`CHECKLIST.csv`](CHECKLIST.csv). Update its
`status`, `owner`, `artifact_path`, and `generating_commit` fields in the same
commit that adds a result. The claim release gates are in
[`CLAIM_EVIDENCE_MATRIX.md`](CLAIM_EVIDENCE_MATRIX.md).

Robi's existing implementations and prior measurements have been audited in
[`ROBI_CODE_AUDIT.md`](ROBI_CODE_AUDIT.md) and
[`../results/tables/robi_available_runs.csv`](../results/tables/robi_available_runs.csv).
Those artifacts distinguish reusable code/prior evidence from results that
must be repeated under the frozen current-paper protocol.

## Submission-critical sequence

### 1. Freeze the evaluation

- Assign the four owners in `docs/OWNERS.md`.
- Freeze and hash split IDs, feature vocabulary, exclusions, false-alert
  budgets, the fixed 3-of-5 decision rule, and the sensor-retention rule.
- Record the preregistration SHA-256 and this protocol amendment in
  `docs/DECISION_LOG.md`.
- Confirm that no held-out test traces were used for feature, sensor, model, or
  threshold selection.

Gate:

```bash
make validate data-audit test
sha256sum configs/preregistration.yaml
```

### 2. Qualify synchronized electrical acquisition

For every instrumented GPU/channel pairing, collect idle, GEMM, memory-copy,
and burst traces. Report gain, noise floor, clipping, missingness, drift,
timestamp jitter, alignment, and the verified GPU/channel map.

Required artifacts:

- `results/electrical/calibration.csv`
- `results/electrical/channel_map.csv`
- `results/electrical/failure_ledger.csv`

Pass only if mapping is verified, clipping is absent or explained, and
alignment is below one tenth of the smallest retained temporal feature.

### 3. Collect the matched clean corpus and negative exposure

Collect synchronized, unprofiled NVML plus rail-power traces for training,
matched inference, and non-ML controls across GPU and day groups. Record useful
work, duration, energy, logger CPU, missingness, and failures. Run Nsight
characterization separately.

At least 71.9 zero-event negative GPU-hours are required before a one-sided
95% Poisson upper bound falls below 1/24 alerts/GPU-hour.

### 4. Freeze matched-overlap pairs

Create `results/paper/matched_roofline.csv` using
`docs/FIGURE_DATA_CONTRACT.md`. Freeze `pair_id` before inspecting sensor
predictions.

Two collection suites are ready. Previewing either suite is safe and does not
touch the GPU:

```bash
# Test 1: current-paper RTX 3090 application roofline.
python3 scripts/roofline/run_application_roofline.py \
  --gpu-index 0 --platform rtx3090-node2 --suite rtx3090_application

# Test 2: identical GPT-2 timing-shaping bridge cases for both GPU families.
python3 scripts/roofline/run_application_roofline.py \
  --gpu-index 0 --platform rtx3090-node2 --suite cross_gpu_bridge
```

The application pairs are declared before collection in
`configs/application_roofline_pairs.csv`; the script never chooses the closest
pairs after seeing the measurements. After three repetitions, materialize the
fixed RTX pairs with:

```bash
python3 scripts/roofline/materialize_application_pairs.py \
  --points results/roofline/applications/rtx3090-node2/rtx3090_application/application-roofline-points.csv \
  --output results/paper/matched_application_roofline.csv
```

After verifying that the selected GPU is idle, execute with `--execute` and
provide the sustained FP16 compute and DRAM-bandwidth ceilings measured under
the same clock/power settings:

```bash
python3 scripts/roofline/run_application_roofline.py \
  --gpu-index 0 --platform rtx3090-node2 --suite rtx3090_application \
  --peak-tflops RTX_MEASURED_TFLOPS --peak-gbps RTX_MEASURED_GBPS --execute

python3 scripts/roofline/run_application_roofline.py \
  --gpu-index 0 --platform rtx3090-node2 --suite cross_gpu_bridge \
  --peak-tflops RTX_MEASURED_TFLOPS --peak-gbps RTX_MEASURED_GBPS --execute
```

Run the same `cross_gpu_bridge` command on the H200 with a distinct platform
label and its measured ceilings. Then join the two outputs:

```bash
python3 scripts/roofline/compare_cross_gpu_roofline.py \
  --input results/roofline/applications/rtx3090-node2/cross_gpu_bridge/application-roofline-points.csv \
  --input results/roofline/applications/h200-nvl/cross_gpu_bridge/application-roofline-points.csv \
  --output results/paper/cross_gpu_roofline_link.csv
```

The bridge uses the same synthetic GPT-2 124M architecture, shapes, precision,
seeds, and gaps on both GPUs. It does not pretend that raw RTX and H200 TFLOP/s
are directly comparable: arithmetic intensity and throughput are normalized by
each GPU's measured ridge point. Nsight runs remain separate from unprofiled
sensor traces. FLOP counts are PyTorch-profiler estimates, so one case should
be checked against an analytical or NCU FLOP count before using absolute
TFLOP/s in the paper; the identical code path still supports the normalized
cross-GPU comparison.

### 5. Collect held-out evasions and tamper cases

The untouched evaluation must include:

- fused-update/custom-kernel training;
- power or duty-cycle shaping;
- training/inference interleaving;
- GPU migration;
- one sensor-aware mimicry attack;
- NVML freeze/replay, bias/scale, clipping, sample drop, timestamp shift, and
  GPU/channel swap; and
- physical-channel disconnect or failure.

Each compute evasion must report useful optimization progress and energy per
progress. An attack that stops meaningful optimization is not successful.

### 6. Screen sensors on validation only

Evaluate sensor-only and NVML-plus-sensor random forests on identical grouped
folds. Retain a modality only when its paired 95% interval excludes zero,
relative worst-family miss rate falls by at least 10%, monitor overhead is
below 1%, and its channel-health test passes. Keep failed modalities in the
paper table.

Also run the lightweight cross-evasion diagnostic with the same two-stage
random forest, 30-second windows, 15-second stride, and fixed 3-of-5 rule:

1. **Pairwise transfer:** train on ordinary data plus evasion family A, then
   test on a different family B.
2. **Leave-one-evasion-out:** train on ordinary data plus every available
   development evasion except B, then test on B.

Use a preassigned `control_test` set to measure false alerts; those control
runs must never enter model fitting. Start from
`configs/evasion_transfer_labels.example.csv`, replace its illustrative rows
with real run IDs and trace paths, and run:

```bash
python3 scripts/analysis/evasion_transfer.py \
  --labels results/evaluation/evasion_transfer_labels.csv \
  --output results/evaluation/evasion-transfer.json
```

The command writes a source-to-target transfer matrix and run-level
predictions. It always excludes `fused_update_kernel`, even if that family is
listed in the labels file. The excluded name is read from the preregistration,
not from a command-line override. This known, preregistered final family is not
used by the development diagnostic; only newly collected current-test traces
are evaluated after the pipeline is frozen.

For the requested two-seen/one-model-unseen test, use timing shaping and
interleaving as represented families and exclude memory minimization from all
fitting:

```bash
python3 scripts/analysis/evasion_transfer.py \
  --labels results/evaluation/evasion_transfer_labels.csv \
  --family-plan configs/evasion_two_seen_one_unseen.yaml \
  --output results/evaluation/evasion-transfer.json
```

This produces three comparisons: timing-shaping to memory-minimization,
interleaving to memory-minimization, and both seen families together to
memory-minimization. `fused_update_kernel` remains excluded. “Model-unseen”
means absent from fitting; it does not mean the research team was unaware of
the attack concept.

### 7. Freeze fusion and the fixed run-level rule

Compare NVML, each retained sensor, add-one, leave-one-out, best pair, selected
set, and all sensors on validation only. Select the smallest sensor set within
one standard error of the best worst-family TPR. Freeze the random-forest
model, calibration, 3-of-5 rule, 0.75 threshold, health policy, split hash, and
code hash before opening the final test set.

### 8. Run the untouched test once and complete statistics

Report run-level worst-family TPR, false alerts/GPU-hour, hierarchical-bootstrap
intervals, PR-AUC, calibration, first 3-of-5 time to alert with undetected runs
right-censored, useful-work penalty, and per-family confusion counts. Pair
sensor improvements on identical runs and Holm-correct multiple comparisons.

### 9. Derive the minimum current-sensor specification

For retained electrical sensing, sweep sample rate, bit depth, dynamic range,
and duty cycle with causal filtering. Select the lowest-cost point within one
standard error of full-fidelity validation performance and confirm it with a
small hardware-in-loop rerun. Report bytes/GPU-hour, logger CPU, latency, and
useful-work overhead.

### 10. Regenerate and audit the paper

Populate the required schemas in `docs/FIGURE_DATA_CONTRACT.md`, regenerate
all tables and figures from a clean checkout, compile with the official ASPLOS
class, and verify that every numerical sentence maps to a committed artifact.

```bash
python3 -m pip install -r requirements-analysis.txt
make data-audit test validate baseline-audit figures
git diff --exit-code
```

Do not replace uncollected measurements with simulated values.
