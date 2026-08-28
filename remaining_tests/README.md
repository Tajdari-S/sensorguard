# Remaining tests before the ASPLOS submission

This directory is the execution queue for evidence that is still missing from
the paper. It separates completed pipeline diagnostics from measurements that
must be collected before a claim can move into the abstract or primary-results
table.

The machine-readable tracker is [`CHECKLIST.csv`](CHECKLIST.csv). Update its
`status`, `owner`, `artifact_path`, and `generating_commit` fields in the same
commit that adds a result. The claim-level release gates are in
[`CLAIM_EVIDENCE_MATRIX.md`](CLAIM_EVIDENCE_MATRIX.md).

## P0: submission-critical

### 1. Freeze the evaluation before collecting more test data

- Assign the four owners in `docs/OWNERS.md`.
- Freeze and hash split IDs, feature vocabulary, exclusions, false-alert
  budgets, CUSUM selection, and the sensor-retention rule.
- Record the preregistration SHA-256 in `docs/DECISION_LOG.md`.
- Confirm that no held-out test traces have been used for feature, sensor, or
  threshold selection.

Gate:

```bash
make validate data-audit test
sha256sum configs/preregistration.yaml
```

### 2. Qualify synchronized electrical acquisition

For each electrically instrumented GPU/channel pairing, collect idle, GEMM,
memory-copy, and burst traces. Rotate channels where practical. Report gain,
noise floor, clipping, missingness, drift, timestamp jitter, and alignment.

Required artifacts:

- `results/electrical/calibration.csv`
- `results/electrical/channel_map.csv`
- `results/electrical/failure_ledger.csv`
- synchronized manifests with GPU UUID, channel ID, scope settings, ambient
  condition, and generating commit.

Pass only if GPU/channel mapping is verified, clipping is explained or absent,
and alignment is below one tenth of the smallest retained temporal feature.

### 3. Collect the matched clean corpus

Collect synchronized, unprofiled NVML plus rail-power traces for ordinary
training, matched inference, and non-ML controls across GPU and day groups.
Record useful work, duration, energy, logger CPU, missingness, and failures.
Run Nsight characterization separately; never train on profiled traces.

Minimum output: enough training and negative exposure in every planned test
group to make the fixed false-alert target statistically interpretable. At
least 71.9 zero-event negative GPU-hours are required before a one-sided 95%
Poisson upper bound falls below 1/24 alerts/GPU-hour.

### 4. Freeze the matched-overlap pairs

Create `results/paper/matched_roofline.csv` using the schema in
`docs/FIGURE_DATA_CONTRACT.md`. Match without inspecting sensor predictions,
freeze `pair_id`, then generate the classical roofline.

### 5. Collect held-out evasions and tamper cases

The untouched evaluation must include at least:

- fused-update/custom-kernel training;
- power or duty-cycle shaping;
- training/inference interleaving;
- GPU migration;
- one sensor-aware mimicry attack;
- NVML freeze/replay, bias/scale, clipping, sample drop, timestamp shift, and
  GPU/channel swap;
- physical-channel disconnect or failure.

Each compute evasion must report useful optimization progress and energy per
progress. An attack that stops meaningful optimization is not a successful
evasion.

### 6. Screen sensors on validation only

Evaluate sensor-only and NVML-plus-sensor models on identical grouped folds.
Retain a modality only when its paired 95% interval excludes zero, relative
worst-family miss rate falls by at least 10%, monitor overhead is below 1%, and
the channel-health test passes. Failed modalities remain in the paper table.

### 7. Freeze fusion and sequential detection

Compare NVML, each retained sensor, add-one, leave-one-out, best pair, selected
set, and all sensors. Calibrate probabilities and CUSUM using validation only.
Commit the frozen model, threshold, health policy, split hash, and code hash
before opening the final test set once.

### 8. Run the untouched test once and complete statistics

Report run-level worst-family TPR, false alerts/GPU-hour, Wilson or hierarchical
bootstrap intervals, PR-AUC, calibration, time to alert with undetected runs
right-censored, useful-work penalty, and per-family confusion counts. Pair
sensor improvements on identical runs and Holm-correct multiple comparisons.

## P1: strong additions

- Complete WAVE trace-size, solver-time, and recovered-parameter tables.
- Run NVML replay/freeze/shift and cross-modal residual tests.
- Measure privacy probes and storage/CPU cost for every retained modality.
- Confirm the selected system on all available RTX 3090 GPUs when feasible.
- Add contact temperature only if it supplies conditional value beyond
  electrical sensing.

## P2: run only after the validation gate

- Ultrasound, RF, thermal-camera, and mirrored-Ethernet expansion.
- Exhaustive three-sensor combinations.
- Broad cross-platform or cloud-GPU transfer.

Cut these before cutting held-out attacks, confidence intervals, useful-work
measurement, or the fixed-threshold run-level evaluation.

## Final artifact and paper gate

Populate the six headline schemas in `docs/FIGURE_DATA_CONTRACT.md`, regenerate
all tables and figures from a clean checkout, compile with the official ASPLOS
class, and verify that every numerical sentence maps to a committed artifact.

```bash
python3 -m pip install -r requirements-analysis.txt
make data-audit test validate baseline-audit figures
git diff --exit-code
```

Do not replace uncollected physical measurements with simulated values.
