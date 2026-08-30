# GitHub issue backlog

Create these issues after the private repository is available. Assign only after the four GitHub usernames are verified.

## P0 submission-critical

### 1. [P0][Aug 14] Complete system inventory and safety qualification

- Fill every `TBD` in `docs/SYSTEM_SPEC.md`.
- Run the inventory script and add a topology diagram.
- Verify scope/probe isolation, limits, and channel-to-GPU mapping.
- Exit: independent reviewer signs off before energized rail measurements.

### 2. [P0][Aug 15] Freeze preregistration, splits, and false-alert budget

- Resolve all values in `configs/preregistration.yaml`.
- Freeze group IDs and held-out GPU/evasion family.
- Hash the preregistration and record it in `docs/DECISION_LOG.md`.
- Exit: validator passes; no test trace has been inspected.

### 3. [P0][Aug 17] Calibrate and synchronize active sensor channels

- Run idle, GEMM, memcpy, and burst tests on every GPU/channel permutation.
- Quantify drift, clipping, missingness, jitter, alignment, and noise.
- Exit: calibration report declares pass/fail for each channel.

### 4. [P0][Aug 18] Reproduce the NeurIPS NVML baseline

- Strict 1 Hz/166-feature/two-stage RF reproduction.
- Separate RTX-3090-tuned model using development data only.
- Report run-level fixed-threshold metrics and overhead.
- Exit: feature code and baseline thresholds are frozen.

### 5. [P0][Aug 21] Collect synchronized core and overlap corpus

- Training, matched inference, non-ML, and hard-evasion runs.
- At least three independent screening repetitions.
- Separate unprofiled sensor and profiler characterization passes.
- Exit: run audit has commands, seeds, UUIDs, health, and useful work.

### 6. [P0][Aug 21] Freeze roofline-overlap matched pairs

- Derive FLOPs and DRAM bytes without contaminating detector traces.
- Match by the preregistered four-dimensional caliper.
- Exit: pair IDs and sensitivity analysis are frozen before sensor scoring.

### 7. [P0][Aug 22] Gate 1: retain or drop each sensor

- Evaluate sensor-only and NVML+sensor on validation folds.
- Apply the exact miss-reduction/CI/overhead rule.
- Exit: accepted and rejected sensor table, including negative results.

### 8. [P0][Aug 27] Complete held-out adversarial corpus

- At least five unseen strategies plus telemetry tamper/failure cases.
- Record throughput, optimizer progress, and energy.
- Exit: entire attack family remains held out from initial defender training.

### 9. [P0][Aug 29] Freeze fusion, fixed 3-of-5 rule, and test protocol

- Compare random-forest sensor configurations on validation.
- Choose smallest set within one standard error.
- Exit: model, calibration, health policy, 3-of-5 rule, 0.85 threshold, and commit hash frozen.

### 10. [P0][Aug 30] Gate 2: open the test split once and freeze claims

- Produce primary table, per-family breakdown, confidence intervals, calibration, and time-to-alert.
- Exit: claim/evidence matrix signed by two collaborators.

### 11. [P0][Sep 2] Derive and confirm minimum electrical-sensor specification

- Causal filtering, rate and bit-depth sweep, duty-cycle sweep.
- Confirm selected point with new hardware-in-loop runs.
- Exit: minimum rate/bits/dynamic range and resource costs.

### 12. [P0][Sep 8] Produce and verify final anonymous ASPLOS PDF

- Clean-checkout figure regeneration.
- Internal technical, statistical, overlap, anonymization, and formatting reviews.
- Upload candidate one day early and inspect rendered PDF.

## P1 strengthening

### 13. [P1][Aug 19] Reproduce WAVE on one RTX 3090

Pin artifact, map Ampere counters, run representative models, and report recovery fidelity plus profiling overhead. Do not expand to all GPUs unless stable by the deadline.

### 14. [P1][Aug 22] Compare contact temperature against thermal imaging

Retain the cheaper channel if it lies within one standard error of the thermal camera on validation worst-family TPR.

### 15. [P1][Sep 2] Evaluate cross-modal tamper residuals and privacy leakage

Test replay/freeze/shift/channel swap and model/application identity probes using retained features.

## P2 conditional

### 16. [P2][Aug 22] Decide whether ultrasound adds conditional value

Screen ultrasound conditional on NVML plus electrical sensing. Drop it unless it improves the frozen validation endpoint and passes the health, privacy, and overhead criteria. **RF/SDR resolved 2026-08-29: DROPPED** — bench test produced no usable GPU-correlated signal (receiver works, picks up radio stations).

### 17. [P2][Aug 22] Decide whether mirrored Ethernet is in scope

Validate the 10Gtek NIC plus TP-Link TL-SG108E port-mirroring path. Drop for single-node experiments unless observed Ethernet traffic is relevant; it cannot observe PCIe/NVLink collectives.
