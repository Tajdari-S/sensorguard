# Experiment protocol and decision criteria

## E0: preregistration and leakage control

**Purpose:** prevent post-hoc sensor and threshold selection.

Steps:

1. Freeze the primary endpoint: run-level true-positive rate at a fixed global false-alert budget. The proposed primary budget is one false alert per 24 GPU-hours, with sensitivity at one per 8 and one per 72 GPU-hours; the team must confirm this choice by August 15.
2. Group by run, GPU, day, application configuration, and evasion family.
3. Hold out at least one GPU and one entire evasion family.
4. Freeze the false-alert budget, fixed 3-of-5 rule and threshold, sensor-retention rule, and exclusion policy.
5. Hash the preregistration, split IDs, and feature vocabulary.

Pass: validator succeeds and no final-test traces are visible to sensor/model selection.

## E1: sensor calibration and synchronization

For every active channel, collect idle, fixed GEMM, memory copy, and bursty-load traces.

Measure noise floor, gain, clipping, drift, missingness, timestamp jitter, trigger alignment, and cross-GPU/channel variability. Rotate scope channels across GPUs where practical. Record ambient inlet temperature and start thermal runs only inside the frozen starting-temperature band.

Pass: no unexplained clipping; missingness below the preregistered limit; alignment error below one-tenth of the smallest retained temporal feature; GPU/channel mapping verified by blinded activation.

## E2: NeurIPS NVML reproduction

1. Collect the same nine NVML signals at 1 Hz.
2. Reproduce the 166-feature, two-stage random-forest, 30-second causal window, 15-second stride configuration.
3. Report strict published hyperparameters and a separately labeled RTX-3090-tuned version.
4. Split by run, not window; report run-level alerts and the earlier 3-of-5 rule at threshold 0.75.
5. Measure logger overhead and data loss.

Output: per-family AUC/PR-AUC, fixed-threshold TPR, false alerts/GPU-hour, time to alert, and 95% hierarchical-bootstrap intervals.

## E3: WAVE reproduction on Ampere

1. Pin the public WAVE artifact commit and environment.
2. Run `ncu --query-metrics`; map every requested counter and record unavailable/substituted metrics.
3. Reproduce representative GPT-2, LLaMA, Qwen, and split-layer cases on one RTX 3090.
4. Measure recovered layer/token/batch/hidden/FFN parameters, error bounds, solver time, trace size, and runtime slowdown.
5. Profile counter families and single counters separately to estimate marginal overhead.

Decision: expand beyond one GPU only if the adapted reproduction is stable by August 19. WAVE is an offline reference, not an always-on monitor.

## E4: workload corpus and matched overlap

Minimum classes:

- Training: ResNet-50, BERT, GPT-2; 1B/3B/7B decoder fine-tuning where feasible; LoRA; DDP; gradient accumulation; checkpointing.
- Inference: matched models/precision/batch; LLM prefill/decode; continuous batching; image, diffusion, and speech inference as available.
- Non-ML: GEMM, FFT, SpMV/stencil, N-body, memcpy, hashing/mining proxy, Blender, video encode, network stress, idle.
- Evasion: low utilization, idle dilution, preallocation, power/clock shaping, inference interleaving, fake collectives, LoRA/frozen backbone, training-as-inference, fused update, GPU migration.

For each configuration, collect an unprofiled synchronized trace and a separate characterization pass for FLOPs and DRAM bytes. Match training and non-training runs in standardized log operational intensity, log FLOP/s, mean NVML power, and mean GPU utilization using a 0.25 pooled-standard-deviation caliper; report sensitivity at 0.1 and 0.5. Freeze pair IDs before sensor evaluation.

## E5: single-sensor screening

For each modality, compare sensor-only and NVML+sensor on identical grouped folds, including the overlap and unseen-family subsets. Record storage, host CPU, setup cost, privacy exposure, and failure rate.

Retain a sensor only if all are true:

- worst-family miss rate falls by at least 10% relative at the same false-alert budget;
- the paired 95% run-bootstrap interval for improvement excludes zero;
- monitoring overhead is below 1%; and
- the modality passes its health/reliability check.

Failed modalities remain in the paper's ruled-out table.

## E6: fusion and fixed run-level detection

Using validation only, compare NVML; each sensor; NVML plus each sensor; best pair; best triple if time permits; all sensors; selected set; and leave-one-out variants. Use random-forest window classifiers and choose the smallest sensor set within one standard error of the best validation worst-family TPR.

Convert window probabilities to run alerts with the fixed 3-of-5 rule at probability 0.75. Freeze the model, calibration transform, health policy, rule, threshold, and code commit before opening the held-out test set once. Measure time to alert at the first causal 3-of-5 trigger and right-censor undetected runs.

## E7: adversarial and tamper robustness

Run at least five held-out attack families: fused-update training, power/duty-cycle shaping, training-inference interleaving, GPU migration, and one sensor-aware mimicry attack. Always report useful work and energy; an attack that stops meaningful optimization is not successful evasion.

Inject NVML freeze/replay, bias, scale, clipping, sample drop, timestamp shift, and GPU-channel swap. Then fail one physical channel at a time. Report direct classification, health-alarm detection, graceful degradation, and residual-based tamper detection.

## E8: minimum hardware specification

From full-fidelity retained traces, causally filter before downsampling. Sweep electrical sample rate, 16/12/10/8/6/4 effective bits, dynamic range, duty cycle, thermal ROI reduction, ultrasound sample rate/band aggregation, and mirrored-Ethernet timestamp/bin granularity as applicable. Retrain inside identical grouped folds. (RF/SDR bandwidth sweep dropped 2026-08-29 — modality yields no usable GPU-correlated signal.)

Select the lowest-cost point within one standard error of full-fidelity validation performance. Confirm that point with a small hardware-in-loop rerun. Report latency, bytes/GPU-hour, logger CPU, useful-work overhead, and privacy exposure.

## E9: final statistical audit

- Hierarchical bootstrap: resample workload families, then runs; preserve windows/modalities within a run.
- Pair improvements on identical runs; Holm-correct multiple sensor comparisons.
- Report calibration, per-family results, worst-family TPR, false alerts/GPU-hour, time to alert, and useful-work penalty.
- Regenerate every paper figure/table from a clean checkout and immutable processed data.
