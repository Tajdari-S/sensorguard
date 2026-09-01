# Fused-update recovery

## What failed

The original frozen SensorGuard detector did not alert on any of the three
sealed fused-update runs. This result is final and is not replaced by the
work below. NVML, electrical current, and their fusion each detected 0/3.

The fused-update and dummy-write control execute nearly identical matrix work.
The semantic difference is that fused update changes model weights while the
control writes a gradient to scratch memory. NVML and one 10 kS/s GPU-current
trace therefore have little zero-shot information about the destination of the
write.

## Diagnostics completed

- Adding ordinary AdamW examples without fused examples did not transfer:
  fused-update detection remained 0/3 for random forest, Extra Trees, and
  logistic models across NVML, electrical, and fused features.
- CUSUM did not recover any fused-update run.
- Adding electrical drift features did not make one-repetition calibration
  stable across all train/calibrate/test rotations.
- Once the opened fused-update runs were treated as development data, pooled
  leave-one-repetition-out evaluation found a lightweight recovery candidate.

## Frozen recovery candidate

- Sensor: one GPU-current channel at 10 kS/s; NVML is not required for this
  candidate.
- Model: 400-tree Extra Trees classifier.
- Window: 30 seconds with 15-second stride.
- Alert: probability at least 0.25 for three consecutive windows.
- Development-only result: 3/3 fused-update runs and 3/3 AdamW positive
  controls detected; 0/9 matched-control alerts; median fused alert time 60 s.
- Frozen artifact SHA-256:
  `2f922014b1a132b20cf605746584dea074d6a72bcbc91607589bf3389a9db648`.

These are post-sealed development results, not a new sealed claim.

## Fresh frozen validation result

The artifact above was committed before collection and evaluated once on three
fresh configurations: smaller/deeper, the original shape with a new seed, and
wider/shallower. Each configuration included fused update, forward3, AdamW,
dummy write, and forward. All 15 node runs and all 15 paired PicoScope captures
returned code 0. Every trace contains 999,156 samples from serial `12789/2929`,
with zero overflow and zero clipping. All six learning runs made meaningful
optimization progress.

The frozen detector alerted on 3/3 fused-update and 3/3 AdamW runs, but also on
9/9 matched controls. This is 48 false alerts per GPU-hour over 0.1875 negative
GPU-hours, so the recovery candidate is rejected. Attempts 1--4 were never
scored: they exposed pre-measurement setup, CUDA-to-physical-GPU mapping, and
scope-cadence errors. Attempt 5 is the only valid fresh validation corpus.

A held-configuration diagnostic also failed. Training on the first two fresh
configurations did not detect the third fused variant with random forest or
Extra Trees; logistic regression instead alerted on its matched controls.
CUSUM cannot repair probabilities with this ordering.

## What is required to solve this family

Aggregate current and NVML observe the amount and timing of work, not whether a
matrix result is written to a parameter or a scratch buffer. The fused-update
attack and dummy-write control intentionally make those observable operations
nearly identical. Therefore no new RF threshold, sensor subset, or sequential
rule on the present signals is supported as a solution.

A defensible fused-update detector requires one additional trusted semantic
observable, such as secret randomized parameter-block hashing, a trusted model
weight-integrity sentinel, or protected write-address telemetry from the GPU
driver/hypervisor. Freeze that mechanism and its alert rule, then collect a new
unseen fused implementation plus matched dummy-write controls. Without that
additional trust or telemetry, report fused update as an observability limit;
do not claim that SensorGuard detects it.
