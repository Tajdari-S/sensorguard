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

## Required validation

Evaluate the frozen artifact once on three fresh fused-update configurations
and their matched forward, three-forward, dummy-write, and AdamW controls. Do
not change the model, features, threshold, or run rule after opening those
runs. If all three fresh fused configurations are detected with zero matched
control alerts, report this as a corrective within-family validation. A new
unseen attack family is still required for a new zero-shot sealed-family claim.
