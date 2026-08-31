# Additional sensor-subset and CUSUM tests

## NVML-anchored sensor subsets

All eight available configurations were tested on the same 36 paired RTX 3090
runs using leave-one-workload-family-out evaluation and the fixed 0.85,
3-of-5 rule.

- NVML: 18/26 training runs detected and 0/10 false alerts.
- NVML + GPU current: 22/26 detected and 0/10 false alerts. The gain is 15.4
  percentage points (paired run bootstrap 95% interval: 3.85 to 30.77 points).
- NVML + motherboard current: identical decisions to NVML.
- NVML + ultrasound: identical decisions to NVML.
- NVML + motherboard current + ultrasound: identical decisions to NVML.
- Every subset containing GPU current: 22/26 detected and 0/10 false alerts;
  adding motherboard current or ultrasound changes no run decision.

The smallest empirically best subset is therefore **NVML + GPU current**.
Adding the other available sensors increases the feature count without a
conditional detection gain. Every subset still has 0% worst-family detection,
principally because the held-out `adversarial_H_mimicry_cufft` family remains
undetected; `bert_sst2_amp` also remains undetected. This is a development
ablation, not a new sealed result.

## CUSUM

A causal one-sided CUSUM was implemented and passed four hand-calculated unit
checks. Drift and threshold were selected with development repetitions 1-2 at
the fixed-rule false-alert budget; repetition 3 was kept out of calibration.

On repetition 3, electrical CUSUM detected the same 2/5 positive runs as
3-of-5 and reduced median alert time from 90 to 52.5 seconds, but introduced
one false alert instead of zero. Fusion CUSUM also detected the same 2/5,
changed median alert time from 90 to 75 seconds, and introduced one false
alert. NVML CUSUM detected 0/5 and removed the fixed rule's one false alert.

On the prior fused-update corpus, CUSUM detected 0/3 fused-update runs for all
modalities. It also failed to detect the three AdamW positive controls that
fixed-rule NVML detected. Therefore CUSUM does not solve the fused-update
failure and is not selected for the current detector. These fused results are
post-hoc secondary evidence, not a new untouched sequential test.
