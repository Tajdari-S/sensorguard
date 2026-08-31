# Current-paper synchronized physical tests (2026-08-31)

## Setup

RTX 3090 GPU1 on node1; PicoScope serial `12789/2929`, channel A; 10 kS/s electrical acquisition plus 1 Hz NVML. Each of ten workload families has three independent 105-second runs (30 runs total).

## Integrity

All 30 paired node/verifier cells completed successfully: 30/30 returned zero on both sides. All scope traces contain 1,299,444 samples, zero overflow, and zero clipping.

## Frozen development evaluation

The protocol is leave-one-workload-family-out, 30-second causal windows with 15-second stride, a 400-tree random forest, probability threshold 0.85, and fixed 3-of-5 alerting.

- NVML: 0/15 training runs detected; 2/15 false-alert runs; 4.571 false alerts/GPU-hour over 0.4375 GPU-hours; worst-family detection 0.0%.
- Electrical: 6/15 training runs detected; 0/15 false-alert runs; 0.000 false alerts/GPU-hour over 0.4375 GPU-hours; worst-family detection 0.0%.
- NVML + electrical: 6/15 training runs detected; 0/15 false-alert runs; 0.000 false alerts/GPU-hour over 0.4375 GPU-hours; worst-family detection 0.0%.

Electrical high-frequency features detect all duty-shaping and migration runs, but do not transfer to ordinary, interleaved, or memory-minimal training. Therefore the new corpus does not support a positive worst-family gain claim at the frozen threshold. The sealed fused-update family was not used in these results or in feature/model selection.

## Remaining sealed test

Run fused-update once using the frozen model hashes in `freeze_manifest.json`, then append its predictions without retraining or changing the threshold.
