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

## Sealed fused-update test

After freezing the model at commit `bf78b81`, 15 additional 90-second runs
were collected: three fused-update runs, three AdamW positive controls, and
nine matched non-training controls (forward, three-forward, and dummy-write).
All 15 valid paired runs completed successfully, all learning runs made
measurable optimization progress, and all scope traces had zero overflow and
zero clipping.

- Fused update: NVML 0/3, electrical 0/3, and NVML + electrical 0/3 alerts.
- AdamW positive control: NVML 3/3 alerts at 90 seconds; electrical and fusion
  0/3.
- Matched negatives: 0/9 false-alert runs for all three methods.

The first sealed queue used an insufficient 120-second capture cadence. Three
paired cells completed, then the verifier refused cell 4 before capture with
return code 98. The two owned screens were stopped, no process remained, the
completed fused run was not repeated, and the untouched remaining cells were
collected at 135-second cadence. Only the 15 complete paired cells are scored.

Combining development and sealed matched negatives, electrical and fusion had
0 alerts over 0.6625 GPU-hours (measured rate 0.000 alerts/GPU-hour); NVML had
2 alerts over the same exposure (3.019 alerts/GPU-hour). The sealed
fused-update time to alert is not estimable because no detector alerted.

## Paper consequence

These new independent tests do **not** support the intended positive
worst-family claim: both NVML and SensorGuard have 0% worst-family detection,
so the measured worst-family gain is 0 percentage points. The older 36-run
development result remains valid as a proof of concept, but it must not be
presented as though it passed this sealed test. A positive paper claim now
requires a redesigned detector trained without using fused-update results and
a different, newly sealed attack family; fused update cannot be reused as a
fresh holdout.

## Minimum retained electrical specification

Offline decimation and quantization selected 10 kS/s at 8-bit as the lowest
tested configuration that exactly preserved all 30 full-rate run decisions.
At 1 kS/s, one of the six full-rate detections was lost. This is an offline
retention result, not a hardware-in-loop confirmation at the selected setting.

## Monitor overhead

The existing five-repetition physical-logger experiment measured a mean
useful-work penalty of -0.045%, with a 95% confidence interval from -0.353% to
0.262%; therefore no monitor overhead was resolved at this sample size.
