# Remote run status — 2026-08-29

All times below are UTC. This note records acquisition status, not final model
results. Do not report the projected exposure as a measured false-alert rate.

## RTX 3090 application roofline (R07A complete at 200 W)

The corrected node1 GPU1 run completed all five FP16 application cases with
three finite repetitions per case: ResNet-50 training and inference, GPT-2
training, prefill, and autoregressive decode. The measured ceilings at the
same 200 W setting were 64.73 TFLOP/s and 804.12 GB/s, giving a ridge point of
80.50 FLOP/DRAM byte.

| case | role | median intensity (FLOP/byte) | median wall throughput (TFLOP/s) |
|---|---|---:|---:|
| ResNet-50 batch 64 | training | 14.61 | 5.67 |
| ResNet-50 batch 64 | inference | 44.56 | 17.98 |
| GPT-2 batch 2, sequence 256 | training | 25.10 | 11.82 |
| GPT-2 batch 2, sequence 256 | prefill inference | 113.79 | 18.98 |
| GPT-2 batch 2, 32-token decode | decode inference | 14.22 | 0.77 |

The standalone roofline characterization is in
`results/figures/rtx3090-application-roofline.pdf` with SVG and PNG copies. It
must not be presented as a detector: training and inference ranges overlap in
both normalized intensity and normalized throughput. Its raw 15-row input is under
`results/roofline/applications/validated-bb6b232/`; the six preregistered
training/inference pair rows are in
`results/paper/matched_application_roofline_node1_200w.csv`. These are Nsight
characterization results, not detector or sensor-accuracy results. H200 is not
part of this collection.

The replacement paper-motivation figure is
`results/figures/sensor-motivation-evidence.pdf`. Panel A now compares NVML and
the GPU-current detector on the exact same 36 paired RTX 3090 runs (26 training
and 10 non-training). It leaves each workload label out of fitting in turn and
applies the amended 3-of-5 rule at probability 0.85. NVML gives TP=18, FN=8,
FP=0, TN=10; SensorGuard gives TP=24, FN=2, FP=0, TN=10. SensorGuard recovers
six runs that NVML misses: two `adversarial_H_mimicry_cufft`, two
`pytorch_mlp_cifar10`, and two `adversarial_J_pid_75` runs. Both methods miss the two short
`bert_sst2_amp` runs because each yields only two 30-second windows and cannot
satisfy a three-window alert. This is a matched development diagnostic, not
the sealed final attack-family test.

The broader compatible development inventory contains 61 paired runs (51
training and 10 non-training), including adaptive physical red-team attacks.
Under strict leave-one-workload-label-out scoring at 0.85, NVML detects 44/51
training runs with 0/10 false alerts; the current-only SensorGuard model detects
42/51 with 2/10 false alerts. This deliberately harder, non-sealed stress test
shows that the current-only model still needs adversarial hardening and more
independent controls. It is reported separately in
`results/tables/matched-extended-family-heldout-confusion.csv` and must not be
substituted for the exact matched 36-run comparison. The SPAR project notes
corroborate the prior physical-evasion slowdown result but contain no additional
synchronized NVML/current traces, so they do not increase either cohort.

The separate NVML-only inventory is larger: 118 runs (23 training and 95
non-training) have usable software telemetry but no matched GPU-current trace.
At 0.85, the run-grouped and GPU-grouped audits each detect 22/23 training runs
with no observed alert among 95 non-training runs. A strict held-out-family
audit detects 0/23 training runs and alerts on 2/95 non-training runs, showing
that this older inventory does not support a generalization claim to unseen
workload families. The day audit is not fully estimable because one training
fold lacks the required classes. These results are in
`results/evaluation/baseline-*-grouped.json`; they broaden NVML diagnostics but
cannot fairly enlarge Panel A because the corresponding current traces do not
exist.

Panel B is a matched overhead comparison. WAVE and the SensorGuard NVML+DCGM
base logger were measured on the exact same
six configurations (two GPT-2, two LLaMA, and two Qwen), the same power-capped
RTX 3090 (GPU-392b25f7...), three repetitions per condition, and the same
whole-process wall-time definition. WAVE averages 28.497x runtime with a
23.54--31.59x configuration range; SensorGuard's base logger averages 1.050x
with a 1.027--1.083x range. All 18 monitored runs passed the NVML and DCGM
trace-health gates. Raw aggregate measurements and the run manifest are in
`results/wave/matched_sensor_overhead_3090.csv` and `.json`.

Suggested caption: **A:** Leave-one-workload-family-out training detections on
the same 36 paired runs. Under the amended 3-of-5 rule at probability 0.85,
NVML detects 18/26 training runs while SensorGuard's GPU-current detector
detects 24/26. **B:** On six
identical GPT-2, LLaMA, and Qwen configurations on the same RTX 3090, WAVE
costs 28.50x runtime while the SensorGuard NVML+DCGM base logger costs 1.05x;
whiskers show the range across configurations. The overhead bar covers the
NVML+DCGM base logger only.

For the final claim, collect more synchronized application-level pairs chosen
from NVML-hard development cases before inspecting their sensor predictions:
matched GPT-2 and BERT training/inference, the fused-update held-out attack,
and useful-work-preserving timing/dilution or online-update training paired
with inference, FFT, mining, and rendering controls. Freeze which families are
validation and which are final test before fitting. These new electrical runs
remain blocked only on safe PicoScope probe wiring and the channel-to-GPU map.

## Negative exposure (R06)

The following unprofiled NVML+DCGM campaigns were observed or started:

| host | GPUs | start | wall budget per GPU | status |
|---|---:|---:|---:|---|
| node2 | 0,1,2 | 19:48 | 8 h | Robi's campaign running; first GEMM manifests completed with both channels healthy |
| verifier | 0 | 20:31 | 8 h | Robi's campaign running |
| node2 | 3 | 20:42 | 30 h | Started remotely for R06; PID 11637; CUDA/NVML UUID mapping verified |

The preregistered held-out node2 GPU4 (`GPU-127e16de...`) remains unused by
the added campaign. If all listed runs complete, pass channel-health gates,
and later yield zero false alerts under the frozen detector, the prior 11.45 h
plus 62 h of new exposure gives 73.45 GPU-hours. This exceeds the 71.9 h
minimum for a one-sided 95% zero-event Poisson upper bound below 1/24 alerts
per GPU-hour. Calculate the final exposure from manifest workload timestamps,
not directory counts.

### Interim validated upload

The first six completed node2 runs (GEMM and FFT on GPUs 0--2) passed manifest
schema, channel-health, and artifact SHA-256 checks. Their measured workload
durations contribute 3.004316 GPU-hours. With the prior 11.45 hours, the
current auditable total is 14.454316 GPU-hours. The traces and manifests are
under `results/negative_exposure/node2/`; `interim_summary.json` intentionally
leaves the false-alert bound null because the frozen detector has not yet been
applied.

### Quarantined node1 campaign

Do not count the 20260829 node1 campaign. GPU0 reports `GPU requires reset`,
and the attempted five-GPU run produced CUDA/NVML mapping and marker failures.
At inspection time its completed manifests included 343
`failed:load marker failed`, two `flagged_channel_health`, and only two marked
`completed`; the entire campaign is quarantined because simultaneous markers
can make an incorrectly mapped channel appear healthy. Repair/reset the host,
then rerun a blinded one-GPU-at-a-time mapping check before collecting data.

## Electrical preflight (R03/R04)

`pico_logger.py --list` successfully opened and closed six verifier scopes:

`13239/2963`, `13239/1303`, `13239/2968`, `13239/2980`, `13239/0093`, and
`12789/2929`.

This proves USB/API availability only. Electrical calibration, GPU/channel
mapping, and synchronized rail traces remain blocked on approved probe wiring
and safety sign-off. No dry-run signal-generator or rail capture was attempted.

## Monitor overhead (R20 partial)

The committed verifier result covers NVML and DCGM only. Both reported 0.0%
useful-work penalty in the 90 s GEMM check, with mean logger CPU use of 0.56%
and 0.04%, respectively. Physical electrical-logger overhead, randomized A/B
repetitions, privacy probes, and final representative-workload measurements
remain outstanding.

## Fused-update held-out family (R08 prepared, not opened)

Robi's private repository was audited and the prior TinyLlama symmetric-GEMM
implementation/results were found. A SensorGuard-native acquisition adapter
and guarded matrix runner were added. The fused-update and AdamW learning paths,
dummy-write control, and forward controls passed CPU execution locally; the
fused-update learning path also passed a low-priority CPU-only test on node2.

No fused-update GPU telemetry was collected or queued. The preregistration
names `fused_update_kernel` as the held-out evasion family, so the runner blocks
execution until R15 is frozen and an explicit final-test authorization is
given. This is a protocol gate, not a missing-code or access blocker.
