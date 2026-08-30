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
`results/figures/sensor-motivation-evidence.pdf`. Panel A shows that 101/286
(35.3%) non-training configurations fall inside the training
arithmetic-intensity range and notes that the NVML held-out-family detector
finds 0/23 training runs. Panel B compares the measured WAVE runtime multiplier
(28.5x mean; 23.5--31.6x range) with the 1.0x SensorGuard base-logger check.
The x-axis names the three evaluated application families for each method:
WAVE uses GPT-2, LLaMA, and Qwen; the current SensorGuard corpus uses ResNet-50,
GPT-2, and BERT. MLP remains a separate physical pilot, fused update is pending,
and the 1.0x SensorGuard bar must not be described as end-to-end overhead until
the physical logger is measured.

Suggested caption: **A:** In a prior roofline sweep, 101 of 286 non-training
configurations occupy the training arithmetic-intensity range; roofline
position alone is therefore ambiguous, and the NVML held-out-family test
detects 0/23 training runs. **B:** WAVE's three-family architectural-verification
reproduction costs 28.5x runtime on average, whereas the current SensorGuard
base logger measures 1.0x on three training targets (ResNet-50, GPT-2, and
BERT). SensorGuard's end-to-end physical-logger overhead remains pending.

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
