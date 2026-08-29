# Remote run status — 2026-08-29

All times below are UTC. This note records acquisition status, not final model
results. Do not report the projected exposure as a measured false-alert rate.

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
