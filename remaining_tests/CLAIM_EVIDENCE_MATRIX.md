# Current-paper claim-to-evidence release matrix

No row may be promoted to the abstract until its release gate passes.

| Candidate claim | Present evidence | Missing evidence | Release gate |
|---|---|---|---|
| The NVML pipeline is leakage-audited | Run-grouped out-of-fold predictions and split-disjointness tests | None for the narrow diagnostic | Keep explicitly labeled diagnostic |
| NVML interpolates across this RTX 3090 fleet | Leave-device-out: 22/23 training and 0/95 false-positive runs | More hardware diversity for a broader claim | Limit wording to the homogeneous fleet |
| NVML transfers to unseen workloads | Leave-family-out: 0/23 training and 10/95 false-positive runs | New training-family diversity and frozen transfer test | Do not claim with current evidence |
| Deployment false-alert target is met | Zero events in 11.45 negative GPU-hours for run/device grouping | At least 71.9 zero-event hours for a 95% upper bound below 1/24 GPU-hour | Report exposure and bound together |
| Electrical sensing improves evasion detection | No synchronized headline electrical result yet | Matched paired traces and sensor-only/NVML-plus-sensor confidence intervals | Paired interval excludes zero and retention rule passes |
| SensorGuard detects held-out evasions | Planned workloads only | R08--R13 attack artifacts and untouched R16 evaluation | Useful-work-preserving attacks remain held out and fixed-threshold test passes |
| Final SensorGuard false-alert rate and time to alert are known | Baseline diagnostic only | R06 negative exposure R16 final test and R18 first-trigger analysis | Frozen 3-of-5 rule evaluated once with right-censored latency |
| SensorGuard monitor overhead is below 1% | No final monitor measurement | R20 randomized overhead measurements | Report CPU storage bytes latency and useful-work penalty |
| Minimum retained electrical specification is sufficient | Pilot suggests GPU clamp carries the available signal | R14 retention decision and R19 causal rate/bit sweep with hardware confirmation | Lowest-cost point is within one SE and confirmed in hardware |
| Final system meets the paper claim | Architecture and diagnostic baselines exist | R03--R22 in `CHECKLIST.csv` | Every numerical sentence maps to immutable committed evidence |
