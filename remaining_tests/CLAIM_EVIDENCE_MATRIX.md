# Claim-to-evidence release matrix

No row may be promoted to the abstract until its release gate passes.

| Candidate claim | Present evidence | Missing evidence | Release gate |
|---|---|---|---|
| The NVML pipeline is leakage-audited | Run-grouped out-of-fold predictions; split-disjointness tests | None for the narrow pipeline claim | Keep explicitly labeled diagnostic |
| NVML interpolates across this RTX 3090 fleet | Leave-GPU-out: 22/23 training and 0/95 false-positive runs | More hardware diversity for a broader claim | Limit wording to the homogeneous fleet |
| NVML transfers to unseen workloads | Leave-family-out: 0/23 training and 10/95 false-positive runs | New training-family diversity and a frozen transfer test | Do not claim with current evidence |
| Deployment false-alert target is met | Zero events in 11.45 negative GPU-hours for run/GPU grouping | At least 71.9 zero-event hours for a 95% upper bound below 1/24 GPU-hour | Report exposure and bound together |
| Electrical sensing improves evasion detection | No synchronized headline electrical result yet | Matched paired traces; sensor-only and NVML+sensor CIs | Paired CI excludes zero and frozen retention rule passes |
| SensorGuard detects fused-update training | Planned architecture and workload registry | Held-out useful-work-preserving fused-update corpus | Untouched-family test at fixed threshold |
| SensorGuard detects telemetry tampering | Cross-modal residual design only | Replay/freeze/shift/swap and channel-failure injections | Report classification and health-alarm outcomes separately |
| Minimum ADC specification is sufficient | Data contract exists | Causal rate/bit sweep and hardware-in-loop confirmation | Selected point within one SE of full fidelity and confirmed in hardware |
| WAVE is too expensive for continuous use on the 3090 | Six configurations; measured 161% one-counter and 2750% full-set mean overhead | Trace-size and solver-time completion strengthens the comparison | Retain adapted-reproduction label |
| Final system meets ASPLOS claim | Architecture and diagnostic baselines exist | R03--R22 in `CHECKLIST.csv` | Every numerical sentence maps to immutable committed evidence |
