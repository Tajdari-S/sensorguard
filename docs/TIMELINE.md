# 27-day ASPLOS execution timeline

Dates are in 2026. The official September-cycle full-paper deadline is **September 9 AoE**; there is no separate abstract deadline.

## Priority tiers

- **P0: submission-critical.** Frozen NVML baseline; synchronized electrical sensing; matched-overlap evaluation; held-out evasion; calibrated run-level rule; overhead; one hardware co-design sweep; complete paper.
- **P1: strengthens the paper.** WAVE reproduction on one RTX 3090; contact temperature; tamper/replay; privacy probe; all-six-GPU confirmation.
- **P2: include only if early evidence is positive.** Ultrasound, RF, and mirrored-Ethernet expansion; exhaustive triples; broad cross-platform cloud study.

## Schedule

| Date | Gate and work | Primary lane | Required output |
|---|---|---|---|
| Aug 13 | Kickoff; confirm four owners; freeze paper claim and P0/P1/P2 scope | Paper/integration | Owner map; scope decision log |
| Aug 14 | Inventory six RTX 3090 GPUs, host, topology, clocks, software, and every sensor; safety review | Sensors | Complete system table; topology diagram; inventory archive |
| Aug 15 | Freeze run manifest, workload matrix, false-alert budget, split groups, exclusions, and retention rule | Analysis | Hashed preregistration; immutable split IDs |
| Aug 16-17 | Validate synchronized loggers with idle/GEMM/memcpy/bursty tests; run roofline smoke; measure noise, drift, clipping, jitter | Sensors | Calibration report; channel map; roofline smoke JSON/SVG; pass/fail per modality |
| Aug 16-18 | Reproduce strict NeurIPS NVML pipeline on one GPU, then all six; lock feature code | Analysis | Strict and 3090-tuned baseline table |
| Aug 17-19 | Pin WAVE commit; map Ampere counters; run GPT-2/LLaMA/Qwen representative cases on one GPU | Paper/integration | Adaptation log; fidelity and overhead table |
| Aug 18-21 | P0 synchronized collection: training, matched inference, GEMM/FFT/memcpy/rendering/mining proxy, and hard evasions | Workloads | Audited development/validation corpus |
| Aug 20-21 | Run separate Nsight roofline characterization passes and generate the RTX-3090 roofline; freeze matched-overlap pairs | Analysis | NCU CSV, parsed roofline JSON/SVG, command log, frozen pair IDs |
| **Aug 22** | **Gate 1: sensor retain/drop decision** using validation only | All | Decision record; P2 modalities killed or retained |
| Aug 23-26 | Full repetitions for NVML, electrical, and retained sensors across GPU/day groups; collect useful-work costs | Sensors + workloads | P0 evaluation corpus; failure ledger |
| Aug 24-27 | Implement five evasion families and telemetry replay/freeze/shift/channel-swap tests | Workloads | Zero-shot held-out attack corpus |
| Aug 27-29 | Train random-forest single-sensor and fusion models; freeze 3-of-5 rule; bootstrap CIs; leave-one-family-out | Analysis | Frozen model/rule/threshold hash; primary result table |
| **Aug 30** | **Gate 2: claims freeze**; open held-out test split once | All | Final headline numbers; claims matrix |
| Aug 31-Sep 2 | Sampling-rate/bit-depth/duty-cycle/ROI sweeps; hardware-in-loop confirmation; monitor overhead | Sensors + analysis | Minimum sensor specification; overhead table |
| Aug 31-Sep 3 | Paper writing: methods, results, threat model, limitations, appendix; generate figures | Paper/integration | Complete anonymous draft |
| Sep 4 | Internal full-paper review; check ASPLOS scope, novelty, and overlap with prior NeurIPS work | All | Consolidated review issues |
| Sep 5 | Revision; statistical and provenance audit; reproduce figures from clean checkout | All | Reproducible near-final PDF |
| Sep 6 | Red-team review: claims vs. evidence, leakage, split integrity, false-alert interpretation | Workloads + analysis | Signed-off claim/evidence matrix |
| Sep 7 | Anonymization, formatting, references, artifact links, supplementary appendix | Paper/integration | Submission candidate |
| **Sep 8** | **Freeze PDF and metadata; upload draft to HotCRP and verify rendering** | Paper/integration | Upload-ready final PDF |
| **Sep 9 AoE** | Emergency buffer only; final submission verification | All | Submitted paper |

## Daily cadence

- 10 minutes: blockers and machine/sensor availability.
- 15 minutes: audit newly completed runs and exclusions.
- End of day: commit small derived artifacts, update issues, and record the next day's exact run queue.
- Every gate: one collaborator not responsible for the analysis verifies split integrity and the claim/evidence mapping.

## Scope cuts if behind

1. Cut full ultrasound, RF, and mirrored-Ethernet evaluation unless a modality gives a validation-only gain beyond NVML plus electrical sensing.
2. Limit WAVE to one GPU and representative model sizes; keep it as a high-information/overhead reference.
3. Prefer contact temperature over full thermal imaging if it lies within one standard error of the camera.
4. Cut exhaustive three-sensor combinations before cutting held-out attack families or confidence intervals.
5. Never cut the fixed-threshold run-level evaluation, useful-work cost, or leakage-safe split.
