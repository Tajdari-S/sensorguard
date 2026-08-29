# Daily execution runbook

The dates below complement `TIMELINE.md`. Queue generation is runnable now; GPU execution remains gated on adapter status in `configs/workloads.json`.

| Date | Queue/test | Command available now | Exit criterion |
|---|---|---|---|
| Aug 13 | Registry and owner freeze | `python3 scripts/check_workload_coverage.py` | Zero schema/coverage errors; owners assigned |
| Aug 14 | Six-GPU and sensor inventory | `bash scripts/capture_inventory.sh results/inventory` | GPU UUID/topology and every sensor channel recorded |
| Aug 15 | Preregistration and complete application registry | `python3 scripts/validate_preregistration.py configs/preregistration.yaml` | Hash stored; split groups and exclusions frozen |
| Aug 16 | Idle/GEMM/memcpy/burst calibration plus roofline smoke | `make roofline-test` then `make roofline-smoke` | Sensor checks pass; benchmark JSON and SVG generated |
| Aug 17 | Clean training smoke tests | `python3 scripts/run_matrix.py --group training --priority P0` | Each native adapter produces manifest + samples |
| Aug 18 | Clean inference smoke tests | `python3 scripts/run_matrix.py --group inference --priority P0` | Matched controls and prefill/decode labels verified |
| Aug 19 | NVML baseline queue | create queue after adapters pass | Nine 1 Hz signals, causal windows and run IDs match protocol |
| Aug 20 | HPC/render/mining controls | `python3 scripts/run_matrix.py --group non_ml` | All P0 controls run; P1 failures logged |
| Aug 21 | Full roofline characterization | `bash scripts/roofline/run_ncu_roofline.sh results/roofline/ncu 0` then parse/plot commands in `docs/ROOFLINE_RUNBOOK.md` | NCU CSV, summary JSON and SVG committed; profiling separated from sensor corpus |
| Aug 22 | Sensor gate | validation-only analysis | Retain/drop record signed; no test-set access |
| Aug 23–24 | Full clean repetitions | materialize P0 training/inference/non-ML queues | Repetition/GPU/day quotas met |
| Aug 25–26 | NeurIPS evasions R1–R3 | `python3 scripts/run_matrix.py --group evasion --priority P0` | A–L families audited with useful work |
| Aug 27 | NeurIPS evasions R4–R5 | same queue, filtered in run ledger | All 20 families and variants accounted for |
| Aug 28 | Custom/fused, signal-noise, tamper and adaptive cases | `python3 scripts/run_matrix.py --group evaluation --priority P0` | Five names resolved; fused controls and P0 robustness tests collected |
| Aug 29 | Random-forest calibration, fixed 3-of-5 rule, and LOO | analysis adapter (to implement) | Model, rule, threshold and code commit frozen |
| Aug 30 | Held-out test once | frozen analysis command only | Claims table and immutable output hash |
| Aug 31–Sep 2 | Sensor rate/bit/ROI sweeps | hardware sweep adapter (to implement) | Minimum spec within one SE, hardware confirmed |
| Sep 3–8 | Figures, audits, paper freeze | clean-checkout reproduction command (to implement) | Every figure regenerated and PDF frozen |

At each day’s start, generate a CSV queue under `results/queues/`, add actual owner/GPU/date, and review it before any run. At day’s end, keep the queue, manifests, checksums and failure ledger; do not commit raw traces.
