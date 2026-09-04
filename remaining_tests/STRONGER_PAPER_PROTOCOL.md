# Stronger-paper physical red-team protocol

Status: **code-ready; not yet collected**. Nothing in this file is a measured
result. Existing result files remain unchanged.

## Question

Does verifier-owned 10 kS/s GPU-rail current improve fixed-threshold,
held-out-family training detection beyond 1 Hz NVML on an RTX 3090, and can a
brief optimizer-event rule recover attacks that a window classifier misses?

## Experiment cells

| Phase | Family | Label | Purpose |
|---|---|---:|---|
| Development | honest full training | 1 | ordinary positive control |
| Development | chunked optimizer | 1 | spreads parameter updates across short chunks |
| Development | throttled training | 1 | reduces utilization and duty cycle |
| Development | inference dilution | 1 | hides each update among 20 forward passes |
| Development | matched inference | 0 | same model and batch without weight updates |
| Sealed | diluted LoRA | 1 | hardest unseen low-rank update family |
| Sealed | matched inference | 0 | sealed false-alert control |

The default plan uses 10 independent repetitions per cell: 50 development
runs followed, only after freezing, by 20 sealed runs. Runs are
repetition-major so slow thermal drift is distributed across families. Every
positive run records weight change and optimization progress; a run without a
measurable parameter change fails closed.

## Frozen comparison

Score **NVML alone**, **GPU current alone**, and **NVML + GPU current** on the
same causal 30-second windows, leave-one-family-out development splits, fixed
0.85 threshold, and fixed 3-of-5 run rule. Report run-level TP/FN/FP/TN,
per-family detection, worst-family detection, false alerts per GPU-hour, and
time to alert. Calibrate an optimizer-event rescue threshold only on
development negatives, freeze it, and report its incremental detections and
false alerts separately. Re-run the already implemented CUSUM comparison on
the new frozen probabilities as a secondary analysis; retain the 3-of-5 rule
unless CUSUM improves held-out detection without exceeding its false-alert
budget.

Current features must use anti-aliased resampling and frequency bands defined
in physical hertz. `scripts/analysis/physical_current_features.py` implements
this contract. This replaces stride-only diagnostic decimation for the new
campaign; it does not retroactively change existing reported results.

## Commands

Build plans (substitute the verified RTX 3090 UUID and scope serial):

```bash
python scripts/experiments/build_stronger_paper_plan.py \
  --output-dir collected/stronger_paper/plans \
  --gpu-uuid GPU-REPLACE-ME \
  --scope-serial 12789/2929
```

Run the development plan concurrently on node and verifier using
`scripts/experiments/run_synchronized_physical.py`. Freeze the trained model,
feature list, threshold, run rule, rescue threshold, and development run IDs
in a JSON manifest with `"status": "frozen"`. Then regenerate the sealed plan
with fresh timestamps and run it with `--frozen-manifest PATH` on both hosts.
The runner refuses a sealed plan without that manifest.

## Additional evidence needed for a strong paper

1. Repeat the sealed campaign on multiple RTX 3090s and collection days; the
   default single-GPU plan establishes only within-testbed transfer.
2. Collect at least 60 GPU-hours of synchronized current plus NVML negatives;
   the existing 60.581-hour negative corpus supports NVML only.
3. Report exact confidence intervals. Ten repetitions per attack family are a
   practical minimum; twenty are preferable for narrow family-level intervals.
4. Keep the previously sealed fused-update result as a semantic-boundary test:
   electrical activity may identify training-like execution without proving a
   model weight was updated.
