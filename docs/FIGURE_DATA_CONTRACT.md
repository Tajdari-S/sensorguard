# Paper figure and table data contract

All figures use out-of-fold or untouched-test predictions. Plotting code must
refuse missing columns and must never substitute simulated values.

## Figures generated from the current repository

Run `make data-audit figures` to generate:

- `results/figures/corpus-coverage.{pdf,svg}`
- `results/figures/baseline-diagnostic.{pdf,svg}`
- `results/figures/wave-overhead.{pdf,svg}`
- `results/figures/calibration-alignment.{pdf,svg}`
- `results/tables/corpus-exposure.{csv,tex}`
- `paper/generated/current_evidence.tex`

## Headline figures awaiting new measurements

### Aligned physical traces

Input: `results/paper/aligned_trace_panels.csv` with columns
`run_id,family,true_class,time_s,nvml_power_w,nvml_util_pct,rail_power_w,
power_residual_w,fusion_probability,first_weight_update_s,alert_time_s`.
Required families: matched inference, ordinary training, fused-update training.

### Sensor-augmented roofline

Input: `results/paper/matched_roofline.csv` with columns
`pair_id,run_id,true_class,family,operational_intensity,achieved_flops,
nvml_probability,electrical_probability,fusion_probability,fold`.
Every plotted probability must be out-of-fold. Pair IDs are frozen before the
sensor columns are inspected.

### Deployment operating curve

Input: `results/paper/operating_curve.csv` with columns
`monitor,threshold,false_alerts_per_gpu_hour,worst_family_tpr,
worst_family_tpr_ci_low,worst_family_tpr_ci_high,negative_gpu_hours`.
The plot includes the preregistered 1/8, 1/24, and 1/72 GPU-hour budgets.

### Time to alert

Input: `results/paper/time_to_alert.csv` with columns
`run_id,family,monitor,duration_s,alert_time_s,detected`. Undetected runs are
right-censored at `duration_s`; do not drop them or display only successful
alerts.

### Sensor ablation

Input: `results/paper/sensor_ablation.csv` with columns
`configuration,comparison,delta_worst_family_tpr,ci_low,ci_high,
false_alerts_per_gpu_hour,monitor_overhead_fraction,bytes_per_gpu_hour,
retention_decision`.

### Sampling rate and quantization

Input: `results/paper/hardware_sweep.csv` with columns
`sample_rate_hz,effective_bits,worst_family_tpr,false_alerts_per_gpu_hour,
bytes_per_gpu_hour,logger_cpu_fraction,hardware_confirmed`. Offline degradation
and hardware-in-loop confirmation are distinguished by the final column.

## Required main-paper tables

1. Testbed and sensor specifications, separated by NVML fleet, electrically
   instrumented subset, and WAVE host.
2. Dataset composition and negative exposure by family.
3. Primary fixed-operating-point comparison with confidence intervals.
4. Leave-one-GPU/day/application/evasion-family generalization.
5. Add-one and leave-one-out sensor selection.
6. Held-out attacks with useful-work and energy penalties.
7. WAVE fidelity, trace size, solver time, and overhead.

Every result row records the generating commit, input artifact hashes, split
identifier, and whether it is diagnostic, validation-selected, or untouched
test evidence.
