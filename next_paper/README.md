# Next-paper experiment roadmap

This directory holds experiments intentionally excluded from the current
ASPLOS submission. They must not be described as implemented or measured in
the current paper.

## Primary extension: sequential evidence accumulation

Implement and compare one-sided CUSUM, EWMA, and HMM decision rules against the
current fixed 3-of-5 rule. Use the same frozen window probabilities and grouped
development/validation/test partitions so the comparison isolates the
run-level decision method.

Required work:

1. Define causal alert-time semantics and preserve undetected runs as
   right-censored observations.
2. Calibrate CUSUM drift and threshold on validation only at the same
   false-alert budgets as 3-of-5.
3. Freeze parameters and code before opening a new untouched test set.
4. Compare worst-family TPR, false alerts/GPU-hour, median and 95th-percentile
   time to alert, useful-work penalty, and calibration.
5. Test whether accumulation helps LoRA, diluted updates, and fused-update
   kernels without increasing long-idle or bursty-control false alerts.

## Additional deferred extensions

- Complete WAVE trace-size, solver-time, and recovered-parameter tables.
- Confirm the chosen monitor across broader GPU architectures and hosts.
- Run the prepared normalized RTX/H200 application bridge only after an H200
  host becomes available; historical H200 traces are not a matched substitute.
- Test contact temperature, ultrasound, RF, thermal-camera, and mirrored
  Ethernet only under a new validation/test protocol.
- Study exhaustive three-sensor combinations and broader cloud/platform
  transfer.

The machine-readable queue is [`CHECKLIST.csv`](CHECKLIST.csv).
