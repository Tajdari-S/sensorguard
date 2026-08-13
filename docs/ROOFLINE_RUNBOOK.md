# RTX 3090 roofline runbook

Roofline characterization is a separate profiled experiment. Do not use Nsight-instrumented runs as headline physical-sensor traces because profiler overhead changes execution.

## When to run

| Date | Test | Purpose |
|---|---|---|
| Aug 16 | Unit tests and one-GPU microbenchmark smoke | Verify formulas, CUDA/PyTorch, output schema and plotting |
| Aug 20 | Lock clocks/power, run microbenchmark peaks, validate NCU metrics | Establish empirical compute and DRAM ceilings |
| Aug 21 | Profile application matrix: ResNet, ViT, GPT-2 prefill/decode, training/forward-only, FFT and copies | Produce the paper roofline and overlap-pair candidates |
| Aug 22 | Freeze roofline data and matched-pair IDs before the sensor gate | Prevent post-hoc pair selection |
| Sep 5 | Re-run parser/plot/tests from a clean checkout | Reproducibility audit only; do not recollect unless a logged defect exists |

## Commands

First validate the tooling without a GPU:

```bash
make roofline-test
```

Run the CUDA microbenchmark smoke on GPU 0:

```bash
make roofline-smoke GPU=0
```

Run the full Nsight Compute pass:

```bash
bash scripts/roofline/run_ncu_roofline.sh results/roofline/ncu 0
python3 scripts/roofline/parse_ncu.py results/roofline/ncu \
  --output results/roofline/summary.json
```

Generate the SVG after replacing the example ceilings with measured locked-clock values:

```bash
python3 scripts/roofline/plot_roofline.py results/roofline/summary.json \
  --peak-tflops 35.58 --peak-gbps 936.2 \
  --output results/roofline/rtx3090-roofline.svg
```

The numeric values above are nominal RTX 3090 reference values, not accepted experimental measurements. Measure and record the actual sustained ceilings for each GPU/clock/power configuration.

## Application matrix

`configs/roofline_matrix.json` contains the required application sweeps. Application adapters must expose a single profiled command and a corresponding unprofiled sensor command with identical model, batch, precision, sequence length, useful-work target, clocks and power limit.

For every point retain: GPU UUID, driver/CUDA/PyTorch/Nsight versions, command, git commit, dtype, shapes/batch/sequence length, clocks, power limit, warmups, iterations, NCU CSV, parsed JSON, FLOP definition, DRAM-byte definition and profiler slowdown.

## Pass criteria

- Unit tests pass and every command returns zero.
- All seven microbenchmark cases produce nonempty JSON and NCU CSV.
- GEMM approaches the compute ceiling and copy approaches the bandwidth ceiling closely enough to diagnose setup errors; deviations must be explained, not silently corrected.
- Arithmetic intensity uses measured NCU DRAM bytes when available and clearly labels analytical minimum-byte estimates otherwise.
- Repeated points have coefficient of variation below the preregistered limit.
- The SVG can be regenerated from committed small derived JSON without raw traces.
