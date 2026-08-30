# Robi repository audit and execution ownership

Audited 2026-08-29. The machine-readable inventory is
[`../results/tables/robi_available_runs.csv`](../results/tables/robi_available_runs.csv).

## What already exists

Robi's private `robirahman/GPU-monitoring` repository contains more than the
current SensorGuard registry previously indicated:

- `main` at `101027906b63067ded909c272e43df19e56c75c1` contains the TinyLlama
  symmetric-GEMM fused-update ladder, its run queue, prior detector results,
  throughput/energy summaries, PPO-shaped controls, and A/B/D adversarial
  workloads.
- `shaping-evasion-study` at
  `84160bb55e51ddf8c7e3fedbcdb4100690dc4bae` contains timing shaping,
  memory-minimization, and co-execution-cover implementations and H200 results.
- `physical-sensor-detection` at
  `baf61087ac478811af548fc4827fccb12d0546c2` contains the API-based physical
  sensor logger, sensor-aware attacks, physical RF analysis, and multi-GPU
  clamp aggregation code.
- `robirahman/sigil` was checked and does not contain relevant GPU monitoring
  or experiment code.

The source repository has no declared GitHub license. Keep the source paths,
commits, and authorship in provenance. Do not copy the full private source into
a public repository without the project owners' agreement.

## What SensorGuard now runs itself

`scripts/workloads/fused_update_workload.py` is a SensorGuard-native matched
fused-update acquisition adapter. It reports loss reduction, weight change,
throughput, and whether useful optimization progress occurred. It has five
matched modes: fused update, three-forward negative, ordinary AdamW positive,
dummy dW write, and one-forward negative.

Generate or execute the matrix with:

```bash
# Safe preview; does not touch the GPU.
python scripts/experiments/run_fused_update_matrix.py --gpu-index 3 --dry-run

# Untouched final acquisition only after R15 is frozen and authorized.
python scripts/experiments/run_fused_update_matrix.py \
  --gpu-index 4 --repetitions 3 --duration-s 600 --sensors nvml,dcgm \
  --final-test-authorized --allow-held-out-gpu4
```

The runner refuses a GPU that already has a compute process. It refuses any
execution until the operator confirms the model/split/rule freeze, because
`fused_update_kernel` is the preregistered held-out family. It also refuses GPU
4, the held-out GPU, unless the second explicit protocol authorization is
supplied. R08 remains incomplete until that untouched run is collected; a
separate synchronized electrical run is also required if electrical sensing is
retained.

## What not to ask Robi to do

Do not ask Robi to write the fused-update, shaping, interleaving, or physical
attack commands: implementations already exist, and Sabiha/Codex owns adapting
and running them in SensorGuard. Do not ask him to stop or move his running
jobs.

## The one essential hardware question

Ask only for information that cannot be derived safely in software:

> Hi Robi, I found the existing attack and sensor code, so I will handle the
> adapters and runs. Could you please confirm the current physical-sensor
> wiring/channel map for each GPU (scope serial number, channel, rail/probe,
> polarity and conversion factor), and whether the wiring is approved for
> unattended acquisition? I will not alter or stop any running jobs.

If the current setup still uses the old SAIGE sensor API, also request the
credential location; never request or send the password in email.

## Why the existing results are not automatically final-paper results

They are valuable prior evidence and eliminate code duplication, but most were
collected on a different testbed, classifier, sensor set, or adaptive analysis
round. The paper's primary numbers require the same frozen grouped split,
random forest, 0.85 threshold, 3-of-5 rule, synchronized channels, held-out
families, and reporting pipeline. Therefore they guide selection and provide
provenance; they do not replace the untouched final test.
