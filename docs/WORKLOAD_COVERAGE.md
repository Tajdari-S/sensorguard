# Workload and application coverage contract

This file answers two different questions explicitly:

1. **Is an application represented in the ASPLOS evaluation plan?** See `configs/workloads.json`.
2. **Can it run today?** Only entries marked `runnable` can be launched without new adapter work. At repository creation, the registry and queue tooling exist, but GPU workload adapters still need to be ported and tested.

The coverage contract consolidates the applications needed for the ASPLOS evaluation: public-paper workloads, SPAR project applications, hard evasion cases, deployment environments, and roofline corner cases. It contains no review text or private source material.

## Coverage summary

| Class | Named coverage |
|---|---|
| Training | ResNet/CIFAR-10, GPT-2/WikiText, BERT, simple MLP, DDP, FSDP 7B–70B, LoRA including the 7B white-box case, PPO, multi-node training |
| Inference | ResNet-18/50/101/152, ViT-S/B/L, GPT-2 prefill/decode, llama.cpp 7B, TinyLlama, Phi-3, Qwen2.5 3B–72B, Mistral, Mixtral, DeepSeek-V3/R1, Whisper, SDXL, multi-query inference, multi-user serving |
| Scientific/HPC | GEMM, FFT/cuFFT, N-body, GROMACS, SpMV/stencil, memory copy |
| Other non-ML | mining/Ethash-like proxy, Blender BMW/Classroom, FFmpeg NVENC, network stress, idle, ETL/data processing, database acceleration, predictive analytics, mixed ML/HPC, JAX/XLA, production-trace replay |
| NeurIPS evasions | All 20 named R1–R5 families, including every published variant count |
| Robustness/deployment | low-utilization, PID, dilution, white-box full/LoRA, custom CUDA variants, fused updates, causal latency, AMD/TPU, MIG/virtualization, signal perturbations, telemetry tamper, adaptive attacks, and meta/adversarial learning |

## Original study strata to reproduce

| Stratum | Published composition | SensorGuard rule |
|---|---:|---|
| Training | 106 workloads / 867 runs | Reproduce a balanced RTX-3090 subset; preserve family labels and run grouping |
| Inference | 40 workloads / 352 runs | Pair models, precision, batch and temporal mode with training where possible |
| Non-ML | 16 workloads / 185 runs in the paper table | Include HPC, mining, render/video, idle and miscellaneous controls |
| Total | 162 workloads / 1,404 runs | Do not compare raw windows split across the same run |

When reproducing a public baseline, preserve the public workload/run group labels exactly and keep new SensorGuard experiments in a separate corpus version.

## Project-critical acceptance tests

| Case | Why it matters | Required result |
|---|---|---|
| `B_low_util` | Published leave-one-family-out detection was difficult | Fixed-threshold run TPR, false alerts/GPU-hour and useful work |
| `J_pid` | Closed-loop telemetry shaping | Same metrics plus controller trace and target error |
| `L_diluted` | Inference between updates can make training sparse | Sweep 2/5/10/20 inference calls per update |
| `whitebox_full` | Sensor-aware evader | All eight variants; never tune on held-out variant |
| `whitebox_lora` | Low-footprint updates are a hard evasion class | Published variants plus separately labeled 1B/3B/7B runs |
| Fused update kernel | A custom fused kernel challenges 1 Hz telemetry | Training, forward-only and dummy-write controls at 1 Hz and higher rate |
| Latency | Window and run-level latency must be unambiguous | Causal time-to-alert CDF with window/stride/decision rule shown |
| Threshold transfer | High AUC does not establish deployability | Freeze one threshold on validation; report unchanged on new GPU/day/app/family |
| External benign applications | False positives were under-tested | Add ETL, database acceleration, JAX/XLA, mixed ML/HPC, multi-user/multi-query serving and independent traces |
| Platform/environment shift | NVIDIA-only clean conditions were challenged | AMD/TPU where available; MIG/VM; missing-signal, noise and cooling-shift sweeps |
| Adaptive and integrity attacks | Pre-designed framework attacks are insufficient | Real-time/surrogate optimization plus telemetry freeze/replay/tamper tests and useful-work cost |

## Exact-name blocker for five custom CUDA variants

The project scope reserves five custom-kernel variants, but their canonical implementation names have not yet been pinned in this repository. The registry therefore contains `custom_kernel_variant_1` through `_5` with status `blocked_exact_name`.

Do not rename these by inference. Close the blocker only after the team points to an exact implementation commit. Then record for each case: canonical name, kernel semantics, base model, useful-work definition, repetitions, sampler rate, matched controls, and source commit.

## RTX 3090 feasibility rules

- Native: use the exact application/checkpoint where it fits and freeze precision, batch, sequence length, clocks and GPU count.
- Multi-GPU or quantized: allowed only with explicit labels. A quantized 7B run is not an exact replacement for an unquantized 7B run.
- External hardware: full 70B/72B, Mixtral and DeepSeek-V3/R1 checkpoints remain in the contract even when the six 24 GB GPUs cannot reproduce them. Use an external accelerator or a preregistered trace-transfer test; do not claim a native 3090 reproduction.
- Roofline: characterize separately with profiling. Never collect the headline physical-sensor trace under Nsight overhead.

Roofline code and commands are documented in [`ROOFLINE_RUNBOOK.md`](ROOFLINE_RUNBOOK.md).

## How to use the registry now

```bash
python3 scripts/check_workload_coverage.py --table
python3 scripts/run_matrix.py --group training --priority P0
python3 scripts/run_matrix.py --group evasion --priority P0 \
  --repetitions 5 --gpu-set 0 --duration-s 300 \
  --write-queue results/queues/p0-evasion.csv
```

These commands validate/list/materialize an auditable queue. They intentionally do **not** launch GPU jobs. Execution becomes enabled only after each adapter has a reviewed command, dependency lock, smoke test and expected-output check.

## Adapter definition of done

An application moves from `adapter_needed` to `runnable` only when all are present:

- pinned code/model/dataset checksums and license review;
- a 30-second smoke test with nonzero useful work;
- deterministic run ID and manifest capture;
- synchronized sensor start/stop markers;
- timeout, cleanup and exit-code propagation;
- matched inference/non-ML control where applicable;
- expected output, minimum sample count and no-clipping checks; and
- one command that succeeds from a clean environment.

Private collaborator code may be used as a porting source only with permission; it must not be copied into this public repository implicitly.
