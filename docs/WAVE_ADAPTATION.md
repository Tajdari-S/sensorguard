# WAVE reproduction on RTX 3090 (E3): adaptation log

Scope decision (2026-08-24, DECISION_LOG): representative subset on one GPU
(node1), not the full 44-config grid. WAVE is a P1 offline reference.

## Pinned environment

| Item | WAVE paper/artifact | Ours (node1, `testbed-node1`) |
|---|---|---|
| Artifact commit | github.com/sept-usc/Wave | `29773cdce6bf06313dc340bce5df86d920e2ed21` (2025-12-24) |
| GPU | RTX 4090 / RTX 5080 / H100 | RTX 3090 (GA102, Ampere, 24 GB), UUID GPU-392b25f7… |
| CUDA | 12.8 | driver 595.84 (CUDA 13 era); uv-pinned torch wheel carries its own CUDA runtime |
| PyTorch | 2.7.0 | 2.7.0 via `uv sync` (artifact lockfile) |
| Nsight Compute | 2025.1.1.0 / 2025.2.1.0 | 2025.4.1 (`/opt/nvidia/nsight-compute/2025.4.1/ncu`); distro default 2024.1.1 NOT used |
| hyperfine | required for overhead | 1.19.0 |

## Ampere (GA102) counter mapping — verified 2026-08-24 via `ncu --query-metrics`

All 15 Table-5 base metrics exist under identical names on GA102:

- `smsp__sass_thread_inst_executed_op_{hfma,ffma,dfma,hadd,hmul,fadd,fmul,dadd,dmul}_pred_on.sum` — OK
- `l1tex__t_sectors_pipe_lsu_mem_global_op_{ld,st}_lookup_miss.sum` — OK
- `smsp__sass_inst_executed_op_{global_ld,global_st,shared_ld,shared_st}.sum` — OK
- `sm__ops_path_tensor_src_*` — 20 variants present (optional side-check in WAVE)

No substitutions required. Sector size s_B: to confirm 32 B on GA102 during the
first trace (paper used 32 B on 4090/5080/H100; GA102 L1 sector is also 32 B
architecturally — verify empirically against a known GEMM).

## Known deviations to report with results

1. **node1's GPU is power-limited to 200 W** (other fleet 3090s: 350/370 W).
   Counter values (FLOPs, bytes, instruction counts) are unaffected; wall-clock
   and overhead percentages are measured under this limit. Note in Table-3-style
   overhead results.
2. NCU version is newer than the artifact's (2025.4.1 vs 2025.1/2025.2); record
   any CSV schema differences in preprocessing.
3. FP32 default retained; VRAM equal to 4090's 24 GB, so the paper's 4090 grid
   feasibility (d up to ~5120–6144) is the guide.

## Planned representative subset (per approved plan)

- Lower bound: GPT-2/LLaMA/Qwen at (L=8): d=512, 768, 1024, 2048 (the artifact's
  "small + medium" set, 12 configs), plus batch sweep b∈{1,2,4,8,16} at d=768.
- Upper bound: no-split + 3 split cases at d=1024/d_ffn=4096, g=256.
- Overhead: hyperfine wall-clock for {unprofiled, single-metric, full-9-metric}
  on 3 representative configs.

## Results (2026-08-24 campaign, node1 RTX 3090)

Collected: 16 lower-bound configs (artifact small+medium grid + batch sweep
b∈{1,2,4,8,16} at GPT-2 d=768), 0 collection failures; full upper-bound
split-case set (14 split configs + no-split).

| Evaluation | RTX 3090 (ours) | Paper reference |
|---|---|---|
| Lower bound, tight (false positives) | **16/16 correct (100%)** | 100% on 4090/5080/H100 |
| Lower bound, loose (false negatives) | **15/16 correct (93.8%)** | 86.8% (4090), 72.7% (5080), 93.2% (H100) |
| Upper bound, tight | **15/15 (100%)** | 100% |
| Upper bound, loose | **15/15 (100%)** | 100% |

The single false negative is GPT-2 d=768 b=1 — the same configuration that
fails on the paper's own RTX 4090 and RTX 5080 (Table 7), consistent with
their "excessive global load at odd batch sizes" failure mode. Ampere
replicates the paper's behavior closely.

Overhead (Table-3 style, hyperfine, hw + all modes): in progress.

## Status log

- 2026-08-24: artifact cloned + pinned; NCU 2025.4.1 + hyperfine installed;
  profiling permission opened (RmProfilingAdminOnly=0 via modprobe.d, live
  module reload, no reboot needed); `uv sync` started.
- 2026-08-24 (late): smoke + full representative campaign complete; results
  above. Verification logs: node1 /tmp/wave_verify_{tight,loose}.log,
  /tmp/wave_upper_{tight,loose}.log.
