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

## Status log

- 2026-08-24: artifact cloned + pinned; NCU 2025.4.1 + hyperfine installed;
  profiling permission opened (RmProfilingAdminOnly=0 via modprobe.d, live
  module reload, no reboot needed); `uv sync` started.
