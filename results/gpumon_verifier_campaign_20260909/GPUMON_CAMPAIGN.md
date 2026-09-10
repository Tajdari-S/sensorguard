# GPU-monitoring verifier campaign — coverage summary

All 96 runnable single-GPU workloads from `github.com/robirahman/GPU-monitoring`
were re-run on the **verifier** node (1× RTX 3090) on 2026-09-09, each with
synchronized NVML + DCGM logging (via `scripts/loggers/supervisor.py`) and a
concurrent 6-unit PicoScope capture. Run stamp `20260909T155817Z`. Raw traces
(nvml.csv, dcgm.tsv, pico `.npy`) live in `data/runs/gpumon_*`; this directory
holds only the derived summary.

Runs: **96** | NVML+DCGM captured: **96/96** | completed with genuine
sustained GPU load (≥30 samples at util>50% & mem>1 GB): **51**.

## Caveats (read before using this data)

- **PicoScope is UNCOUPLED.** A 2026-09-09 idle-vs-load test showed 0.0 mV
  delta across all 6 units, and the `picoP2P(mV)` column here is flat (~54–91 mV)
  regardless of GPU load. The probes are not on the verifier's power rails
  (node1, where they are wired, is offline). Pico traces are captured and
  dual-timestamped (CLOCK_MONOTONIC_RAW + epoch_ns) for completeness and
  future alignment, but are **not** a coupled electrical signal for these runs.
  A machine-readable flag `pico_channel.json` (`{"coupled": false,
  "use_as_feature": false, ...}`) is written into every `data/runs/gpumon_*`
  run dir, and `PICO_UNCOUPLED_20260909T155817Z.json` here mirrors it —
  feature-extraction/fusion code must read this and exclude the pico channel
  from any verifier-GPU features. The raw pico traces are retained as the
  documented null control, not deleted.
- **Thermal is N/A** for the verifier (camera aims at node2; see
  `docs/THERMAL_FINDINGS.md`).
- **"sustained_load"** counts NVML samples with util>50% AND mem>1 GB. The two
  GEMM load-markers bracketing every run add ~12 such samples even to an idle
  or failed body, so treat sustained_load < ~20 as "markers only, no workload
  load" and the 51 runs above ~30 as genuine sustained workload telemetry.
- **The 17 `workload_exit_1` runs are missing-dependency failures, not pipeline
  faults** — every one still captured NVML+DCGM+6-pico (~44 s of startup +
  marker telemetry), usable as short-negative coverage:
  - external tools absent on the verifier: `blender_bmw`, `gromacs_adh`,
    `ffmpeg_nvenc` (no blender/gmx/ffmpeg);
  - LoRA/PEFT backend absent: all 5 `llm_train_*_lora` + `whitebox_lora_N5/N10/N20`;
  - quantized/large-model or model-specific deps: `llm_infer_mistral_7b_q4`
    (GPTQ), `llm_infer_qwen25_7b_fp16`, `qwen25_inference`,
    `deepseek_r1_inference`, `whisper_inference`, `whitebox_oracle`.

## Status breakdown

- `completed`: 79
- `workload_exit_1`: 17

## Per-workload

| workload | status | nvml | sust-load | maxP(W) | maxMem | dcgm | pico | picoP2P(mV) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| adversarial_A_util_modulation | completed | 338 | 33 | 355 | 2590 | 338 | 6 | 90.7 |
| adversarial_B_low_util | completed | 338 | 12 | 350 | 1528 | 337 | 6 | 54.4 |
| adversarial_D_temporal_disruption | completed | 352 | 290 | 350 | 4752 | 352 | 6 | 72.6 |
| adversarial_E_memory_minimal | completed | 338 | 13 | 351 | 1528 | 338 | 6 | 72.6 |
| adversarial_F_interleave_30 | completed | 338 | 12 | 351 | 1608 | 337 | 6 | 54.4 |
| adversarial_F_interleave_50 | completed | 338 | 13 | 351 | 1608 | 337 | 6 | 72.6 |
| adversarial_G_clock_throttled | completed | 338 | 12 | 351 | 1528 | 337 | 6 | 54.4 |
| adversarial_H_mimicry_cufft | completed | 338 | 12 | 353 | 1550 | 337 | 6 | 72.6 |
| adversarial_H_mimicry_mining | completed | 338 | 12 | 350 | 1528 | 337 | 6 | 54.4 |
| adversarial_I_stochastic_30 | completed | 338 | 102 | 353 | 1600 | 337 | 6 | 54.4 |
| adversarial_J_pid_75 | completed | 338 | 12 | 350 | 1528 | 338 | 6 | 72.6 |
| adversarial_K_online_learning | completed | 338 | 311 | 352 | 13192 | 337 | 6 | 72.6 |
| adversarial_L_diluted_10 | completed | 338 | 203 | 350 | 13210 | 338 | 6 | 72.6 |
| adversarial_L_diluted_20 | completed | 338 | 176 | 350 | 23608 | 338 | 6 | 72.6 |
| adversarial_L_diluted_2 | completed | 338 | 277 | 350 | 13214 | 338 | 6 | 54.4 |
| adversarial_L_diluted_5 | completed | 338 | 232 | 352 | 13210 | 338 | 6 | 72.6 |
| adversarial_M_composite_memmin_10 | completed | 338 | 12 | 350 | 5000 | 338 | 6 | 54.4 |
| adversarial_M_composite_memmin_20 | completed | 338 | 12 | 350 | 5000 | 338 | 6 | 54.4 |
| adversarial_M_composite_memmin_30 | completed | 338 | 13 | 351 | 5000 | 338 | 6 | 54.4 |
| adversarial_M_composite_memmin_40 | completed | 338 | 12 | 350 | 5000 | 338 | 6 | 72.6 |
| adversarial_N_grad_accum_16 | completed | 338 | 310 | 353 | 13194 | 338 | 6 | 72.6 |
| adversarial_N_grad_accum_4 | completed | 338 | 310 | 351 | 13194 | 337 | 6 | 72.6 |
| adversarial_N_grad_accum_8 | completed | 338 | 308 | 351 | 13194 | 338 | 6 | 72.6 |
| adversarial_composite_00 | completed | 158 | 121 | 355 | 6688 | 158 | 6 | 54.4 |
| adversarial_composite_30 | completed | 158 | 122 | 353 | 6688 | 158 | 6 | 72.6 |
| adversarial_composite_50 | completed | 158 | 121 | 350 | 6688 | 158 | 6 | 54.4 |
| adversarial_composite_70 | completed | 158 | 116 | 350 | 6688 | 158 | 6 | 72.6 |
| adversarial_composite_90 | completed | 158 | 14 | 350 | 6688 | 158 | 6 | 72.6 |
| adversarial_composite_95 | completed | 158 | 18 | 350 | 6688 | 158 | 6 | 54.4 |
| bert_sst2 | completed | 95 | 56 | 350 | 4838 | 95 | 6 | 72.6 |
| bert_sst2_amp | completed | 70 | 33 | 350 | 4120 | 70 | 6 | 54.4 |
| blender_bmw | workload_exit_1 | 34 | 12 | 350 | 1528 | 34 | 6 | 72.6 |
| cufft_benchmark | completed | 636 | 611 | 350 | 1546 | 635 | 6 | 72.6 |
| cufft_benchmark_standalone | completed | 336 | 312 | 350 | 5386 | 335 | 6 | 72.6 |
| deepseek_r1_inference | workload_exit_1 | 40 | 12 | 350 | 1528 | 40 | 6 | 54.4 |
| dilution_sweep_N10 | completed | 338 | 199 | 350 | 13210 | 337 | 6 | 72.6 |
| dilution_sweep_N1 | completed | 338 | 310 | 350 | 13212 | 338 | 6 | 54.4 |
| dilution_sweep_N20 | completed | 338 | 185 | 353 | 13210 | 338 | 6 | 72.6 |
| dilution_sweep_N2 | completed | 338 | 274 | 352 | 13214 | 337 | 6 | 54.4 |
| dilution_sweep_N3 | completed | 338 | 252 | 350 | 13212 | 338 | 6 | 54.4 |
| dilution_sweep_N5 | completed | 338 | 231 | 353 | 13210 | 338 | 6 | 72.6 |
| dilution_sweep_inference | completed | 338 | 238 | 351 | 2938 | 338 | 6 | 90.7 |
| ethash_cuda | completed | 336 | 312 | 350 | 17196 | 335 | 6 | 54.4 |
| ffmpeg_nvenc | workload_exit_1 | 34 | 12 | 350 | 1528 | 34 | 6 | 54.4 |
| gpt2_wikitext2 | completed | 155 | 107 | 356 | 10502 | 154 | 6 | 54.4 |
| gpt2_wikitext2_amp | completed | 113 | 66 | 355 | 10050 | 113 | 6 | 72.6 |
| gromacs_adh | workload_exit_1 | 34 | 12 | 353 | 1528 | 34 | 6 | 72.6 |
| idle | completed | 154 | 12 | 350 | 1528 | 154 | 6 | 54.4 |
| llm_infer_bge_embed_bs128 | completed | 359 | 312 | 359 | 1648 | 358 | 6 | 54.4 |
| llm_infer_clip_embed_batched | completed | 343 | 313 | 359 | 2436 | 342 | 6 | 72.6 |
| llm_infer_clip_embed_bs256 | completed | 366 | 315 | 359 | 4794 | 366 | 6 | 72.6 |
| llm_infer_mistral_7b_q4 | workload_exit_1 | 43 | 13 | 350 | 1528 | 43 | 6 | 72.6 |
| llm_infer_phi3_mini | completed | 437 | 18 | 350 | 8340 | 436 | 6 | 72.6 |
| llm_infer_phi3_mini_seq2048 | completed | 349 | 320 | 350 | 9342 | 349 | 6 | 72.6 |
| llm_infer_qwen25_1p5b | completed | 396 | 12 | 350 | 3800 | 395 | 6 | 72.6 |
| llm_infer_qwen25_1p5b_seq4096 | completed | 355 | 19 | 350 | 4342 | 355 | 6 | 90.7 |
| llm_infer_qwen25_1p5b_streaming | completed | 361 | 14 | 350 | 3836 | 361 | 6 | 54.4 |
| llm_infer_qwen25_3b | completed | 424 | 19 | 351 | 6774 | 423 | 6 | 54.4 |
| llm_infer_qwen25_3b_bs16 | completed | 348 | 317 | 352 | 7262 | 347 | 6 | 72.6 |
| llm_infer_qwen25_3b_bs4 | completed | 343 | 17 | 353 | 6876 | 343 | 6 | 72.6 |
| llm_infer_qwen25_7b_fp16 | workload_exit_1 | 34 | 13 | 353 | 1528 | 34 | 6 | 54.4 |
| llm_infer_sdxl | completed | 413 | 317 | 360 | 15246 | 412 | 6 | 72.6 |
| llm_infer_smollm2_1p7b | completed | 400 | 13 | 350 | 4232 | 399 | 6 | 72.6 |
| llm_infer_tinyllama_bs8 | completed | 373 | 19 | 351 | 3086 | 373 | 6 | 72.6 |
| llm_infer_tinyllama_seq1024 | completed | 346 | 19 | 350 | 3004 | 346 | 6 | 72.6 |
| llm_infer_tinyllama_seq2048 | completed | 346 | 18 | 350 | 3086 | 346 | 6 | 54.4 |
| llm_infer_tinyllama_seq4096 | completed | 342 | 32 | 350 | 3204 | 341 | 6 | 72.6 |
| llm_infer_vit_l_bs64 | completed | 361 | 313 | 358 | 1894 | 361 | 6 | 72.6 |
| llm_train_mistral_7b_lora | workload_exit_1 | 158 | 16 | 350 | 14544 | 157 | 6 | 72.6 |
| llm_train_phi3_mini_lora | workload_exit_1 | 43 | 15 | 353 | 8020 | 43 | 6 | 72.6 |
| llm_train_qwen25_1p5b_lora | workload_exit_1 | 42 | 12 | 350 | 3730 | 42 | 6 | 72.6 |
| llm_train_qwen25_3b_lora | workload_exit_1 | 43 | 14 | 350 | 6688 | 43 | 6 | 72.6 |
| llm_train_smollm2_1p7b_lora | workload_exit_1 | 42 | 13 | 353 | 3996 | 42 | 6 | 54.4 |
| mining_ethash_proxy | completed | 636 | 12 | 353 | 1796 | 635 | 6 | 54.4 |
| nbody_sim | completed | 636 | 611 | 363 | 1604 | 635 | 6 | 72.6 |
| nbody_sim_standalone | completed | 336 | 313 | 352 | 14084 | 335 | 6 | 72.6 |
| pytorch_mlp_cifar10 | completed | 133 | 12 | 354 | 1528 | 133 | 6 | 54.4 |
| pytorch_resnet_cifar10 | completed | 145 | 108 | 360 | 4524 | 145 | 6 | 54.4 |
| pytorch_resnet_cifar10_amp | completed | 103 | 65 | 362 | 2592 | 103 | 6 | 72.6 |
| qwen25_inference | workload_exit_1 | 40 | 11 | 353 | 1528 | 40 | 6 | 72.6 |
| rendering_proxy | completed | 636 | 12 | 351 | 1528 | 635 | 6 | 72.6 |
| resnet50_inference | completed | 639 | 434 | 352 | 4974 | 638 | 6 | 72.6 |
| resnet50_inference_standalone | completed | 139 | 85 | 350 | 4976 | 138 | 6 | 54.4 |
| whisper_inference | workload_exit_1 | 36 | 13 | 351 | 1528 | 36 | 6 | 72.6 |
| whitebox_diluted_N10 | completed | 342 | 313 | 350 | 11918 | 342 | 6 | 72.6 |
| whitebox_diluted_N10_ckpt | completed | 343 | 312 | 354 | 11918 | 342 | 6 | 54.4 |
| whitebox_diluted_N20 | completed | 343 | 313 | 350 | 11918 | 342 | 6 | 72.6 |
| whitebox_diluted_N20_ckpt | completed | 343 | 313 | 351 | 11918 | 342 | 6 | 72.6 |
| whitebox_diluted_N50 | completed | 342 | 314 | 353 | 11918 | 342 | 6 | 54.4 |
| whitebox_diluted_N5 | completed | 343 | 313 | 350 | 11918 | 342 | 6 | 54.4 |
| whitebox_diluted_N5_ckpt | completed | 343 | 313 | 351 | 11918 | 342 | 6 | 54.4 |
| whitebox_inference | completed | 362 | 311 | 350 | 3008 | 362 | 6 | 54.4 |
| whitebox_lora_N10 | workload_exit_1 | 42 | 13 | 351 | 2874 | 41 | 6 | 54.4 |
| whitebox_lora_N20 | workload_exit_1 | 41 | 12 | 353 | 2874 | 41 | 6 | 72.6 |
| whitebox_lora_N5 | workload_exit_1 | 41 | 13 | 354 | 2874 | 41 | 6 | 54.4 |
| whitebox_oracle | workload_exit_1 | 36 | 11 | 354 | 1528 | 36 | 6 | 72.6 |
