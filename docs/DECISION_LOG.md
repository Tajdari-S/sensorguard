# Decision log

Record every scope, sensor, split, threshold, exclusion, and claim decision here.

| Date | Decision | Evidence available before decision | Owner | Consequence |
|---|---|---|---|---|
| 2026-08-13 | Target ASPLOS 2027 September full-paper deadline on Sep 9 AoE; no abstract deadline | Official CFP | Sabiha | Freeze upload-ready PDF Sep 8 |
| 2026-08-13 | Electrical sensing is P0; other physical modalities start as P1/P2 | Prior hard-evasion weakness at 1 Hz and current instrument inventory | Sabiha | Gate weaker sensors on Aug 22 |
| 2026-08-13 | Propose one false alert per 24 GPU-hours as the primary operating budget, with 8-hour and 72-hour sensitivity points | Balances deployment meaning with negative-exposure feasible on six GPUs before the deadline; requires team confirmation by Aug 15 | Unassigned analysis owner | Freeze or revise before data inspection |
| 2026-08-24 | Fleet is 8x RTX 3090 across three nodes (verifier 2, node1 1, node2 5), not 6 as originally spec'd | Remote nvidia-smi/lspci verification on all nodes | Robi | SYSTEM_SPEC corrected; splits and holdouts use the 8-GPU inventory |
| 2026-08-24 | Add DCGM as an official tier-0 software telemetry modality alongside NVML | User confirmation; DCGM absent from all nodes, install approved; WAVE discussion cites DCGM-like reporting as the deployable path | Robi | Install nvidia-dcgm fleet-wide; add dcgm to manifest schema and Gate-1 table |
| 2026-08-24 | WAVE (E3) replicated as representative subset (~15 decode configs + 3-4 split cases + overhead) on node1 only | User confirmation; E3 is P1 and full 44-config grid competes with P0 collection given >1000% profiling overhead | Robi | Adaptation log + Ampere counter-mapping table become paper artifacts |
| 2026-08-24 | Proposed values for the five unresolved preregistration keys: held-out GPU = GPU-127e16de (node2), held-out evasion family = fused_update_kernel, thermal band +/-2 C, max missing fraction 0.01, max alignment error 100 ms (electrical <=1 ms to power edge) | No collection data inspected; choices made before any model/threshold work | Robi (proposer) | Team must countersign at freeze; validate_preregistration.py now fails on unresolved keys and emits the sha256 to record here |
| 2026-08-24 | Thermal camera, ultrasound, thermistors, SDR, and third PicoScope are physically unattached; P0 proceeds with NVML + DCGM + electrical (verifier scopes) only | Remote lsusb//dev enumeration on all nodes | Robi | Technician asked to attach remaining instruments; P1/P2 modalities gated at Gate 1 per TIMELINE scope cuts |
