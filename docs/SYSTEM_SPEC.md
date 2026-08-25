# System and sensor specification

## Confirmed inventory (remote verification 2026-08-24)

- **Eight** NVIDIA RTX 3090 GPUs, Ampere GA102, 24 GB each, across three Tailscale-reachable nodes (driver 595.84 on all):

| Node | Tailscale IP | GPUs | Profilers | Role |
|---|---|---|---|---|
| verifier | 100.68.211.69 | 2× RTX 3090 | none | physical-sensor host, analysis/verification; 128 CPU cores, 123 GiB RAM; PicoScope 7 + VNC :5901 |
| node1 (`testbed-node1`) | 100.105.22.23 | 1× RTX 3090 | ncu, nsys | Nsight/WAVE characterization |
| node2 | 100.79.72.8 | 5× RTX 3090 | ncu, nsys | NVML/DCGM workload corpus |

  GPU UUIDs: verifier `GPU-42e33043…`, `GPU-bff95c83…`; node1 `GPU-392b25f7…`; node2 `GPU-9b96eb67…`, `GPU-37ffc98b…`, `GPU-5443d59d…`, `GPU-46bf67bb…`, `GPU-127e16de…`. The original six-GPU assumption is superseded; see DECISION_LOG 2026-08-24.
- Software telemetry tiers: NVML (tier 0, baseline) and **DCGM (added 2026-08-24 as an official tier-0 modality; not yet installed on any node)**.
- **Two** Pico Technology PicoScope 2000-series two-channel oscilloscopes, enumerated on verifier USB (ID 0ce9:1007); the third spec'd scope is not attached to any node. Exact model and probe inventory remain to be confirmed. Four present channels can cover verifier's two GPUs; six-channel coverage requires the third scope and approved wiring.
- TOPDON TC001-A thermal camera: 256 x 192 native IR resolution, 512 x 384 TISR output, 25 Hz, stated range -20 C to 550 C. **Not attached to any node as of 2026-08-24** (no /dev/video* present).
- Dodotronic USB ultrasound microphone, 200/250 kHz variant; exact model and selected maximum sampling rate remain to be confirmed. **Not attached to any node as of 2026-08-24** (no capture device present).
- `sourcing map` 10K 3950B thermistors; acquisition electronics, placement, sampling rate, and calibration remain to be specified. **No acquisition device attached to any node as of 2026-08-24** (no ttyUSB/ttyACM present).
- 10Gtek network interface card plus TP-Link TL-SG108E gigabit switch. This is a switched-Ethernet observation path, not a passive fiber tap; document port mirroring and timestamping configuration. It cannot observe intra-node PCIe or NVLink traffic.
- NooElec NESDR SMArt Bundle for RF observation; record the exact receiver revision, antenna, center frequencies, gain, bandwidth, and approved placement. **Not attached to any node as of 2026-08-24.**
- No separate visible-light camera is currently specified.

## Required before first paper-quality run

| Item | Value |
|---|---|
| Server make/model | TBD |
| CPU model/socket/core count | TBD |
| DRAM capacity/channels/speed | TBD |
| Storage device and free space | TBD |
| Motherboard/chipset | TBD |
| GPU UUID, SKU, VBIOS per slot | UUIDs recorded above (2026-08-24); SKU/VBIOS per slot TBD via inventory capture |
| PCIe generation/width and NUMA topology | TBD |
| NVLink bridges/topology | TBD |
| PSU model/rating and GPU cable map | TBD |
| NIC and link speed | TBD |
| OS/kernel | verifier: Ubuntu, kernel 7.0.0-30-generic; node1/node2 TBD via inventory capture |
| NVIDIA driver/CUDA/cuDNN/NCCL | driver 595.84 (all nodes); CUDA/cuDNN/NCCL TBD via inventory capture |
| DCGM version/service state | TBD (not installed on any node as of 2026-08-24; installation approved) |
| PyTorch and container digest | TBD |
| Nsight Compute/Systems versions | TBD |
| Scope model/bandwidth/sample rate/firmware | Pico Technology PicoScope, 2 channels, 10 MHz; exact model/rate/firmware TBD |
| Probe/shunt model, isolation, accuracy, placement | TBD |
| Thermal camera model/rate/resolution/accuracy | TOPDON TC001-A; 25 Hz; 256 x 192 IR; 512 x 384 TISR; -20 C to 550 C stated range; accuracy TBD |
| Ultrasound microphone | Dodotronic USB Ultrasound Microphone, 200/250 kHz variant; exact model/rate/firmware TBD |
| Network observation path | 10Gtek NIC + TP-Link TL-SG108E 1 GbE switch; port mirroring/timestamp source TBD |
| RF receiver/antenna | NooElec NESDR SMArt Bundle; revision/frequency plan/gain/bandwidth TBD |
| Contact sensor model/rate/accuracy/placement | sourcing map 10K 3950B thermistors; logger/rate/calibration/placement TBD |
| Shared time source and measured alignment error | TBD |
| Ambient inlet sensor and fan/cooling policy | TBD |

## Safety boundary

Use only manufacturer-rated isolated probes, instrumented cables, or approved shunts. Do not probe mains or exposed GPU rails without a qualified lab procedure. Archive the approved wiring diagram and probe limits before energizing the system.
