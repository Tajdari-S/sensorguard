# System and sensor specification

## Confirmed inventory

- Six NVIDIA RTX 3090 GPUs, Ampere, 24 GB each.
- Three Pico Technology PicoScope two-channel, 10 MHz digital oscilloscopes; exact model and probe inventory remain to be confirmed. Six channels may permit one electrical channel per GPU if the probes and approved wiring support it.
- TOPDON TC001-A thermal camera: 256 x 192 native IR resolution, 512 x 384 TISR output, 25 Hz, stated range -20 C to 550 C.
- Dodotronic USB ultrasound microphone, 200/250 kHz variant; exact model and selected maximum sampling rate remain to be confirmed.
- `sourcing map` 10K 3950B thermistors; acquisition electronics, placement, sampling rate, and calibration remain to be specified.
- 10Gtek network interface card plus TP-Link TL-SG108E gigabit switch. This is a switched-Ethernet observation path, not a passive fiber tap; document port mirroring and timestamping configuration. It cannot observe intra-node PCIe or NVLink traffic.
- NooElec NESDR SMArt Bundle for RF observation; record the exact receiver revision, antenna, center frequencies, gain, bandwidth, and approved placement.
- No separate visible-light camera is currently specified.

## Required before first paper-quality run

| Item | Value |
|---|---|
| Server make/model | TBD |
| CPU model/socket/core count | TBD |
| DRAM capacity/channels/speed | TBD |
| Storage device and free space | TBD |
| Motherboard/chipset | TBD |
| GPU UUID, SKU, VBIOS per slot | TBD |
| PCIe generation/width and NUMA topology | TBD |
| NVLink bridges/topology | TBD |
| PSU model/rating and GPU cable map | TBD |
| NIC and link speed | TBD |
| OS/kernel | TBD |
| NVIDIA driver/CUDA/cuDNN/NCCL | TBD |
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
