# System and sensor specification

## Confirmed inventory

- Six NVIDIA RTX 3090 GPUs, Ampere, 24 GB each.
- Three oscilloscopes; current plan assumes at most three GPUs can be electrically instrumented simultaneously until channel/probe count is confirmed.
- One thermal camera.
- One visible-light camera restricted to equipment-only regions of interest.
- One passive fiber network tap; it cannot observe intra-node PCIe or NVLink traffic.
- Contact temperature sensors.

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
| Scope model/bandwidth/sample rate/firmware | TBD |
| Probe/shunt model, isolation, accuracy, placement | TBD |
| Thermal camera model/rate/resolution/accuracy | TBD |
| Visible camera model/rate/resolution | TBD |
| Fiber tap model/link speed/timestamp source | TBD |
| Contact sensor model/rate/accuracy/placement | TBD |
| Shared time source and measured alignment error | TBD |
| Ambient inlet sensor and fan/cooling policy | TBD |

## Safety boundary

Use only manufacturer-rated isolated probes, instrumented cables, or approved shunts. Do not probe mains or exposed GPU rails without a qualified lab procedure. Archive the approved wiring diagram and probe limits before energizing the system.

