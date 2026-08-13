#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-results/inventory}"
mkdir -p "$out_dir"

date -u +%FT%TZ > "$out_dir/captured_at_utc.txt"
uname -a > "$out_dir/uname.txt"
lscpu > "$out_dir/lscpu.txt"
free -h > "$out_dir/memory.txt"
lsblk -o NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINTS > "$out_dir/storage.txt"
nvidia-smi -q > "$out_dir/nvidia-smi-q.txt"
nvidia-smi topo -m > "$out_dir/nvidia-smi-topo.txt"
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,pci.link.gen.current,pci.link.width.current,vbios_version,power.limit,clocks.max.graphics,clocks.max.memory --format=csv > "$out_dir/gpus.csv"

if command -v nvcc >/dev/null 2>&1; then nvcc --version > "$out_dir/nvcc.txt"; fi
if command -v ncu >/dev/null 2>&1; then ncu --version > "$out_dir/ncu.txt"; fi
if command -v nsys >/dev/null 2>&1; then nsys --version > "$out_dir/nsys.txt"; fi
if command -v docker >/dev/null 2>&1; then docker version > "$out_dir/docker-version.txt" 2>&1 || true; fi

echo "Inventory written to $out_dir"

