#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-results/inventory}"
mkdir -p "$out_dir"

date -u +%FT%TZ > "$out_dir/captured_at_utc.txt"
hostname > "$out_dir/hostname.txt"
uname -a > "$out_dir/uname.txt"
lscpu > "$out_dir/lscpu.txt"
free -h > "$out_dir/memory.txt"
lsblk -o NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINTS > "$out_dir/storage.txt"

# GPU inventory (guarded: the analysis laptop and sensor hosts may have no GPU)
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -q > "$out_dir/nvidia-smi-q.txt"
    nvidia-smi topo -m > "$out_dir/nvidia-smi-topo.txt"
    # field names changed from pci.link.* to pcie.link.* in newer drivers
    nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,vbios_version,power.limit,clocks.max.graphics,clocks.max.memory --format=csv > "$out_dir/gpus.csv" 2>/dev/null \
        || nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,pci.link.gen.current,pci.link.width.current,vbios_version,power.limit,clocks.max.graphics,clocks.max.memory --format=csv > "$out_dir/gpus.csv"
else
    echo "nvidia-smi not present" > "$out_dir/gpus.csv"
fi

# Sensor and peripheral inventory (Aug 14 gate: every sensor channel recorded)
if command -v lsusb >/dev/null 2>&1; then
    lsusb > "$out_dir/lsusb.txt"
    lsusb -v > "$out_dir/lsusb-verbose.txt" 2>/dev/null || true
fi
if command -v lspci >/dev/null 2>&1; then
    lspci > "$out_dir/lspci.txt"
    lspci -vv > "$out_dir/lspci-verbose.txt" 2>/dev/null || true
fi
ls -l /dev/video* /dev/ttyUSB* /dev/ttyACM* > "$out_dir/capture-devices.txt" 2>&1 || true
if command -v arecord >/dev/null 2>&1; then arecord -l > "$out_dir/audio-capture.txt" 2>&1 || true; fi
if command -v ip >/dev/null 2>&1; then ip -d link > "$out_dir/ip-link.txt" 2>&1 || true; fi
if command -v sensors >/dev/null 2>&1; then sensors > "$out_dir/lm-sensors.txt" 2>&1 || true; fi
if command -v dmidecode >/dev/null 2>&1; then
    # requires root; skip silently otherwise
    sudo -n dmidecode > "$out_dir/dmidecode.txt" 2>/dev/null || echo "dmidecode requires root" > "$out_dir/dmidecode.txt"
fi
# PicoScope enumeration: USB IDs 0ce9:* plus installed driver packages
{ lsusb 2>/dev/null | grep -i "0ce9\|pico" || echo "no PicoScope USB devices"; } > "$out_dir/picoscope-usb.txt"
if command -v dpkg >/dev/null 2>&1; then
    { dpkg -l 2>/dev/null | grep -iE "picoscope|libps[0-9]" || echo "no PicoScope packages"; } > "$out_dir/picoscope-packages.txt"
fi

# Software telemetry stack
if command -v nvcc >/dev/null 2>&1; then nvcc --version > "$out_dir/nvcc.txt"; fi
if command -v ncu >/dev/null 2>&1; then ncu --version > "$out_dir/ncu.txt"; fi
if command -v nsys >/dev/null 2>&1; then nsys --version > "$out_dir/nsys.txt"; fi
if command -v dcgmi >/dev/null 2>&1; then dcgmi --version > "$out_dir/dcgmi.txt" 2>&1 || true; fi
if command -v docker >/dev/null 2>&1; then docker version > "$out_dir/docker-version.txt" 2>&1 || true; fi
if command -v python3 >/dev/null 2>&1; then
    python3 --version > "$out_dir/python3.txt"
    python3 -m pip list > "$out_dir/pip-list.txt" 2>/dev/null || true
fi

echo "Inventory written to $out_dir"
