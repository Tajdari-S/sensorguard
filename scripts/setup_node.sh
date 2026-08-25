#!/usr/bin/env bash
# Idempotent per-node environment setup for the SensorGuard fleet.
# Usage: bash scripts/setup_node.sh [role]   role: workload | sensor (default: workload)
set -euo pipefail

role="${1:-workload}"
venv="$HOME/sg-venv"

# Base tooling (needs sudo; skipped silently if venv already usable)
if ! python3 -m venv --help >/dev/null 2>&1 || ! command -v pip3 >/dev/null 2>&1; then
    if sudo -n true 2>/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3-venv python3-pip
    else
        echo "WARNING: python3-venv/pip missing and no passwordless sudo; venv creation may fail" >&2
    fi
fi

if [ ! -d "$venv" ]; then
    python3 -m venv "$venv"
fi
# shellcheck disable=SC1091
source "$venv/bin/activate"
pip -q install --upgrade pip

# Common scientific + telemetry stack. nvidia-ml-py is the maintained NVML binding.
pip -q install numpy pandas scikit-learn pyyaml nvidia-ml-py

# CUDA PyTorch for workload nodes and any node running calibration loads.
python - <<'EOF' || pip -q install torch
import sys
try:
    import torch
    sys.exit(0 if torch.cuda.is_available() else 1)
except Exception:
    sys.exit(1)
EOF

if [ "$role" = "sensor" ]; then
    # Verification/analysis + instrument host extras
    pip -q install z3-solver picosdk
fi

python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available(),
      "devices:", torch.cuda.device_count())
EOF
echo "setup_node complete: role=$role venv=$venv"
