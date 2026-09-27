#!/bin/bash

# Stop on error
set -e

echo "🚀 Starting setup script for Ubuntu..."

# Run from the repo root (the folder this script lives in)
cd "$(dirname "$0")"

# Determine if sudo is needed/available
if [ "$(id -u)" -eq 0 ]; then
    echo "Running as root, skipping sudo..."
    SUDO=""
else
    SUDO="sudo"
fi

# 1. Install System Packages (tmux, vim, git, curl)
echo "📦 Installing system packages..."
$SUDO apt-get update -y
$SUDO apt-get install -y tmux vim curl git

# 2. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# 3. Fix "missing or unsuitable terminal: xterm-ghostty"
# This happens when connecting from Ghostty terminal to a server without its terminfo.
if [[ "$TERM" == "xterm-ghostty" ]]; then
    echo "👻 Ghostty terminal detected. Applying compatibility fix..."
    echo "export TERM=xterm-256color" >> ~/.bashrc
    export TERM=xterm-256color
    echo "✅ Added 'export TERM=xterm-256color' to ~/.bashrc"
fi

# 4. Check the GPU and driver
# The locked torch wheel is built for CUDA 13, which needs NVIDIA driver >= 580.
echo "🖥️  GPU / driver:"
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found: no NVIDIA driver on this machine."
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# 5. Set up the project environment
echo "🐍 Running uv sync..."
uv sync

# 6. Make sure torch can actually use the GPU (otherwise training silently falls back to CPU)
uv run python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not visible to torch. The cu13 wheel needs NVIDIA driver >= 580; check the driver version above.'
print('✅ torch', torch.__version__, '| CUDA', torch.version.cuda, '|', torch.cuda.get_device_name(0))
"

echo "🎉 Setup complete!"
echo "Run 'source \$HOME/.local/bin/env' to activate uv in this shell."
