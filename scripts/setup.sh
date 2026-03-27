#!/usr/bin/env bash
# Setup script for local (non-Docker) deployment.
# Run from the watermark-service root directory.
set -euo pipefail

VENV=".venv"
PYTHON="python3"
TRAINING_DIR="../watermark-removal/training"

echo "=== Watermark Removal Service Setup ==="

# 1. Create virtual environment
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv "$VENV"
fi
source "$VENV/bin/activate"

# 2. Install PyTorch (detect CUDA)
echo "Detecting CUDA..."
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "CUDA detected — installing PyTorch with CUDA 12.1"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "No CUDA — installing CPU-only PyTorch"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# 3. Install the service package
pip install -e ".[dev]"

# 4. Symlink checkpoints from training repo if they exist
mkdir -p models
if [ -d "$TRAINING_DIR/checkpoints" ]; then
    LATEST_REMOVAL=$(ls -t "$TRAINING_DIR/checkpoints/"*.pth 2>/dev/null | head -1)
    if [ -n "$LATEST_REMOVAL" ]; then
        ln -sf "$(realpath "$LATEST_REMOVAL")" models/removal_best.pth
        echo "Linked removal checkpoint: $LATEST_REMOVAL"
    fi
fi
if [ -d "$TRAINING_DIR/checkpoints_seg" ]; then
    LATEST_SEG=$(ls -t "$TRAINING_DIR/checkpoints_seg/"*.pth 2>/dev/null | head -1)
    if [ -n "$LATEST_SEG" ]; then
        ln -sf "$(realpath "$LATEST_SEG")" models/seg_best.pth
        echo "Linked seg checkpoint: $LATEST_SEG"
    fi
fi

echo ""
echo "=== Setup complete ==="
echo "To run the service:"
echo "  source $VENV/bin/activate"
echo "  uvicorn src.app:create_app --factory --host 0.0.0.0 --port 8000"
