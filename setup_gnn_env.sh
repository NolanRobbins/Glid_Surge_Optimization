#!/bin/bash
# =============================================================================
# GNN Environment Setup for NVIDIA Grace Blackwell (GX10)
# =============================================================================
# This script provides two options:
#   1. Docker-based (recommended) - Most reliable, fully compatible
#   2. Local venv fix - Faster if it works
# =============================================================================

set -e

PROJECT_DIR="/home/asus/Desktop/Glid_Surge_Optimization"
cd "$PROJECT_DIR"

echo "=============================================="
echo "GLID SURGE - GNN ENVIRONMENT SETUP"
echo "NVIDIA Grace Blackwell (GX10)"
echo "=============================================="

# Check GPU
echo ""
echo "[1/5] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Option selector
echo ""
echo "Choose setup method:"
echo "  1) Docker container (recommended - fully compatible)"
echo "  2) Fix local venv (faster, may have issues)"
echo ""
read -p "Enter choice [1]: " CHOICE
CHOICE=${CHOICE:-1}

if [ "$CHOICE" == "1" ]; then
    echo ""
    echo "[2/5] Building Docker container..."
    docker build -f Dockerfile.gnn -t glid-gnn:latest .
    
    echo ""
    echo "[3/5] Testing GPU access in container..."
    docker run --rm --gpus all glid-gnn:latest python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

import cudf
print(f'cuDF: {cudf.__version__}')

import cugraph
print(f'cuGraph: {cugraph.__version__}')

import torch_geometric
print(f'PyG: {torch_geometric.__version__}')
print('✓ All systems GO!')
"
    
    echo ""
    echo "=============================================="
    echo "DOCKER SETUP COMPLETE!"
    echo "=============================================="
    echo ""
    echo "To run GNN training:"
    echo "  docker compose up gnn-train"
    echo ""
    echo "For interactive shell:"
    echo "  docker compose run gnn-shell"
    echo ""
    
else
    echo ""
    echo "[2/5] Fixing local venv..."
    
    # Set CUDA paths
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
    
    source .venv/bin/activate
    
    # Try installing with --no-deps to skip bundled NVIDIA libs
    echo "Installing PyTorch (using system CUDA libs)..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126 \
        --no-deps 2>/dev/null || true
    
    # Install dependencies separately
    pip install filelock fsspec jinja2 networkx setuptools sympy typing-extensions pillow
    
    echo ""
    echo "[3/5] Testing installation..."
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
    
    echo ""
    echo "=============================================="
    echo "LOCAL VENV SETUP COMPLETE!"
    echo "=============================================="
    echo ""
    echo "To run GNN training:"
    echo "  source .venv/bin/activate"
    echo "  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    echo "  python train_gpu.py"
    echo ""
fi

echo "[4/5] Clearing uv cache to free space..."
rm -rf ~/.cache/uv/archive-v0/* 2>/dev/null || true
echo "Cache cleared."

echo ""
echo "[5/5] Setup complete!"






