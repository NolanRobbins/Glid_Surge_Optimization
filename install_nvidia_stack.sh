#!/bin/bash
# ==============================================================================
# NVIDIA DGX Spark Installation Script
# Glid Surge Optimization - Full GPU Stack
# ==============================================================================

set -e

echo "=============================================="
echo "NVIDIA DGX SPARK SETUP FOR GLID OPTIMIZATION"
echo "=============================================="

# Check for NVIDIA GPU
echo ""
echo "[1/6] Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
    echo "WARNING: nvidia-smi not found. Ensure NVIDIA drivers are installed."
fi

# Check CUDA version
echo ""
echo "[2/6] Checking CUDA..."
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
else
    echo "CUDA toolkit not in PATH. Checking for container CUDA..."
    if [ -d "/usr/local/cuda" ]; then
        echo "CUDA found at /usr/local/cuda"
        export PATH=/usr/local/cuda/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    fi
fi

# Create/activate virtual environment
echo ""
echo "[3/6] Setting up Python environment..."
cd /home/asus/Desktop/Glid_Surge_Optimization

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 12.1
echo ""
echo "[4/6] Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Install PyTorch Geometric
echo ""
echo "[5/6] Installing PyTorch Geometric..."
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# Install RAPIDS (cuDF, cuML, cuGraph)
echo ""
echo "[6/6] Installing NVIDIA RAPIDS..."
# RAPIDS is best installed via conda, but we can try pip for CUDA 12
pip install --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12 cuml-cu12 cugraph-cu12 rmm-cu12

# Install remaining requirements
echo ""
echo "[FINAL] Installing remaining packages..."
pip install -r requirements.txt --ignore-installed cudf-cu12 cuml-cu12 cugraph-cu12

echo ""
echo "=============================================="
echo "INSTALLATION COMPLETE!"
echo "=============================================="
echo ""
echo "Verify installation:"
echo "  python3 -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}')\""
echo "  python3 -c \"import cugraph; print('cuGraph OK')\""
echo "  python3 -c \"import cuml; print('cuML OK')\""
echo "  python3 -c \"import torch_geometric; print('PyG OK')\""
echo ""
echo "Run the full pipeline:"
echo "  python main.py --stage all"
echo ""






