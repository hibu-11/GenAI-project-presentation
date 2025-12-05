#!/bin/bash
# Setup script for Transformer Code Documentation Generation Project

echo "=========================================="
echo "  Transformer Code Documentation Setup"
echo "=========================================="
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

if ! python3 -c 'import sys; assert sys.version_info >= (3, 8)' 2>/dev/null; then
    echo "ERROR: Python 3.8+ required"
    exit 1
fi

# Create virtual environment
echo ""
echo "[2/6] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "[3/6] Activating virtual environment..."
source venv/bin/activate || {
    echo "ERROR: Failed to activate virtual environment"
    exit 1
}
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "[4/6] Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "[5/6] Installing dependencies..."
echo "This may take several minutes..."

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet || {
    echo "WARNING: CUDA version of PyTorch not available, installing CPU version..."
    pip install torch torchvision torchaudio --quiet
}

pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "[6/6] Creating project directories..."
mkdir -p data/processed
mkdir -p models/checkpoints
mkdir -p evaluation/results
mkdir -p notebooks
mkdir -p logs
echo "✓ Directories created"

# Print GPU info
echo ""
echo "=========================================="
echo "  System Information"
echo "=========================================="
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Note: Running on CPU. GPU recommended for training.")
EOF

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Download and preprocess data:"
echo "   cd data && python preprocess.py"
echo ""
echo "3. Run the dashboard:"
echo "   streamlit run dashboard/app.py"
echo ""
echo "4. Check the README.md for more information"
echo ""
echo "Happy coding! 🚀"
