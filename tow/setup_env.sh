#!/bin/bash

# Wet Woodland Research Environment Setup
# ======================================

set -e  # Exit on any error

echo "🌲 Setting up Wet Woodland Research Environment"
echo "================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  No NVIDIA GPU detected - will use CPU only"
fi

# Create environment
echo ""
echo "📦 Creating conda environment 'wwr'..."
conda env create -f environment.yml

# Activate environment
echo ""
echo "🔧 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate wwr

# Verify PyTorch CUDA installation
echo ""
echo "🧪 Testing PyTorch CUDA installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️  CUDA not available - will use CPU')
"

# Test segmentation models
echo ""
echo "🧪 Testing segmentation_models_pytorch..."
python -c "
import segmentation_models_pytorch as smp
print(f'SMP version: {smp.__version__}')
model = smp.UnetPlusPlus('resnet34', in_channels=67, classes=1)
print(f'✅ UNet++ model created successfully')
print(f'   Parameters: {sum(p.numel() for p in model.parameters()):,}')
"

# Test geospatial libraries
echo ""
echo "🧪 Testing geospatial libraries..."
python -c "
import geopandas as gpd
import rasterio
import fiona
print('✅ Geospatial libraries working')
"

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "To activate the environment in the future:"
echo "   conda activate wwr"
echo ""
echo "To start training:"
echo "   python tow/src/trainer.py --data-dir path/to/features --labels-dir path/to/labels"
echo ""
echo "Happy training! 🌲" 