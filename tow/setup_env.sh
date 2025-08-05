#!/bin/bash

echo "🌳 Setting up TOW (Trees Outside Woodland) conda environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Create the environment from the yml file
echo "📦 Creating conda environment from environment.yml..."
conda env create -f environment.yml

if [ $? -eq 0 ]; then
    echo "✅ Environment created successfully!"
    echo ""
    echo "🚀 To activate the environment, run:"
    echo "   conda activate tow"
    echo ""
    echo "📋 To verify installation, run:"
    echo "   python -c \"import geopandas, rasterio, shapely; print('All packages installed successfully!')\""
    echo ""
    echo "🔧 To update the environment later, run:"
    echo "   conda env update -f environment.yml"
else
    echo "❌ Failed to create environment"
    exit 1
fi 