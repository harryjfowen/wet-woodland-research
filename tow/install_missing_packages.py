#!/usr/bin/env python3
"""
Install Missing Packages for Species Classification
==================================================

Checks for required packages and installs any missing ones.
"""

import subprocess
import sys
import importlib

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} - MISSING")
        return False

def install_package(package_name):
    """Install a package using conda or pip."""
    print(f"📦 Installing {package_name}...")
    
    # Try conda first
    try:
        result = subprocess.run(['conda', 'install', '-y', package_name], 
                              capture_output=True, text=True, check=True)
        print(f"   ✅ Installed via conda")
        return True
    except subprocess.CalledProcessError:
        print(f"   ⚠️  Conda failed, trying pip...")
        
        # Try pip
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', package_name], 
                                  capture_output=True, text=True, check=True)
            print(f"   ✅ Installed via pip")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package_name}")
            print(f"   Error: {e.stderr}")
            return False

def main():
    print("🔍 Checking required packages for species classification...")
    print("=" * 60)
    
    # Core packages
    required_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('rasterio', 'rasterio'),
        ('albumentations', 'albumentations'),
        ('segmentation-models-pytorch', 'segmentation_models_pytorch'),
        ('tqdm', 'tqdm'),
        ('geopandas', 'geopandas'),
        ('shapely', 'shapely'),
        ('scikit-learn', 'sklearn'),
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            missing_packages.append(package_name)
    
    print("\n" + "=" * 60)
    
    if missing_packages:
        print(f"📦 Installing {len(missing_packages)} missing packages...")
        print("=" * 60)
        
        for package in missing_packages:
            install_package(package)
            print()
    else:
        print("🎉 All required packages are already installed!")
    
    # Final check
    print("🔍 Final verification...")
    print("=" * 60)
    
    all_good = True
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            all_good = False
    
    if all_good:
        print("\n✅ All packages ready! You can now run:")
        print("   python test_species_data.py --data-dir data/features --labels-dir data/labels")
    else:
        print("\n❌ Some packages are still missing. Please install them manually.")
        print("   Try: conda install -c conda-forge <package_name>")

if __name__ == "__main__":
    main()
