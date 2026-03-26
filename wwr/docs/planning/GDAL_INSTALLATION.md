# Installing GDAL Python Bindings

## Why Install GDAL?

The script uses GDAL to create **Virtual Raster (VRT)** files, which are much faster and use less disk space than merging tiles. Without GDAL, the script falls back to merging all tiles into one large GeoTIFF, which is slower and uses more storage.

**VRT vs Merged:**
- VRT: Fast, ~1KB file, instant creation
- Merged: Slow, GBs of disk space, minutes to create

## Current Status

If you see this message:
```
! VRT creation failed (GDAL not available), falling back to merging tiles...
```

Then GDAL Python bindings are not installed. The script will still work, but slower.

---

## Installation Options

### Option 1: Conda (Recommended - Easiest)

```bash
# If using conda/mamba environment
conda install -c conda-forge gdal

# Or with mamba (faster)
mamba install -c conda-forge gdal
```

**Pros**: Handles all dependencies automatically
**Cons**: Requires conda/mamba

### Option 2: System GDAL + Python Bindings

#### On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install gdal-bin python3-gdal
```

#### On macOS (with Homebrew):
```bash
brew install gdal
pip install gdal==$(gdal-config --version)
```

#### On Red Hat/CentOS:
```bash
sudo yum install gdal gdal-python3
```

**Pros**: System-wide installation
**Cons**: Version matching can be tricky

### Option 3: pip (Can be tricky)

```bash
pip install gdal
```

**Warning**: This often fails because it needs GDAL C++ libraries installed first. If you get errors, use Option 1 or 2.

---

## Verify Installation

After installing, test it:

```bash
python -c "from osgeo import gdal; print(f'GDAL {gdal.__version__} installed')"
```

Should output:
```
GDAL 3.x.x installed
```

If you get an error, GDAL is not properly installed.

---

## Alternative: Use Command-Line GDAL

If you have `gdalbuildvrt` command available but Python bindings fail:

```bash
# Check if command exists
which gdalbuildvrt
```

If this returns a path, the script will automatically use it! You don't need Python bindings.

---

## Do I Need This?

### You DON'T need GDAL if:
- ✅ You're using `--wet-woodland-raster` with a single raster file
- ✅ You have few tiles (<10) and merging is fast enough
- ✅ You have plenty of disk space for merged files

### You SHOULD install GDAL if:
- ⚠️ You have many tiles (>50)
- ⚠️ Merging takes too long
- ⚠️ You're running out of disk space
- ⚠️ You need to process multiple study areas repeatedly

---

## Troubleshooting

### Error: `ImportError: No module named 'osgeo'`
**Solution**: GDAL Python bindings not installed. Use Option 1 or 2 above.

### Error: `ImportError: DLL load failed` (Windows)
**Solution**: GDAL C++ library not in PATH. Use conda installation (Option 1).

### Error: `ModuleNotFoundError: No module named '_gdal'`
**Solution**: Version mismatch between GDAL library and Python bindings.
```bash
# Check versions
gdal-config --version  # System GDAL
python -c "import osgeo; print(osgeo.__version__)"  # Python bindings

# They should match! If not, reinstall with conda.
```

### Script still works without GDAL
**Yes!** The script automatically falls back to merging tiles. It just takes longer and uses more disk space.

---

## Performance Comparison

Example: 1000 tiles, 10m resolution, 100km² area

| Method | Time | Disk Space | Memory |
|--------|------|------------|--------|
| VRT | <1 second | ~1 KB | Low |
| Merge | 5-30 minutes | ~5 GB | High |

**Verdict**: Install GDAL if you work with many tiles!

---

## Still Having Issues?

The script will work fine without GDAL - it just merges tiles instead. If merging works for you, you can skip installing GDAL entirely.
