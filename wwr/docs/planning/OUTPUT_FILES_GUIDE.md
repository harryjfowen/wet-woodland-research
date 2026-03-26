# Wet Woodland Statistics - Output Files Guide

## Overview

The `wet_woodland_stats.py` script produces multiple output files. This guide explains what each file contains and how to use it.

---

## Files for Scientific Papers 📄

### **`wet_woodland_REPORT.txt`** ⭐ **START HERE**
**Purpose**: Human-readable summary report with all statistics formatted for easy reading and inclusion in scientific papers.

**Contains**:
- Study area information
- Total wet woodland area (with on-peat/off-peat breakdown)
- Patch statistics (count, mean, median, modal size)
- Patch size distribution (<5ha, <20ha, >20ha)
- TOW classification (Lone Trees, Group of Trees, Small Woodlands)
- Landscape fragmentation metrics (LDI, Effective Mesh Size)
- Nearest neighbor distances
- Proximity index
- Distance to water features (with percentiles)
- Elevation statistics
- Floodplain association
- **Pre-written paragraph for your Methods/Results section**

**How to use**:
1. Open in any text editor
2. Copy statistics directly into your paper
3. Use the "SUMMARY FOR SCIENTIFIC PAPER" section as a template paragraph
4. Cite specific metrics with proper formatting

**Example citation from report**:
> "The study area contained 12,345.7 ha of wet woodland distributed across 45,823 discrete patches, with a mean patch size of 0.27 ha (median = 0.05 ha)."

---

## Files for Data Analysis 📊

### **`wet_woodland_summary.csv`**
**Purpose**: Single-row CSV with all summary statistics as columns for easy import into R, Python, or Excel.

**Contains**: All metrics from the report in machine-readable format

**Use cases**:
- Import into spreadsheet for graphs
- Combine results from multiple study areas
- Statistical analysis in R/Python

**Example R code**:
```r
stats <- read.csv("wet_woodland_summary.csv")
print(paste("Total area:", stats$total_wet_woodland_ha, "ha"))
```

---

### **`wet_woodland_patch_metrics.csv`**
**Purpose**: One row per patch with detailed metrics for each individual wet woodland patch.

**Columns**:
- `patch_id`: Unique identifier for each patch
- `area_ha`: Area in hectares
- `centroid_x`, `centroid_y`: Geographic coordinates of patch center
- `pixel_count`: Number of pixels in patch
- `nearest_neighbour_m`: Distance to nearest other patch (meters)
- `proximity_index`: Connectivity metric (higher = more connected)
- `proximity_neighbors`: Count of patches within 1km radius
- `area_m2`: Area in square meters
- `height_cover_frac`: Fraction of patch meeting height threshold (if height data provided)
- `height_pass`: Whether patch meets height criteria
- `category_tow`: TOW classification (Lone Tree, Group of Trees, Small Woodland, or None)

**Use cases**:
- Analyze individual patches
- Create histograms of patch sizes
- Map patches by category
- Filter patches by size or proximity

**Example analysis**:
```python
import pandas as pd
patches = pd.read_csv("wet_woodland_patch_metrics.csv")

# Get all large patches
large_patches = patches[patches['area_ha'] > 20]

# Calculate isolation
isolated_patches = patches[patches['nearest_neighbour_m'] > 1000]
```

---

### **`wet_woodland_lnrs_aggregation.csv`** (if LNRS data provided)
**Purpose**: Summary statistics aggregated by LNRS (Local Nature Recovery Strategy) region.

**Columns**:
- `lnrs_id`: Region identifier
- `region_area_ha`: Total area of region
- `wet_area_ha`: Wet woodland area in region
- `wet_density_ha_per_km2`: Density of wet woodland
- `NAME`: Region name (if available in input data)

**Use cases**:
- Compare wet woodland across regions
- Identify priority regions
- Regional reporting

---

## Spatial Files (GeoTIFFs) 🗺️

### **`wet_woodland_patch_labels.tif`**
**Purpose**: Raster where each patch has a unique integer ID matching `patch_id` in the CSV.

**Use in QGIS/ArcGIS**:
1. Load raster
2. Join with `wet_woodland_patch_metrics.csv` using `patch_id`
3. Symbolize by TOW category, size, or connectivity

---

### **`wet_woodland_density_1km_ha_per_km2.tif`**
**Purpose**: 1km resolution grid showing wet woodland density (hectares per square kilometer).

**Use cases**:
- Create density heat maps
- Identify hotspots
- Regional planning

---

### **`wet_woodland_density_10km_ha_per_km2.tif`**
**Purpose**: 10km resolution grid for broader-scale density analysis.

**Use cases**:
- Landscape-scale patterns
- National-scale maps
- Regional comparisons

---

### **`wet_woodland_distance_to_water_m.tif`** (if water data provided)
**Purpose**: Distance in meters from each pixel to nearest water feature.

**Use cases**:
- Analyze hydrological connectivity
- Identify wet woodlands far from mapped water
- Validate predictions (wet woodlands should be near water)

---

## Metadata File 📋

### **`wet_woodland_manifest.json`**
**Purpose**: JSON file listing all output files with their full paths.

**Use cases**:
- Automated workflows
- Verify all outputs were created
- Scripting batch processing

---

## Quick Reference Table

| File Type | For Papers? | For Analysis? | For Mapping? |
|-----------|-------------|---------------|--------------|
| `*_REPORT.txt` | ⭐ YES | No | No |
| `*_summary.csv` | Yes | ⭐ YES | No |
| `*_patch_metrics.csv` | Yes | ⭐ YES | ⭐ YES |
| `*_lnrs_aggregation.csv` | Yes | ⭐ YES | ⭐ YES |
| `*_patch_labels.tif` | No | Yes | ⭐ YES |
| `*_density_1km.tif` | No | Yes | ⭐ YES |
| `*_density_10km.tif` | No | Yes | ⭐ YES |
| `*_distance_to_water_m.tif` | No | Yes | ⭐ YES |

---

## Common Workflows

### Writing a Paper
1. Open `wet_woodland_REPORT.txt`
2. Copy relevant statistics to your Methods/Results section
3. Use the pre-written summary paragraph as a template
4. Create figures from `*_patch_metrics.csv` (histograms, box plots)
5. Make maps from density and patch label rasters

### Regional Analysis
1. Import `wet_woodland_lnrs_aggregation.csv` into spreadsheet
2. Sort by `wet_density_ha_per_km2` to find priority regions
3. Create bar charts comparing regions

### Patch-Level Analysis
1. Load `wet_woodland_patch_metrics.csv`
2. Filter patches by size or connectivity criteria
3. Export filtered patch IDs
4. Use patch IDs to select patches in `*_patch_labels.tif`

### Making Maps
1. Load `wet_woodland_patch_labels.tif` in QGIS
2. Join with `wet_woodland_patch_metrics.csv` on `patch_id`
3. Style by TOW category or size class
4. Add density rasters as background layers

---

## Statistics Glossary

**Landscape Division Index (LDI)**: 0 = one continuous patch, 1 = completely fragmented
**Effective Mesh Size**: Average size if landscape was divided into equal patches
**Proximity Index**: How close a patch is to other patches (weighted by their size)
**Nearest Neighbor Distance**: Distance to closest other patch
**TOW Categories**: Trees Outside Woodland classification by UK Forestry Commission

---

## Questions?

If you need help interpreting any statistic or using these files, check the main documentation or the comments in `wet_woodland_stats.py`.
