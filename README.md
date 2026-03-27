<!--
Optional banner placement:

<p align="center">
  <img src="wwr/assets/images/banner.png" alt="Wet Woodland Research banner" width="100%" />
</p>
-->

# Wet Woodland Research

Wet Woodland Research (WWR) is the active codebase for mapping wet woodland-associated
tree cover and wet woodland restoration opportunity. The current workflow focuses on
England and combines label preparation, Google Earth Engine features, GPU XGBoost
modelling, Elapid/MaxEnt suitability modelling, post-processing, validation, and
publication-oriented visualisation.

## What This Repository Contains

- A 10 m current extent workflow for likely wet woodland-associated tree cover within the
  wider tree-cover domain.
- A national environmental suitability workflow for wet woodland establishment and
  restoration opportunity.
- Post-processing, evaluation, reporting, and figure-generation scripts for analysis and
  communication.
- Archived legacy code kept separately from the active pipeline.

## Repository Layout

```text
wet-woodland-research/
├── README.md
└── wwr/
    ├── code/        # Active pipeline scripts
    ├── data/        # Local inputs, outputs, and validation assets
    ├── docs/        # Runbooks, methods notes, and planning docs
    ├── visualise/   # Publication-style map rendering scripts
    ├── assets/      # Logos and other lightweight assets
    └── archive/     # Legacy code retained outside the active workflow
```

## Web Viewer

An interactive national map is published at [harryjfowen.github.io/wetwoodland-map](https://harryjfowen.github.io/wetwoodland-map), built on deck.gl and geotiff.js. It visualises the current extent probability surface and restoration suitability across England.

### Cloud-Optimised GeoTIFF architecture

The probability raster is served as a Cloud-Optimised GeoTIFF (COG) from Cloudflare R2 object storage. This is the industry-standard approach used by USGS, ESA, and NASA for distributing large raster datasets — known as cloud-native geospatial.

**How it works:**
- The COG stores the raster in 512×512 pixel tiles with a pre-computed overview pyramid (7 levels: 2×–128×). At national scale the browser fetches the 128× overview; zooming into a county fetches full-resolution tiles.
- Cloudflare R2 supports HTTP `Range` requests, so geotiff.js reads the COG's internal index (a few KB), locates the precise byte offsets of the needed tiles, and fetches only those bytes. A single viewport may transfer 50 KB from a 700 MB file.
- Overviews use RMS resampling rather than the default average. For sparse data like wet woodland extent (< 1% of England), RMS preserves the density signal — dense patches appear stronger, sparse patches faint — rather than washing everything to near-zero or near-uniform.
- Pixels are quantized to uint8 (0 = zero probability/transparent, 1–254 = probability × 254, 255 = nodata) with DEFLATE compression. The sparse zero-heavy raster compresses to a fraction of its uncompressed size.
- The viewer source lives at [github.com/harryjfowen/wetwoodland-map](https://github.com/harryjfowen/wetwoodland-map).

## Active Pipeline

The canonical workflow lives under `wwr/code/`:

1. `labels/`
   Build masks and training labels from the TOW GDB and supporting habitat inputs.
2. `preprocess/`
   Build terrain and abiotic predictor layers for modelling.
3. `model/`
   Train the GPU XGBoost current extent model.
4. `inference/`
   Apply trained models across embedding tiles.
5. `postprocess/`
   Mosaic, threshold, validate, and summarize outputs.
6. `potential/`
   Run wet woodland suitability modelling with `maxent.py` as the preferred entrypoint.
7. `visualise/`
   Produce publication-ready national and thematic figures.

Key production scripts include:

- `wwr/code/labels/tow_gdb_processor.py`
- `wwr/code/labels/gather_wetwoodland_labels.py`
- `wwr/code/preprocess/build_dtm_metrics.py`
- `wwr/code/preprocess/build_abiotic_stack.py`
- `wwr/code/model/gpu_xgboost_trainer.py`
- `wwr/code/inference/gpu_xgboost_predictor.py`
- `wwr/code/potential/maxent.py`
- `wwr/code/postprocess/hysteresis_threshold.py`
- `wwr/code/postprocess/wet_woodland_stats.py`
- `wwr/code/postprocess/recall_from_kml.py`

## Getting Started

1. Work from the repository root:

   ```bash
   cd /path/to/wet-woodland-research
   ```

2. Prepare a Python environment with the GIS and modelling stack used by the scripts.
   In practice this includes packages such as `rasterio`, `geopandas`, `fiona`,
   `xgboost`, `optuna`, `scikit-learn`, and `elapid`. GPU training and inference also
   require a compatible CUDA-enabled environment.

3. Populate local inputs under `wwr/data/input/` and `wwr/data/raw/`.
   Large geospatial inputs and generated outputs are intentionally not tracked in git.

4. Inspect stage-specific CLIs before running the workflow:

   ```bash
   python wwr/code/labels/tow_gdb_processor.py --help
   python wwr/code/model/gpu_xgboost_trainer.py --help
   python wwr/code/potential/maxent.py --help
   python wwr/code/postprocess/wet_woodland_stats.py --help
   ```

5. Install the extra figure-rendering dependencies only if you need the visualisation
   scripts:

   ```bash
   pip install -r wwr/visualise/requirements.txt
   ```

## Data and Outputs

The active data tree is rooted at `wwr/data/`.

- Inputs live under `wwr/data/input/`
- Raw upstream assets live under `wwr/data/raw/`
- Validation assets live under `wwr/data/validation/`
- Generated outputs are typically written under `wwr/data/output/`

Common output folders include:

- `wwr/data/output/labels`
- `wwr/data/output/models`
- `wwr/data/output/predictions`
- `wwr/data/output/preprocess`
- `wwr/data/output/potential`
- `wwr/data/output/postprocess`
- `wwr/data/output/reports`

## Documentation

- [Workflow runbook](wwr/docs/RUNBOOK.md)
- [Data layout](wwr/data/README.md)
- [Code index](wwr/code/README.md)
- [Visualisation guide](wwr/visualise/README.md)
- [Wet woodland suitability framework](wwr/docs/methods/WET_WOODLAND_SUITABILITY_FRAMEWORK.md)

## Project Conventions

- Active code should remain under `wwr/code/`.
- Legacy or superseded code should be moved to `wwr/archive/legacy_code/`, not deleted.
- Keep large inputs, rasters, and generated outputs out of version control.
- Treat the top-level README as the GitHub landing page and keep detailed operational
  commands in the linked docs.
