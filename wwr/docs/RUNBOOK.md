# Wet Woodland Runbook

Copy/paste commands for the current pipeline. Run all commands from repo root:

```bash
cd /Users/harryjfowen/Software/wet-woodland-research
```

## 1) Google Earth Engine — Input Data

Embedding tiles and terrain data are generated via [Google Earth Engine](https://earthengine.google.com/) and are too large to store in this repository. GEE scripts live in `wwr/code/gee/`. Exact Earth Engine assets used:

| Asset | GEE path |
|---|---|
| Google Satellite Embeddings | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` |
| EA 1m LiDAR terrain (DTM/DSM) | `UK/EA/ENGLAND_1M_TERRAIN/2022` |
| Copernicus GLO-30 DEM (gap-fill) | `COPERNICUS/DEM/GLO30` |
| Forest compartment mask | `users/harryjfowen/compartment_duo_mask` |

Run `wwr/code/gee/gee_patch.js` in the Earth Engine code editor to export embedding tiles to Google Drive, then pull locally:

```bash
rclone copy gdrive: wwr/data/input/embeddings/training_embeddings \
  --drive-root-folder-id 1KUZKnbDjuWE6WoOFRZhjCwnZ-FJJnQO8 \
  --drive-acknowledge-abuse --progress --transfers 4 --checkers 8

rclone copy gdrive: wwr/data/input/embeddings/inference_embeddings \
  --drive-root-folder-id 1KUZKnbDjuWE6WoOFRZhjCwnZ-FJJnQO8 \
  --drive-acknowledge-abuse --progress --transfers 4 --checkers 8
```

## 2) Build Forest Mask and Wet Woodland Labels

Training labels are drawn from three sources for comprehensive coverage:

| Source | Link |
|---|---|
| Forestry England Subcompartments 2025 | [data.gov.uk](https://www.data.gov.uk/dataset/b8cb3475-f5d1-4907-b82a-13477fd6cf69/forestry-england-subcompartments2025) |
| National Forest Inventory (NFI) parcels | [Forest Research](https://www.forestresearch.gov.uk/tools-and-resources/national-forest-inventory/) |
| Trees Outside Woodland (TOW) map | [Forest Research](https://www.forestresearch.gov.uk/tools-and-resources/national-forest-inventory/trees-outside-woodland/) |

Build the forest domain mask from the TOW GDB (vector + raster for Earth Engine export):

```bash
python wwr/code/labels/tow_gdb_processor.py \
  --gdb-dir wwr/data/input/tow_gdb \
  --method identify \
  --output wwr/data/output/labels/forest_mask.gpkg

python wwr/code/labels/tow_gdb_processor.py \
  --gdb-dir wwr/data/input/tow_gdb \
  --method identify \
  --export-raster \
  --raster-resolution 10 \
  --output wwr/data/output/labels/forest_mask.tif
```

Build the merged wet woodland label raster (combines all three sources):

```bash
python wwr/code/labels/gather_wetwoodland_labels.py
```

## 4) Preprocess for Elapid (DTM + Abiotics + Predictor Stack)

Stage 1: build `dtm_metrics.tif`

```bash
python wwr/code/preprocess/build_dtm_metrics.py \
  --dtm-dir wwr/data/input/dtm
```

Stage 2: build the abiotic predictor stack

If `wwr/data/output/preprocess/dtm_metrics.tif` is missing, this script now fails
immediately and tells you to run the DTM stage first.

```bash
python wwr/code/preprocess/build_abiotic_stack.py
```

## 5) Train XGBoost

```bash
python wwr/code/model/gpu_xgboost_trainer.py \
  --bg-ratio 1 \
  --gpu 0 \
  --optuna-samples 45000 \
  --trials 100 \
  --save-model wwr/data/output/models/wetwoodland65.json \
  --save-oof-probe-data auto \
  --compute-shap \
  --find-threshold \
  --spatial-buffer 0 \
  --n-folds 10 \
  --exclude-features dtm_elevation
```

Defaults:
- training embeddings: `wwr/data/input/embeddings/training_embeddings/`
- labels: `wwr/data/output/labels/wetwoodland.tif`
- label schema: `split_bg_0123`
- training target: binary wet/non-wet
- validation: spatial CV
- hyperparameter search: Optuna
- thresholding: policy-calibrated hysteresis seed threshold from OOF folds
- threshold policy defaults: `q10 >= 25%` deployment precision, `min recall = 0%`
- models: timestamped names such as `wwr/data/output/models/wetwoodland_binary20260310153045.json`
- OOF hardness probe bundle: `wwr/data/output/models/<model_stem>.embedding_hardness_oof.npz` when `--save-oof-probe-data auto` is used
- discarded-label cache: `wwr/data/validation/eval_background.tif`
- reports: `wwr/data/output/reports/`

## 6) Run GPU Inference

```bash
python wwr/code/inference/gpu_xgboost_predictor.py \
  --workers 2 \
  --gpu-start 0 \
  --exclude-features dtm_elevation
```

Defaults:
- model: auto-detect the newest file in `wwr/data/output/models/`
- input tiles: `wwr/data/input/embeddings/inference_embeddings/`
- outputs: `wwr/data/output/predictions/tiles/`

If you want a specific older model, pass `--model` explicitly.

## 7) Build Embedding Hardness / Reliability Tiles

Requires the OOF probe bundle exported during training (see `--save-oof-probe-data auto`
above). If omitted from training, rerun training once with that flag to create it.

```bash
python wwr/code/postprocess/embedding_hardness_map.py
```

Defaults:
- OOF probe bundle: auto-detect newest `*.embedding_hardness_oof.npz` in `wwr/data/output/models/`
- input tiles: `wwr/data/input/embeddings/inference_embeddings/`
- outputs:
  - hardness: `wwr/data/output/postprocess/embedding_hardness/hardness_tiles/`
  - reliability: `wwr/data/output/postprocess/embedding_hardness/reliability_tiles/`
  - probe/report: `wwr/data/output/postprocess/embedding_hardness/`

## 8) Hysteresis Postprocess Mosaic

```bash
python wwr/code/postprocess/hysteresis_threshold.py \
  --input wwr/data/output/predictions/tiles \
  --output wwr/data/output/postprocess/wet_woodland_mosaic_hysteresis.tif
```

## 9) Independent Recall from KML

```bash
python wwr/code/postprocess/recall_from_kml.py \
  --kml wwr/data/validation/wetwoodlands.kml \
  --wet-woodland-raster wwr/data/output/postprocess/wet_woodland_mosaic_hysteresis.tif \
  --outdir wwr/data/output/reports \
  --erode-pixels 1
```

## 10) Elapid Potential

```bash
python wwr/code/potential/maxent.py
```

Publication-style run:

```bash
python wwr/code/potential/maxent.py \
  --landvalue-shp wwr/data/input/boundaries/agricultural_land_classification.shp \
  --lnrs-shp wwr/data/input/boundaries/lnrs_areas.shp \
  --urban-shp wwr/data/input/boundaries/england_urban.shp \
  --shap \
  --compute-10m
```
