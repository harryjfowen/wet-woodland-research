# Wet Woodland Runbook

These are copy/paste commands for the current cleaned repo layout.

All commands below assume you run from repo root:

```bash
cd /Users/harryjfowen/Software/wet-woodland-research
```

## 1) Pull Embeddings (rclone)

Training embeddings:

```bash
rclone copy gdrive: wwr/data/input/embeddings/training_embeddings \
  --drive-root-folder-id 1KUZKnbDjuWE6WoOFRZhjCwnZ-FJJnQO8 \
  --drive-acknowledge-abuse \
  --progress \
  --transfers 4 \
  --checkers 8
```

Inference embeddings (if stored separately):

```bash
rclone copy gdrive: wwr/data/input/embeddings/inference_embeddings \
  --drive-root-folder-id 1KUZKnbDjuWE6WoOFRZhjCwnZ-FJJnQO8 \
  --drive-acknowledge-abuse \
  --progress \
  --transfers 4 \
  --checkers 8
```

## 2) Build Forest Mask from TOW GDB (vector + raster for Earth Engine)

Vector mask:

```bash
python wwr/code/labels/tow_gdb_processor.py \
  --gdb-dir wwr/data/input/tow_gdb \
  --method identify \
  --output wwr/data/output/labels/forest_mask.gpkg
```

Raster mask:

```bash
python wwr/code/labels/tow_gdb_processor.py \
  --gdb-dir wwr/data/input/tow_gdb \
  --method identify \
  --export-raster \
  --raster-resolution 10 \
  --output wwr/data/output/labels/forest_mask.tif
```

## 3) Build Wet Woodland Labels

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

## 7) Hysteresis Postprocess Mosaic

```bash
python wwr/code/postprocess/hysteresis_threshold.py \
  --input wwr/data/output/predictions/tiles \
  --output wwr/data/output/postprocess/wet_woodland_mosaic_hysteresis.tif
```

## 8) Independent Recall from KML

```bash
python wwr/code/postprocess/recall_from_kml.py \
  --kml wwr/data/validation/wetwoodlands.kml \
  --wet-woodland-raster wwr/data/output/postprocess/wet_woodland_mosaic_hysteresis.tif \
  --outdir wwr/data/output/reports \
  --erode-pixels 1
```

## 9) Elapid Potential

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
