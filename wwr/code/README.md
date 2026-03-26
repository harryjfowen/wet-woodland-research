# WWR Code Index

This directory contains the active wet woodland pipeline code.

## `gee/`

- `gee_train.js`
- `gee_inference.js`
- `gee_terrain.js`

## `labels/`

- `tow_gdb_processor.py`
- `gather_wetwoodland_labels.py`
- `gather_functionaltypes_labels.py`
- `create_peat_binary_mask_raster.py`

## `preprocess/`

- `build_dtm_metrics.py`
- `build_abiotic_stack.py`

## `model/`

- `gpu_xgboost_trainer.py`

## `inference/`

- `gpu_xgboost_predictor.py`
- `gpu_batch_predictor.py`

## `potential/`

- `maxent.py`
- `run_elapid_potential.py`

## `postprocess/`

- `conformal_confidence_from_kml.py`
- `hysteresis_threshold.py`
- `wet_woodland_stats.py`
- `recall_from_kml.py`

Archived or superseded scripts should be moved under `wwr/archive/legacy_code/`.
