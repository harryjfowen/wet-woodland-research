# Repository Agent Guide

## Scope

Use this guide when cleaning, reorganizing, or extending code in this repository.

## Canonical code location

- Keep active code under `tow/code/`.
- Keep folders by workflow stage:
  - `tow/code/labels`
  - `tow/code/preprocess`
  - `tow/code/model`
  - `tow/code/inference`
  - `tow/code/potential`
  - `tow/code/postprocess`

## Safety rules for cleanup

- Do not hard-delete scripts that may still be useful.
- Move superseded files to `tow/archive/legacy_code/` with clear names.
- Treat `tow/code/model/gpu_xgboost_trainer.py` and `tow/code/inference/gpu_xgboost_predictor.py` as the current production trainer/predictor pair.
- Prefer keeping latest GPU/XGBoost and Elapid workflows.
- Keep data files and outputs separate from code; avoid adding new scripts into `tow/data` or `tow/data/output`.

## Primary pipeline to preserve

1. GDB and mask prep:
   - `tow/code/labels/tow_gdb_processor.py`
2. Training label generation:
   - `tow/code/labels/gather_wetwoodland_labels.py`
   - `tow/code/labels/gather_functionaltypes_labels.py`
3. Abiotic preprocessing:
   - `tow/code/preprocess/run_potential_pipeline.py`
   - `tow/code/preprocess/preprocess_potential_dtm_tiled.py`
   - `tow/code/preprocess/preprocess_potential.py`
4. GPU model training/inference:
   - `tow/code/model/gpu_xgboost_trainer.py`
   - `tow/code/inference/gpu_xgboost_predictor.py`
5. Elapid potential:
   - `tow/code/potential/run_elapid_potential.py`
6. Post-processing:
   - `tow/code/postprocess/hysteresis_threshold.py`
   - `tow/code/postprocess/wet_woodland_stats.py`
   - `tow/code/postprocess/recall_from_kml.py`

## Expected cleanup behavior

- Consolidate duplicates by selecting one canonical script.
- Archive old variants (`backup`, `test`, deprecated model families) rather than deleting.
- Update README paths if scripts are moved.
- Keep command line interfaces stable where practical (`--help` should still work after moves).
- Put non-core utility scripts under `tow/archive/legacy_code/non_core_scripts/` when minimizing the active code surface.
