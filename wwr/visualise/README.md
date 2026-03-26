# Visualise

Static publication-map scripts for this repo. Run from repo root.

Default inputs now point to the canonical pipeline outputs in `wwr/data/...`:
- predictions: `wwr/data/output/postprocess/wet_woodland_mosaic_hysteresis.tif`
- suitability: `wwr/data/output/potential/maxent/wet_woodland_potential.tif`
- terrain: `wwr/data/output/potential/potential_predictors_100m.tif`
- ALC polygons: `wwr/data/input/boundaries/agricultural_land_classification.shp`
- boundary: `wwr/data/input/boundaries/england.shp`

Default outputs go to `wwr/visualise/output/`.

These scripts assume the upstream outputs already exist locally, especially:
- `wwr/data/output/potential/maxent/wet_woodland_potential.tif`
- `wwr/data/output/potential/potential_predictors_100m.tif`

## Setup

```bash
pip install -r wwr/visualise/requirements.txt
```

## 1. Suitability map

```bash
python wwr/visualise/render_rasters.py \
  --mode direct \
  --size 3500 \
  --max-size 4000 \
  --hillshade-strength 0.95 \
  --hillshade-clip-low 1 \
  --hillshade-clip-high 99 \
  --hillshade-gamma 0.85 \
  --value-gamma 1.1 \
  --overlay-alpha 0.9 \
  --thematic-alpha-min 0.18 \
  --thematic-alpha-gamma 0.75
```

Output: `wwr/visualise/output/suitability.png`

## 2. Wet woodland extent map

```bash
python wwr/visualise/render_rasters.py \
  --mode hexbin \
  --size 3500 \
  --max-size 4000 \
  --density-cell-m 1000 \
  --density-normalize-max \
  --predictions-value-mask-below 0.005 \
  --predictions-value-gamma 0.75 \
  --suitability-value-gamma 1.1 \
  --hillshade-strength 0.95 \
  --hillshade-clip-low 1 \
  --hillshade-clip-high 99 \
  --hillshade-gamma 0.85 \
  --shade-how eq_hist \
  --value-gamma 1.1 \
  --overlay-alpha 0.9 \
  --thematic-alpha-min 0.18 \
  --thematic-alpha-gamma 0.75
```

Output: `wwr/visualise/output/predictions.png`

## 3. National predictions panel

```bash
python wwr/visualise/render_predictions_panel.py
```

Output: `wwr/visualise/output/predictions_panel.png`

This uses the hysteresis binary output, averaged to a coarser national grid
(default `250 m`) for publication-scale legibility, with the same city-label
style as the central ALC panel.

## 4. ALC suitability panels

```bash
python wwr/visualise/render_alc_suitability_panels.py
```

Output: `wwr/visualise/output/alc_suitability_panels.png`

Add terrain only if you want it:

```bash
python wwr/visualise/render_alc_suitability_panels.py --include-terrain
```

## 5. Bivariate suitability × land value

```bash
python wwr/visualise/render_bivariate_suitability_landvalue.py
```

Output: `wwr/visualise/output/bivariate_suitability_landvalue.png`
