// ============================================================================
// WET WOODLAND EARTH ENGINE TERRAIN EXPORT
// ============================================================================
//
// This script supports:
//   - national EA DTM export
//   - RAF Fylingdales terrain gap-fill export
//   - optional export of the EA hole mask itself
//
// Unlike the inference script, this terrain export is not woodland-masked and
// does not use the TOW mask. It exports all terrain pixels inside exportRegion.

// ---------------------------
// 0) CONFIG
// ---------------------------
var CRS = 'EPSG:27700';
var OUTPUT_SCALE_M = 4;

var EXPORT_MODE = 'gap_fill_raf_fylingdales'; // 'gap_fill_raf_fylingdales' | 'full_england'
var FILL_MODE = 'ea_with_global_fallback';    // 'ea_only' | 'ea_with_global_fallback'
var EXPORT_HOLE_MASK = true;

var EA_TERRAIN_ASSET = 'UK/EA/ENGLAND_1M_TERRAIN/2022';
var COPDEM_ASSET = 'COPERNICUS/DEM/GLO30';

var RAF_FYLINGDALES = ee.Geometry.Point([-0.6697, 54.3589]);
var GAP_FILL_BUFFER_M = 12000;
var gapFillBbox = RAF_FYLINGDALES.buffer(GAP_FILL_BUFFER_M).bounds();
var englandBounds = ee.Geometry.Rectangle([-6, 49, 2, 56], null, false);

// ---------------------------
// 1) EXPORT REGION + DIAGNOSTICS
// ---------------------------
var exportRegion = EXPORT_MODE === 'gap_fill_raf_fylingdales'
  ? gapFillBbox
  : englandBounds;

var exportFootprint = ee.Image.constant(1)
  .rename('export_footprint')
  .clip(exportRegion);

var eaDtm = ee.Image(EA_TERRAIN_ASSET).select('dtm');
var eaCoverage = ee.Image.constant(1)
  .rename('ea_terrain_coverage')
  .updateMask(eaDtm.mask())
  .reproject({crs: CRS, scale: OUTPUT_SCALE_M});

var eaMissingTerrain = exportFootprint.unmask(0).eq(1)
  .and(eaCoverage.unmask(0).eq(0))
  .selfMask()
  .clip(exportRegion)
  .rename('ea_missing_terrain');

print('Export mode:', EXPORT_MODE);
print('Fill mode:', FILL_MODE);
print('Export region (km^2):', exportRegion.area().divide(1e6));
print('EA hole area in export region (ha):',
  ee.Image.pixelArea().divide(1e4).updateMask(eaMissingTerrain).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: exportRegion,
    scale: OUTPUT_SCALE_M,
    maxPixels: 1e12
  }).get('area'));

// ---------------------------
// 2) TERRAIN SURFACES
// ---------------------------
var eaDtm4m = eaDtm
  .reduceResolution({
    reducer: ee.Reducer.mean(),
    maxPixels: 1024
  })
  .reproject({crs: CRS, scale: OUTPUT_SCALE_M})
  .toFloat()
  .rename('dtm_4m_m');

var globalDem30m = ee.ImageCollection(COPDEM_ASSET)
  .select('DEM')
  .filterBounds(exportRegion)
  .mosaic()
  .rename('dem_30m');

var globalDem4m = globalDem30m
  .resample('bilinear')
  .reproject({crs: CRS, scale: OUTPUT_SCALE_M})
  .toFloat()
  .rename('dtm_4m_m');

var terrainExport = FILL_MODE === 'ea_only'
  ? eaDtm4m
  : eaDtm4m.unmask(globalDem4m).rename('dtm_4m_m');

// ---------------------------
// 3) EXPORTS
// ---------------------------
Export.image.toDrive({
  image: terrainExport.clip(exportRegion),
  description: EXPORT_MODE === 'gap_fill_raf_fylingdales'
    ? 'terrain_gapfill_raf_fylingdales'
    : 'England_DTM_2022_4m_float32',
  folder: EXPORT_MODE === 'gap_fill_raf_fylingdales'
    ? 'wet_woodland_terrain_gap_fill'
    : 'EA_DTM_2022',
  fileNamePrefix: EXPORT_MODE === 'gap_fill_raf_fylingdales'
    ? 'terrain_gapfill_raf_fylingdales'
    : 'DTM_2022_4m_m',
  region: exportRegion,
  scale: OUTPUT_SCALE_M,
  crs: CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true,
    noData: -9999
  },
  skipEmptyTiles: false
});

if (EXPORT_HOLE_MASK) {
  Export.image.toDrive({
    image: eaMissingTerrain.unmask(0).toByte().rename('ea_missing_terrain'),
    description: EXPORT_MODE === 'gap_fill_raf_fylingdales'
      ? 'terrain_hole_mask_raf_fylingdales'
      : 'terrain_hole_mask_england',
    folder: EXPORT_MODE === 'gap_fill_raf_fylingdales'
      ? 'wet_woodland_terrain_gap_fill'
      : 'EA_DTM_2022',
    fileNamePrefix: EXPORT_MODE === 'gap_fill_raf_fylingdales'
      ? 'terrain_hole_mask_raf_fylingdales'
      : 'terrain_hole_mask_england',
    region: exportRegion,
    scale: OUTPUT_SCALE_M,
    crs: CRS,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF',
    formatOptions: {
      cloudOptimized: true,
      noData: 0
    },
    skipEmptyTiles: false
  });
}

// ---------------------------
// 4) VISUALIZATION
// ---------------------------
Map.centerObject(exportRegion, EXPORT_MODE === 'gap_fill_raf_fylingdales' ? 11 : 7);

Map.addLayer(
  exportFootprint,
  {min: 0, max: 1, palette: ['white', 'black']},
  'Export footprint'
);

Map.addLayer(
  eaMissingTerrain,
  {palette: ['ff0000']},
  'EA terrain hole'
);

Map.addLayer(
  terrainExport.clip(exportRegion),
  {min: 0, max: 500, palette: ['000000', '444444', '888888', 'cccccc', 'ffffff']},
  'Terrain export'
);

Map.addLayer(
  globalDem4m.clip(exportRegion),
  {min: 0, max: 500, palette: ['081d58', '225ea8', '41b6c4', 'a1dab4', 'ffffcc']},
  'Global fallback DEM'
);

print('Script ready. Start the export(s) from the Tasks tab.');
print('Hole mask export enabled:', EXPORT_HOLE_MASK);
