// Training data extraction for wet woodland detection
// Combines Google embeddings + LiDAR features, masked to mature woodland areas

// Load training labels mask
var maskRaster = ee.Image('users/harryjfowen/species_labels_with_peat')
  .rename('mask')
  .unmask(255)
  .eq(255)
  .not()
  .reproject({crs: 'EPSG:27700', scale: 5});

// Define study area
var studyArea = maskRaster.geometry();
print('Study area:', studyArea);
print('Study area size (km²):', studyArea.area().divide(1e6));

var binaryMask = maskRaster.gte(0.5);

// Load Google 64-dimensional embeddings
var englandBounds = ee.Geometry.Rectangle([-6, 49, 2, 56]);
var embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterDate('2022-01-01', '2023-01-01')
  .filterBounds(englandBounds)
  .mosaic()
  .updateMask(binaryMask);

var embeddings_5m = embeddings
  .reproject({crs: 'EPSG:27700', scale: 10})
  .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
  .reproject({crs: 'EPSG:27700', scale: 5});


// Load Environment Agency LiDAR
var dsm = ee.Image('UK/EA/ENGLAND_1M_TERRAIN/2022').select('dsm_first').updateMask(binaryMask);
var dtm = ee.Image('UK/EA/ENGLAND_1M_TERRAIN/2022').select('dtm').updateMask(binaryMask);
var chm = dsm.subtract(dtm).rename('canopy_height');

var dsm_5m = dsm.reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
  .reproject({crs: 'EPSG:27700', scale: 5}).rename('dsm_mean');
var dtm_5m = dtm.reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
  .reproject({crs: 'EPSG:27700', scale: 5}).rename('dtm_mean');
var chm_5m = chm.reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
  .reproject({crs: 'EPSG:27700', scale: 5}).rename('chm_mean');

// Create mature woodland mask (CHM >= 3m)
var matureWoodlandMask = chm_5m.gte(3);
Map.addLayer(matureWoodlandMask, {min: 0, max: 1, palette: ['white', 'darkgreen']}, 'Mature Woodland Mask (CHM >= 3m)');

// Combine all features and apply mature woodland mask
var allFeatures = embeddings_5m
  .addBands(dsm_5m)
  .addBands(dtm_5m) 
  .addBands(chm_5m)
  .updateMask(matureWoodlandMask);

// Export training data
Export.image.toDrive({
  image: allFeatures.toFloat(),
  description: 'wet_woodland_training',
  folder: 'wet_woodland_train',
  region: studyArea,
  scale: 5,
  maxPixels: 1e13,
  crs: 'EPSG:27700',
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true},
  skipEmptyTiles: true
});

// Visualization
var bbox_visualise = ee.Geometry.Rectangle([-1.13, 53.23, -0.96, 53.30]);
Map.centerObject(bbox_visualise, 13);

var embeddingsPreview = embeddings_5m.clip(bbox_visualise).updateMask(matureWoodlandMask);
var chmPreview = chm_5m.clip(bbox_visualise).updateMask(matureWoodlandMask);

Map.addLayer(embeddingsPreview, {min: -0.3, max: 0.3, bands: ['A01','A16','A09']}, 'Embeddings');
Map.addLayer(chmPreview, {min: 0, max: 30, palette: ['blue','green','yellow','red']}, 'CHM');

print('✅ Script complete! Check Tasks tab to start export.');