import React, { useRef } from 'react';
import Map, { type MapRef } from 'react-map-gl/maplibre';
import { MapboxOverlay } from '@deck.gl/mapbox';
import { useControl } from 'react-map-gl/maplibre';
import { COGLayer } from '@developmentseed/deck.gl-geotiff';
import { epsgResolver } from '@developmentseed/deck.gl-geotiff';
import 'maplibre-gl/dist/maplibre-gl.css';

const CARTO_DARK =
  'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

// RGBA COG — colours baked in, no JS colormap needed
const COG_URL = './wet_woodland_suitability_cog.tif';

function DeckGLOverlay({ layers }: { layers: any[] }) {
  const overlay = useControl(() => new MapboxOverlay({ interleaved: true, layers }));
  overlay.setProps({ layers });
  return null;
}

export default function App() {
  const mapRef = useRef<MapRef>(null);

  const cogLayer = new COGLayer({
    id: 'wet-woodland-cog',
    geotiff: COG_URL,
    epsgResolver,
    maxError: 0.125,
    onGeoTIFFLoad: (_tiff: any, options: any) => {
      const { west, south, east, north } = options.geographicBounds;
      mapRef.current?.fitBounds([[west, south], [east, north]], {
        padding: 40,
        duration: 1000,
      });
    },
  });

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Map
        ref={mapRef}
        initialViewState={{ longitude: -1.5, latitude: 52.8, zoom: 5.5 }}
        mapStyle={CARTO_DARK}
        style={{ width: '100%', height: '100%' }}
      >
        <DeckGLOverlay layers={[cogLayer]} />
      </Map>

      <div style={{
        position: 'absolute', top: 16, left: 16,
        background: 'rgba(20,20,20,0.85)', color: '#e0e0e0',
        padding: '10px 16px', borderRadius: 6,
        fontFamily: 'system-ui, sans-serif', fontSize: 15,
        fontWeight: 600, letterSpacing: 0.3,
        pointerEvents: 'none', zIndex: 10,
      }}>
        Wet Woodland Suitability
      </div>
    </div>
  );
}
