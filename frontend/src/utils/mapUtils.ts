import mapboxgl from 'mapbox-gl';

// Define GeoJSON types to match mapbox-gl's expected format
export interface CrimeFeature {
  type: 'Feature';
  properties: {
    intensity?: number;
    residential_burglary_risk?: number;
    lsoa_code?: string;
    incidents?: number;
    trend?: string;
    unit_id?: string;
    patrol_type?: string;
    effectiveness?: number;
  };
  geometry: {
    type: 'Point';
    coordinates: number[];
  };
}

export interface FeatureCollection {
  type: 'FeatureCollection';
  features: CrimeFeature[];
}

// Convert crime data to GeoJSON format
export const crimeDataToGeoJSON = (crimeData: any[]): FeatureCollection => {
  return {
    'type': 'FeatureCollection',
    'features': crimeData.map(point => ({
      'type': 'Feature',
      'properties': {
        'intensity': point.intensity,
        'residential_burglary_risk': point.burglary_risk,
        'lsoa_code': point.lsoa_code
      },
      'geometry': {
        'type': 'Point',
        'coordinates': [point.longitude, point.latitude]
      }
    }))
  };
};

// Convert high risk points to GeoJSON
export const highRiskPointsToGeoJSON = (crimeData: any[]): FeatureCollection => {
  return {
    'type': 'FeatureCollection',
    'features': crimeData.filter(p => p.burglary_risk > 70).map(point => ({
      'type': 'Feature',
      'properties': {
        'residential_burglary_risk': point.burglary_risk,
        'lsoa_code': point.lsoa_code,
        'incidents': point.last_month_incidents,
        'trend': point.trend
      },
      'geometry': {
        'type': 'Point',
        'coordinates': [point.longitude, point.latitude]
      }
    }))
  };
};

// Convert police data to GeoJSON
export const policeDataToGeoJSON = (policeData: any[]): FeatureCollection => {
  return {
    'type': 'FeatureCollection',
    'features': policeData.map(unit => ({
      'type': 'Feature',
      'properties': {
        'unit_id': unit.unit_id,
        'patrol_type': unit.patrol_type,
        'effectiveness': unit.effectiveness_score
      },
      'geometry': {
        'type': 'Point',
        'coordinates': [unit.longitude, unit.latitude]
      }
    }))
  };
};

// Add heatmap layer to map
export const addHeatmapLayer = (map: mapboxgl.Map, geojsonHotspots: FeatureCollection) => {
  map.addSource('crime-hotspots', {
    'type': 'geojson',
    'data': geojsonHotspots
  });
  
  map.addLayer({
    'id': 'crime-heat',
    'type': 'heatmap',
    'source': 'crime-hotspots',
    'paint': {
      'heatmap-weight': ['interpolate', ['linear'], ['get', 'intensity'], 0, 0, 1, 2],
      'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 8, 1, 15, 3],
      'heatmap-color': [
        'interpolate',
        ['linear'],
        ['heatmap-density'],
        0, 'rgba(33,102,172,0)',
        0.2, 'rgba(103,169,207,0.5)',
        0.4, 'rgba(209,229,240,0.6)',
        0.6, 'rgba(253,219,199,0.7)',
        0.8, 'rgba(239,138,98,0.8)',
        1, 'rgba(178,24,43,0.9)'
      ],
      'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 8, 10, 13, 25],
      'heatmap-opacity': ['interpolate', ['linear'], ['zoom'], 10, 1, 17, 0.6]
    }
  });
};

// Add points layer to map
export const addPointsLayer = (map: mapboxgl.Map, geojsonPoints: FeatureCollection) => {
  map.addSource('crime-points', {
    'type': 'geojson',
    'data': geojsonPoints
  });
  
  map.addLayer({
    'id': 'crime-points',
    'type': 'circle',
    'source': 'crime-points',
    'paint': {
      'circle-radius': [
        'interpolate', ['linear'], ['zoom'],
        10, 3,
        15, 8
      ],
      'circle-color': [
        'interpolate', ['linear'], ['get', 'burglary_risk'],
        50, '#f1f5f9',
        70, '#fcd34d',
        90, '#f87171'
      ],
      'circle-stroke-width': 1.5,
      'circle-stroke-color': [
        'interpolate', ['linear'], ['get', 'burglary_risk'],
        50, 'rgba(241, 245, 249, 0.8)',
        70, 'rgba(252, 211, 77, 0.8)',
        90, 'rgba(248, 113, 113, 0.8)'
      ],
      'circle-opacity': 0.8
    }
  });
};

// Add police markers layer to map
export const addPoliceMarkersLayer = (map: mapboxgl.Map, geojsonPolice: FeatureCollection) => {
  map.addSource('police-units', {
    'type': 'geojson',
    'data': geojsonPolice
  });
  
  map.addLayer({
    'id': 'police-markers',
    'type': 'circle',
    'source': 'police-units',
    'paint': {
      'circle-radius': 5,
      'circle-color': [
        'match',
        ['get', 'patrol_type'],
        'Vehicle', '#3b82f6',
        'Foot', '#22c55e',
        '#94a3b8'
      ],
      'circle-stroke-width': 2,
      'circle-stroke-color': [
        'match',
        ['get', 'patrol_type'],
        'Vehicle', 'rgba(59, 130, 246, 0.5)',
        'Foot', 'rgba(34, 197, 94, 0.5)',
        'rgba(148, 163, 184, 0.5)'
      ]
    }
  });
};

// Create popup content for crime points
export const createCrimePointPopupHTML = (properties: any) => {
  return `
    <div style="font-family: system-ui; color: #f8fafc; padding: 4px;">
      <p style="font-size: 12px; margin-bottom: 4px; font-weight: bold; color: #94a3b8;">LSOA: ${properties.lsoa_code}</p>
      <p style="font-size: 14px; margin-bottom: 2px;">Residential Burglary Risk: 
        <span style="color: ${properties.residential_burglary_risk > 80 ? '#f87171' : '#fcd34d'}; font-weight: 600;">
          ${properties.residential_burglary_risk}%
        </span>
      </p>
      <p style="font-size: 13px; margin-bottom: 2px;">
        Last Month: ${properties.incidents} incidents
      </p>
      <p style="font-size: 12px; color: ${properties.trend === 'up' ? '#f87171' : '#4ade80'};">
        Trend: ${properties.trend === 'up' ? '↑ Increasing' : '↓ Decreasing'}
      </p>
    </div>
  `;
};

// Create popup content for police markers
export const createPoliceMarkerPopupHTML = (properties: any) => {
  return `
    <div style="font-family: system-ui; color: #f8fafc; padding: 4px;">
      <p style="font-size: 12px; margin-bottom: 4px; font-weight: bold; color: #94a3b8;">Unit: ${properties.unit_id}</p>
      <p style="font-size: 14px; margin-bottom: 2px;">Type: 
        <span style="color: ${properties.patrol_type === 'Vehicle' ? '#3b82f6' : '#22c55e'}; font-weight: 600;">
          ${properties.patrol_type}
        </span>
      </p>
      <p style="font-size: 13px;">
        Effectiveness: ${properties.effectiveness}%
      </p>
    </div>
  `;
};
