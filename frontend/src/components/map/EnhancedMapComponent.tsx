import React, { useEffect, useState, useRef, useCallback } from 'react';
import { MapContainer, TileLayer, GeoJSON, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { api } from '@/api/api';

// Fix for Leaflet default icon issue
// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

// London center coordinates
const LONDON_CENTER: [number, number] = [51.5074, -0.1278];
const LONDON_ZOOM = 10;

// Enhanced interfaces for real London data
interface LSOAProperties {
  'LSOA code': string;
  LSOA11NM?: string;
  burglary_count: number;
  risk_level: string;
  Borough?: string;
}

interface BoroughProperties {
  Borough: string;
  burglary_count: number;
  risk_level: string;
}

interface LSOAFeature {
  type: 'Feature';
  properties: LSOAProperties;
  geometry: any;
}

interface BoroughFeature {
  type: 'Feature';
  properties: BoroughProperties;
  geometry: any;
}

interface LSOAGeoJSON {
  type: 'FeatureCollection';
  features: LSOAFeature[];
}

interface BoroughGeoJSON {
  type: 'FeatureCollection';
  features: BoroughFeature[];
}

interface EnhancedMapProps {
  onLSOASelect?: (lsoa: string) => void;
  onBoroughSelect?: (borough: string) => void;
  selectedLSOA?: string | null;
  selectedBorough?: string | null;
  showPoliceAllocation?: boolean;
  mapLevel?: 'lsoa' | 'borough';
}

// Enhanced color scheme matching London crime risk levels
const getRiskColor = (risk_level: string) => {
  switch (risk_level) {
    case 'Very Low':
      return '#22c55e'; // Green
    case 'Low':
      return '#84cc16'; // Light green
    case 'Medium':
      return '#eab308'; // Yellow
    case 'High':
      return '#f97316'; // Orange
    case 'Very High':
      return '#ef4444'; // Red
    default:
      return '#94a3b8'; // Gray
  }
};

const getFillOpacity = (risk_level: string) => {
  switch (risk_level) {
    case 'Very High': return 0.8;
    case 'High': return 0.7;
    case 'Medium': return 0.6;
    case 'Low': return 0.5;
    case 'Very Low': return 0.4;
    default: return 0.3;
  }
};

// Map bounds component to focus on London
const MapBounds = () => {
  const map = useMap();
  
  useEffect(() => {
    map.setView(LONDON_CENTER, LONDON_ZOOM);
    
    // Set max bounds to London area
    const londonBounds = L.latLngBounds(
      L.latLng(51.28, -0.51), // Southwest
      L.latLng(51.69, 0.34)   // Northeast
    );
    map.setMaxBounds(londonBounds);
  }, [map]);
  
  return null;
};

const EnhancedMapComponent: React.FC<EnhancedMapProps> = ({ 
  onLSOASelect,
  onBoroughSelect,
  selectedLSOA,
  selectedBorough,
  showPoliceAllocation = false,
  mapLevel = 'lsoa'
}) => {
  const [lsoaData, setLsoaData] = useState<LSOAGeoJSON | null>(null);
  const [boroughData, setBoroughData] = useState<BoroughGeoJSON | null>(null);
  const [currentLevel, setCurrentLevel] = useState<'lsoa' | 'borough'>(mapLevel);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const geoJsonLayerRef = useRef<L.GeoJSON | null>(null);

  // Update level when prop changes
  useEffect(() => {
    setCurrentLevel(mapLevel);
  }, [mapLevel]);

  // Fetch boundary data based on current level
  useEffect(() => {
    const fetchBoundaryData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        console.log(`Fetching ${currentLevel} level boundaries...`);
        
        if (currentLevel === 'borough') {
          const response = await api.lsoa.getBoroughBoundaries();
          if (response && response.type === 'FeatureCollection') {
            setBoroughData(response as BoroughGeoJSON);
            console.log(`Successfully loaded ${response.features?.length || 0} borough boundaries`);
          } else {
            throw new Error('Invalid borough boundary data received');
          }
        } else {
          const response = await api.lsoa.getBoundaries();
          if (response && response.type === 'FeatureCollection') {
            setLsoaData(response as LSOAGeoJSON);
            console.log(`Successfully loaded ${response.features?.length || 0} LSOA boundaries`);
          } else {
            throw new Error('Invalid LSOA boundary data received');
          }
        }
      } catch (err: any) {
        console.error(`Error fetching ${currentLevel} boundaries:`, err);
        setError(err.message || `Failed to fetch ${currentLevel} boundaries`);
      }
      
      setLoading(false);
    };

    fetchBoundaryData();
  }, [currentLevel]);

  // Style function for boundary polygons
  const getBoundaryStyle = useCallback((feature: any): L.PathOptions => {
    const riskLevel = feature.properties.risk_level || 'Unknown';
    const isSelected = currentLevel === 'lsoa' 
      ? feature.properties['LSOA code'] === selectedLSOA
      : feature.properties.Borough === selectedBorough;

    return {
      fillColor: getRiskColor(riskLevel),
      weight: isSelected ? 3 : 1,
      opacity: 1,
      color: isSelected ? '#ffffff' : '#374151',
      dashArray: isSelected ? '5, 5' : undefined,
      fillOpacity: getFillOpacity(riskLevel),
    };
  }, [selectedLSOA, selectedBorough, currentLevel]);

  // Event handlers for boundary interactions
  const onEachFeature = useCallback((feature: any, layer: L.Layer) => {
    const props = feature.properties;
    
    // Create popup content based on level
    const popupContent = currentLevel === 'lsoa' ? 
      `<div style="color: #1f2937; padding: 8px;">
        <h3 style="margin: 0 0 8px 0; font-weight: bold;">${props.LSOA11NM || props['LSOA code']}</h3>
        <p style="margin: 4px 0;"><strong>LSOA Code:</strong> ${props['LSOA code']}</p>
        <p style="margin: 4px 0;"><strong>Burglary Count:</strong> ${props.burglary_count || 0}</p>
        <p style="margin: 4px 0;"><strong>Risk Level:</strong> ${props.risk_level || 'Unknown'}</p>
      </div>` :
      `<div style="color: #1f2937; padding: 8px;">
        <h3 style="margin: 0 0 8px 0; font-weight: bold;">${props.Borough}</h3>
        <p style="margin: 4px 0;"><strong>Total Burglaries:</strong> ${props.burglary_count || 0}</p>
        <p style="margin: 4px 0;"><strong>Risk Level:</strong> ${props.risk_level || 'Unknown'}</p>
      </div>`;

    layer.bindPopup(popupContent);

    // Handle clicks
    layer.on({
      click: () => {
        if (currentLevel === 'lsoa' && onLSOASelect) {
          onLSOASelect(props['LSOA code']);
        } else if (currentLevel === 'borough' && onBoroughSelect) {
          onBoroughSelect(props.Borough);
        }
      },
      mouseover: (e: any) => {
        const layer = e.target;
        layer.setStyle({
          weight: 3,
          color: '#ffffff',
          dashArray: '',
          fillOpacity: 0.9
        });
      },
      mouseout: (e: any) => {
        geoJsonLayerRef.current?.resetStyle(e.target);
      }
    });
  }, [currentLevel, onLSOASelect, onBoroughSelect]);

  // Get current data based on level
  const currentData = currentLevel === 'borough' ? boroughData : lsoaData;

  return (
    <div className="relative h-full w-full">
      {/* Level Toggle Control */}
      <div className="absolute top-4 right-4 z-[1000] bg-white rounded-lg shadow-lg p-2">
        <div className="flex space-x-2">
          <button
            onClick={() => setCurrentLevel('lsoa')}
            className={`px-3 py-1 text-xs rounded ${
              currentLevel === 'lsoa' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            LSOA View
          </button>
          <button
            onClick={() => setCurrentLevel('borough')}
            className={`px-3 py-1 text-xs rounded ${
              currentLevel === 'borough' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Borough View
          </button>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="absolute inset-0 z-[1000] bg-black bg-opacity-50 flex items-center justify-center">
          <div className="bg-white p-4 rounded-lg shadow-lg">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-2 text-gray-700">Loading {currentLevel} boundaries...</p>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="absolute top-16 right-4 z-[1000] bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded shadow-lg max-w-sm">
          <p className="font-bold">Error:</p>
          <p className="text-sm">{error}</p>
        </div>
      )}

      {/* Risk Level Legend */}
      <div className="absolute bottom-4 left-4 z-[1000] bg-white rounded-lg shadow-lg p-4">
        <h4 className="font-bold text-gray-800 mb-2">Burglary Risk Level</h4>
        <div className="space-y-1">
          {['Very Low', 'Low', 'Medium', 'High', 'Very High'].map((level) => (
            <div key={level} className="flex items-center space-x-2">
              <div 
                className="w-4 h-4 rounded"
                style={{ 
                  backgroundColor: getRiskColor(level),
                  opacity: getFillOpacity(level)
                }}
              />
              <span className="text-sm text-gray-700">{level}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Map Container */}
      <MapContainer
        center={LONDON_CENTER}
        zoom={LONDON_ZOOM}
        style={{ height: '100%', width: '100%' }}
        className="z-0"
      >
        <MapBounds />
        
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Boundary Layer */}
        {currentData && (
          <GeoJSON
            ref={geoJsonLayerRef}
            data={currentData}
            style={getBoundaryStyle}
            onEachFeature={onEachFeature}
            key={`${currentLevel}-${currentData.features.length}`}
          />
        )}
      </MapContainer>
    </div>
  );
};

export default EnhancedMapComponent; 