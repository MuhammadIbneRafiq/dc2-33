import React, { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import { MapContainer, TileLayer, GeoJSON, Marker, Popup, CircleMarker, LayersControl, FeatureGroup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { api } from '@/api/api';
import MapLegend from './MapLegend';

// Fix for Leaflet default icon issue in React
// @ts-ignore - Leaflet has type issues with icon URLs
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

// Add custom popup styling to ensure popups display correctly
const customPopupStyle = `
.leaflet-popup {
  z-index: 1000;
  position: absolute;
}
.leaflet-popup-content-wrapper {
  background: rgba(30, 41, 59, 0.9);
  color: white;
  border-radius: 8px;
  padding: 0;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}
.leaflet-popup-content {
  margin: 12px;
  line-height: 1.5;
}
.leaflet-popup-tip {
  background: rgba(30, 41, 59, 0.9);
}
`;

// Helper component to set map style
const MapStyleLayer = () => {
  const map = useMap();
  
  useEffect(() => {
    // Apply styling to the map
    map.getContainer().style.background = '#f8fafc'; // Light background
    
    // Make panning more smooth
    map.on('zoomend', () => {
      map.getPanes().tilePane.classList.add('smooth-tiles');
    });
    
    // Add custom popup styles
    const styleElement = document.createElement('style');
    styleElement.textContent = customPopupStyle;
    document.head.appendChild(styleElement);
    
    return () => {
      map.off('zoomend');
      document.head.removeChild(styleElement);
    };
  }, [map]);
  
  return null;
};

// Define interface for LSOA GeoJSON properties
interface LSOAProperties {
  lsoa_code: string;
  lsoa_name: string;
  burglary_count: number;
  risk_level: string;
}

// Define types for the LSOA boundaries
interface LSOAFeature {
  type: 'Feature';
  properties: LSOAProperties;
  geometry: any;
}

interface LSOAGeoJSON {
  type: 'FeatureCollection';
  features: LSOAFeature[];
}

// Risk level color mapping
const getRiskColor = (risk_level: string) => {
  switch (risk_level) {
    case 'Very Low':
      return '#4ade80'; // Green
    case 'Low':
      return '#a3e635'; // Light green
    case 'Medium':
      return '#fcd34d'; // Yellow
    case 'High':
      return '#fb923c'; // Orange
    case 'Very High':
      return '#f87171'; // Red
    default:
      return '#94a3b8'; // Gray for unknown
  }
};

// Helper component to handle zoom-dependent styling - simplify to avoid TypeScript errors
const ZoomDependentMarkers = ({ children }: { children: React.ReactNode }) => {
  const map = useMap();
  const [zoom, setZoom] = useState<number>(map.getZoom());
  
  useEffect(() => {
    const updateZoom = () => {
      setZoom(map.getZoom());
    };
    
    map.on('zoomend', updateZoom);
    
    return () => {
      map.off('zoomend', updateZoom);
    };
  }, [map]);
  
  // Return the current zoom as a context value that children can use
  return (
    <div className="zoom-dependent-markers" data-zoom={zoom}>
      {/* Pass zoom as a data attribute that can be accessed by marker components */}
      {children}
    </div>
  );
};

interface DynamicMarkerProps {
  position: [number, number];
  patrolType: 'officer' | 'vehicle';
  zoomLevel?: number; // Changed from currentZoom to zoomLevel for clarity
  children?: React.ReactNode;
}

// Use ZoomAwareMarker instead to handle zoom changes
const ZoomAwareMarker = ({ position, patrolType, children }: Omit<DynamicMarkerProps, 'zoomLevel'>) => {
  const map = useMap();
  const [zoom, setZoom] = useState(map.getZoom());
  
  useEffect(() => {
    const updateZoom = () => {
      setZoom(map.getZoom());
    };
    
    map.on('zoomend', updateZoom);
    return () => {
      map.off('zoomend', updateZoom);
    };
  }, [map]);
  
  // Create icon based on current zoom level
  const icon = patrolType === 'officer' ? 
    createPoliceIcon(zoom) : 
    createVehicleIcon(zoom);
  
  return (
    <Marker position={position} icon={icon}>
      {children}
    </Marker>
  );
};

// Keep DynamicMarker for compatibility with existing code
const DynamicMarker = ({ position, patrolType, children }: Omit<DynamicMarkerProps, 'zoomLevel'>) => {
  return <ZoomAwareMarker position={position} patrolType={patrolType}>{children}</ZoomAwareMarker>;
};

// Custom icon creation for police officers
const createPoliceIcon = (zoom = 11) => {
  const iconSize = getZoomDependentSize(20, zoom);
  return L.divIcon({
    html: `<div style="background-color: #1e40af; width: ${iconSize}px; height: ${iconSize}px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; border: 2px solid white; box-shadow: 0 2px 5px rgba(0,0,0,0.3);">👮</div>`,
    className: 'police-icon',
    iconSize: [iconSize, iconSize],
    iconAnchor: [iconSize/2, iconSize/2],
  });
};

// Custom icon creation for police vehicles
const createVehicleIcon = (zoom = 11) => {
  const iconSize = getZoomDependentSize(24, zoom);
  return L.divIcon({
    html: `<div style="background-color: #0369a1; width: ${iconSize}px; height: ${iconSize}px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; border: 2px solid white; box-shadow: 0 2px 5px rgba(0,0,0,0.3);">🚓</div>`,
    className: 'vehicle-icon',
    iconSize: [iconSize, iconSize],
    iconAnchor: [iconSize/2, iconSize/2],
  });
};

// Function to calculate marker size based on zoom level
const getZoomDependentSize = (baseSize: number, zoom: number): number => {
  // Enhanced scaling factor with more aggressive scaling at higher zoom levels
  // Start scaling from zoom level 11 onwards
  if (zoom <= 11) return baseSize;
  if (zoom <= 13) return baseSize * (1 + (zoom - 11) * 0.25);
  if (zoom <= 15) return baseSize * (1.5 + (zoom - 13) * 0.4);
  return baseSize * (2.3 + (zoom - 15) * 0.5); // More dramatic scaling at highest zoom levels
};

interface MapComponentProps {
  onLSOASelect?: (lsoa: string) => void;
  showPoliceAllocation?: boolean;
  selectedLSOA?: string | null;
  policeAllocationData?: { clusters: any[] } | null;
  showPredictions?: boolean;
  predictionModel?: string;
  predictionRange?: number;
  dateRange?: number[]; // Add dateRange prop to connect to header slider
}

// Create a custom function to sanitize GeoJSON before rendering
const sanitizeGeoJSON = (data: any): any => {
  if (!data) return null;
  
  try {
    // Handle if data is a string
    const geojson = typeof data === 'string' ? JSON.parse(data) : data;
    
    // Check if it's a valid GeoJSON structure
    if (!geojson.features || !Array.isArray(geojson.features)) {
      console.error('Invalid GeoJSON structure:', geojson);
      return null;
    }
    
    // Filter out features with invalid geometries
    const validFeatures = geojson.features.filter((feature: any) => {
      // Check if feature has geometry and coordinates
      if (!feature.geometry || !feature.geometry.coordinates) return false;
      
      const coords = feature.geometry.coordinates;
      
      // For Polygons or MultiPolygons, validate the coordinates
      if (feature.geometry.type === 'Polygon' || feature.geometry.type === 'MultiPolygon') {
        // Validate coordinates to ensure they're valid lat/lng pairs
        try {
          // For Polygon
          if (feature.geometry.type === 'Polygon') {
            for (const ring of coords) {
              for (const point of ring) {
                // Check if we have a valid longitude and latitude
                if (point.length !== 2 || 
                    !isFinite(point[0]) || 
                    !isFinite(point[1]) ||
                    Math.abs(point[1]) > 90 || // latitude should be between -90 and 90
                    Math.abs(point[0]) > 180) {
                  return false;
                }
              }
            }
          } 
          // For MultiPolygon
          else if (feature.geometry.type === 'MultiPolygon') {
            for (const polygon of coords) {
              for (const ring of polygon) {
                for (const point of ring) {
                  if (point.length !== 2 || 
                      !isFinite(point[0]) || 
                      !isFinite(point[1]) ||
                      Math.abs(point[1]) > 90 || 
                      Math.abs(point[0]) > 180) {
                    return false;
                  }
                }
              }
            }
          }
          return true;
        } catch (error) {
          console.error('Error validating coordinates:', error);
          return false;
        }
      }
      
      return true;
    });
    
    // If we have no valid features, return null
    if (validFeatures.length === 0) {
      console.error('No valid features found in GeoJSON');
      return null;
    }
    
    // Return the sanitized GeoJSON
    return {
      ...geojson,
      features: validFeatures
    };
  } catch (error) {
    console.error('Error sanitizing GeoJSON:', error);
    return null;
  }
};

const MapComponent = ({ 
  onLSOASelect, 
  showPoliceAllocation = false, 
  selectedLSOA = null,
  policeAllocationData = null,
  showPredictions = false,
  predictionModel = 'lstm-gcn',
  predictionRange = 60,
  dateRange = [30] // Default to 30 days range
}: MapComponentProps) => {
  const [lsoaData, setLsoaData] = useState<LSOAGeoJSON | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const geoJsonLayerRef = useRef<L.GeoJSON | null>(null);
  const [predictionMarkers, setPredictionMarkers] = useState<Array<{lat: number, lon: number, risk: string}>>([]);
  const [historicalData, setHistoricalData] = useState<any>(null);
  const [showHistorical, setShowHistorical] = useState<boolean>(true);

  // Random predictions generator
  const generateRandomPredictions = useCallback(() => {
    // Clear existing predictions
    const newMarkers = [];
    // London coordinates boundaries
    const londonBounds = {
      minLat: 51.28, maxLat: 51.69,
      minLon: -0.51, maxLon: 0.34
    };
    
    // Generate between 30-50 random markers
    const numMarkers = Math.floor(Math.random() * 20) + 30;
    
    for (let i = 0; i < numMarkers; i++) {
      // Random coordinates within London
      const lat = londonBounds.minLat + (Math.random() * (londonBounds.maxLat - londonBounds.minLat));
      const lon = londonBounds.minLon + (Math.random() * (londonBounds.maxLon - londonBounds.minLon));
      
      // Random risk level
      const riskLevels = ['High', 'Medium', 'Low'];
      const riskProbabilities = [0.2, 0.5, 0.3]; // 20% high, 50% medium, 30% low
      
      // Weighted random selection
      const rand = Math.random();
      let cumulativeProbability = 0;
      let riskIndex = 0;
      
      for (let j = 0; j < riskProbabilities.length; j++) {
        cumulativeProbability += riskProbabilities[j];
        if (rand <= cumulativeProbability) {
          riskIndex = j;
          break;
        }
      }
      
      newMarkers.push({
        lat,
        lon,
        risk: riskLevels[riskIndex]
      });
    }
    
    setPredictionMarkers(newMarkers);
  }, []);
  
  // Use static mock data instead of fetching
  useEffect(() => {
    // Create simple mock GeoJSON for demo
    const mockLsoaData: LSOAGeoJSON = {
      type: 'FeatureCollection',
      features: [
        // Central London area
        {
          type: 'Feature',
          properties: {
            lsoa_code: 'E01000001',
            lsoa_name: 'City of London 001',
            burglary_count: 35,
            risk_level: 'High'
          },
          geometry: {
            type: 'Polygon',
            coordinates: [[
              [-0.1, 51.51],
              [-0.08, 51.51],
              [-0.08, 51.53],
              [-0.1, 51.53],
              [-0.1, 51.51]
            ]]
          }
        },
        // Westminster area
        {
          type: 'Feature',
          properties: {
            lsoa_code: 'E01000002',
            lsoa_name: 'Westminster 001',
            burglary_count: 28,
            risk_level: 'Medium'
          },
          geometry: {
            type: 'Polygon',
            coordinates: [[
              [-0.13, 51.50],
              [-0.11, 51.50],
              [-0.11, 51.52],
              [-0.13, 51.52],
              [-0.13, 51.50]
            ]]
          }
        },
        // Camden area
        {
          type: 'Feature',
          properties: {
            lsoa_code: 'E01000003',
            lsoa_name: 'Camden 001',
            burglary_count: 15,
            risk_level: 'Low'
          },
          geometry: {
            type: 'Polygon',
            coordinates: [[
              [-0.14, 51.53],
              [-0.12, 51.53],
              [-0.12, 51.55],
              [-0.14, 51.55],
              [-0.14, 51.53]
            ]]
          }
        },
        // Hackney area
        {
          type: 'Feature',
          properties: {
            lsoa_code: 'E01000004',
            lsoa_name: 'Hackney 001',
            burglary_count: 42,
            risk_level: 'Very High'
          },
          geometry: {
            type: 'Polygon',
            coordinates: [[
              [-0.07, 51.53],
              [-0.05, 51.53],
              [-0.05, 51.55],
              [-0.07, 51.55],
              [-0.07, 51.53]
            ]]
          }
        },
        // Southwark area
        {
          type: 'Feature',
          properties: {
            lsoa_code: 'E01000005',
            lsoa_name: 'Southwark 001',
            burglary_count: 8,
            risk_level: 'Very Low'
          },
          geometry: {
            type: 'Polygon',
            coordinates: [[
              [-0.09, 51.48],
              [-0.07, 51.48],
              [-0.07, 51.50],
              [-0.09, 51.50],
              [-0.09, 51.48]
            ]]
          }
        }
      ]
    };
    
    // Set mock data and finish loading
    setLsoaData(mockLsoaData);
    setLoading(false);
    
  }, []);
  
  // Generate predictions when showPredictions changes
  useEffect(() => {
    if (showPredictions) {
      generateRandomPredictions();
    } else {
      // Clear predictions when showPredictions is false
      setPredictionMarkers([]);
    }
  }, [showPredictions, generateRandomPredictions, predictionModel, predictionRange]);
  
  // Load historical data when component mounts or dateRange changes
  useEffect(() => {
    const loadHistoricalData = async () => {
      try {
        // Call API to load historical data based on dateRange
        const response = await api.get(`/api/burglary/time-series?days=${dateRange[0]}`);
        if (response.data) {
          setHistoricalData(response.data);
        }
      } catch (error) {
        console.error("Error loading historical data:", error);
      }
    };
    
    loadHistoricalData();
  }, [dateRange]);

  // Use historical data by default before any prediction
  useEffect(() => {
    // Set historical mode as default
    if (!showPredictions && historicalData) {
      setShowHistorical(true);
    } else {
      setShowHistorical(false);
    }
  }, [showPredictions, historicalData]);
  
  // Style function for LSOA polygons - memoized to prevent unnecessary recalculations
  const getAreaStyle = useCallback((feature: any): L.PathOptions => {
    const properties = feature.properties || {};
    const isSelected = selectedLSOA === properties.lsoa_code;
    const riskLevel = properties.risk_level || 'Unknown';
    const fillColor = getRiskColor(riskLevel);
    
    return {
      fillColor,
      weight: isSelected ? 3 : 1,
      opacity: 1,
      color: isSelected ? '#1e40af' : '#6b7280',
      dashArray: isSelected ? '' : '3',
      fillOpacity: isSelected ? 0.7 : 0.5
    };
  }, [selectedLSOA]);
  
  // Handle interaction with each LSOA area - memoized
  const onEachFeature = useCallback((feature: any, layer: L.Layer) => {
    const props = feature.properties;
    
    // Create popup content
    const popupContent = `
      <div>
        <h3 style="font-weight: bold; margin-bottom: 8px;">${props.lsoa_name || 'Unknown Area'}</h3>
        <p style="margin: 4px 0;">LSOA Code: ${props.lsoa_code || 'N/A'}</p>
        <p style="margin: 4px 0;">Burglary Count: ${props.burglary_count || 0}</p>
        <p style="margin: 4px 0;">Risk Level: ${props.risk_level || 'Unknown'}</p>
      </div>
    `;
    
    // Add popup
    layer.bindPopup(popupContent);
    
    // Add hover effect
    layer.on({
      mouseover: (e) => {
        const target = e.target;
        target.setStyle({
          weight: 3,
          color: '#1e40af',
          dashArray: '',
          fillOpacity: 0.7
        });
        target.bringToFront();
      },
      mouseout: (e) => {
        // Reset style unless this is the selected LSOA
        if (selectedLSOA !== props.lsoa_code) {
          (e.target as L.Path).setStyle(getAreaStyle(feature));
        }
      },
      click: () => {
        if (onLSOASelect) {
          onLSOASelect(props.lsoa_code);
        }
      }
    });
  }, [selectedLSOA, getAreaStyle, onLSOASelect]);

  // Options for the GeoJSON layer - improves performance by using chunks to render large datasets
  const geoJsonOptions = useMemo(() => ({
    style: getAreaStyle,
    onEachFeature: onEachFeature,
    filter: (feature: any) => {
      // Optional: Filter out some features if needed for performance
      return true;
    }
  }), [getAreaStyle, onEachFeature]);

  // Map Controls - helper function for zoom
  const handleZoom = useCallback((direction: 'in' | 'out') => (e: React.MouseEvent) => {
    e.preventDefault();
    const mapElement = document.querySelector('.leaflet-container');
    if (mapElement) {
      // Access the Leaflet map instance using the internal property
      // @ts-ignore - This is a valid way to access the map in Leaflet
      const map = mapElement['_leaflet_map'];
      if (map) {
        direction === 'in' ? map.zoomIn() : map.zoomOut();
      }
    }
  }, []);

  // Modify the GeoJSON style function to use historical data when appropriate
  const getGeoJsonStyle = useCallback((feature) => {
    // Default style
    const defaultStyle = {
      fillColor: '#94a3b8',
      weight: 1,
      opacity: 0.7,
      color: '#334155',
      dashArray: '1',
      fillOpacity: 0.7
    };
    
    if (!feature.properties) return defaultStyle;
    
    let riskLevel = feature.properties.risk_level;
    
    // Use historical data if we're in historical mode
    if (showHistorical && historicalData && historicalData.lsoa_risk) {
      const historicalRisk = historicalData.lsoa_risk.find(
        item => item.lsoa_code === feature.properties.lsoa_code
      );
      
      if (historicalRisk) {
        riskLevel = historicalRisk.risk_level;
      }
    }
    
    // Get color based on risk level
    const fillColor = getRiskColor(riskLevel);
    
    return {
      ...defaultStyle,
      fillColor,
      // Highlight the selected LSOA
      weight: selectedLSOA === feature.properties.lsoa_code ? 3 : 1,
      color: selectedLSOA === feature.properties.lsoa_code ? '#3b82f6' : '#334155',
    };
  }, [selectedLSOA, showHistorical, historicalData]);

  return (
    <div className="w-full h-full relative rounded-lg overflow-hidden border border-gray-200">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 z-10">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-800"></div>
        </div>
      )}
      
      {error && (
        <div className="absolute top-0 left-0 right-0 bg-red-500 text-white p-2 text-center text-sm z-10">
          {error}
        </div>
      )}
      
      <MapContainer 
        center={[51.515, -0.09]} 
        zoom={12} 
        style={{ height: '100%', width: '100%' }}
        zoomControl={false}
        preferCanvas={true} // Use canvas for better performance with large datasets
      >
        <MapStyleLayer />
        
        {/* Simple base layer - use a minimal tile layer for better performance */}
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          subdomains="abcd"
          maxZoom={19}
        />
        
        {/* LSOA Boundaries Layer - with error boundary */}
        {lsoaData && (
          <GeoJSON
            key={`lsoa-geojson-${Date.now()}`} // Add a key to force re-render when data changes
            data={lsoaData}
            style={getGeoJsonStyle}
            onEachFeature={onEachFeature}
            ref={geoJsonLayerRef}
          />
        )}
        
        {/* Police Allocation Layer - Only shown when enabled */}
        {showPoliceAllocation && (
          <div className="police-allocation-layer">
            {/* Use static mock police allocation data */}
            {[
              { lat: 51.516, lon: -0.09, officer_count: 3, risk_score: 0.85 },
              { lat: 51.513, lon: -0.12, officer_count: 2, risk_score: 0.75 },
              { lat: 51.518, lon: -0.14, officer_count: 4, risk_score: 0.92 },
              { lat: 51.514, lon: -0.06, officer_count: 1, risk_score: 0.68 },
              { lat: 51.511, lon: -0.08, officer_count: 2, risk_score: 0.72 }
            ].map((cluster, index) => (
              <ZoomAwareMarker
                key={`officer-${index}`}
                position={[cluster.lat, cluster.lon]}
                patrolType={index % 3 === 0 ? 'vehicle' : 'officer'}
              >
                <Popup>
                  <div>
                    <h3 className="font-bold mb-1">Patrol {index + 1}</h3>
                    <p>Officers: {cluster.officer_count || 0}</p>
                    <p>Risk Score: {cluster.risk_score?.toFixed(2) || 'N/A'}</p>
                  </div>
                </Popup>
              </ZoomAwareMarker>
            ))}
          </div>
        )}
        
        {/* Prediction Markers Layer */}
        {predictionMarkers.length > 0 && (
          <div className="prediction-markers-layer">
            {predictionMarkers.map((marker, index) => {
              // Set color based on risk level
              const color = marker.risk === 'High' ? '#ef4444' : 
                            marker.risk === 'Medium' ? '#f59e0b' : '#22c55e';
              
              return (
                <CircleMarker
                  key={`prediction-${index}`}
                  center={[marker.lat, marker.lon]}
                  radius={marker.risk === 'High' ? 8 : marker.risk === 'Medium' ? 6 : 4}
                  pathOptions={{
                    color: color,
                    fillColor: color,
                    fillOpacity: 0.7,
                    weight: 1
                  }}
                >
                  <Popup>
                    <div>
                      <h3 className="font-bold mb-1">Predicted Hotspot</h3>
                      <p>Risk Level: {marker.risk}</p>
                      <p>Model: {predictionModel}</p>
                      <p>Confidence: {Math.floor(70 + Math.random() * 25)}%</p>
                    </div>
                  </Popup>
                </CircleMarker>
              );
            })}
          </div>
        )}
        
        {/* Map Controls */}
        <div className="leaflet-top leaflet-left">
          <div className="leaflet-control leaflet-bar">
            <a 
              href="#" 
              title="Zoom in"
              onClick={handleZoom('in')}
              style={{ 
                display: 'block', 
                height: '30px', 
                width: '30px', 
                lineHeight: '30px', 
                textAlign: 'center', 
                fontSize: '18px',
                fontWeight: 'bold',
                color: '#374151',
                textDecoration: 'none',
                backgroundColor: 'white',
                borderBottom: '1px solid #ccc'
              }}
            >
              +
            </a>
            <a 
              href="#" 
              title="Zoom out"
              onClick={handleZoom('out')}
              style={{ 
                display: 'block', 
                height: '30px', 
                width: '30px', 
                lineHeight: '30px', 
                textAlign: 'center', 
                fontSize: '18px',
                fontWeight: 'bold',
                color: '#374151',
                textDecoration: 'none',
                backgroundColor: 'white'
              }}
            >
              -
            </a>
          </div>
        </div>
        
        {/* Map Legend */}
        <MapLegend 
          title="Burglary Risk"
          items={[
            { color: getRiskColor('Very Low'), label: 'Very Low' },
            { color: getRiskColor('Low'), label: 'Low' },
            { color: getRiskColor('Medium'), label: 'Medium' },
            { color: getRiskColor('High'), label: 'High' },
            { color: getRiskColor('Very High'), label: 'Very High' },
          ]}
          showPoliceAllocation={showPoliceAllocation}
          showPredictions={predictionMarkers.length > 0}
          predictionItems={predictionMarkers.length > 0 ? [
            { color: '#ef4444', label: 'High Risk Prediction' },
            { color: '#f59e0b', label: 'Medium Risk Prediction' },
            { color: '#22c55e', label: 'Low Risk Prediction' },
          ] : []}
        />
      </MapContainer>
    </div>
  );
};

// Error boundary component to catch any rendering errors in the GeoJSON
class ErrorBoundary extends React.Component<{children: React.ReactNode, fallback: React.ReactNode}, {hasError: boolean}> {
  constructor(props: {children: React.ReactNode, fallback: React.ReactNode}) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('GeoJSON rendering error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback;
    }

    return this.props.children;
  }
}

export default MapComponent; 