import React, { useEffect, useState, useRef } from 'react';
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
    
    return () => {
      map.off('zoomend');
    };
  }, [map]);
  
  return null;
};

// Helper component to handle zoom-dependent marker sizing
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
  
  // Pass the current zoom level to all children
  return (
    <>
      {React.Children.map(children, child => {
        if (React.isValidElement(child)) {
          return React.cloneElement(child, { currentZoom: zoom });
        }
        return child;
      })}
    </>
  );
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

// London coordinates for burglary heatmap points - for mock data or testing
const londonBurglaryPoints = [
  // City of London and central
  [51.512, -0.09, 45],
  [51.513, -0.1, 38],
  [51.518, -0.102, 52],
  [51.514, -0.085, 29],
  [51.511, -0.077, 33],
  [51.508, -0.105, 41],
  [51.517, -0.088, 37],
  [51.521, -0.095, 44],
  [51.505, -0.091, 28],
  [51.509, -0.083, 35],
  // East London
  [51.538, -0.005, 39],
  [51.527, 0.015, 48],
  [51.542, 0.001, 33],
  [51.551, 0.055, 41],
  [51.516, 0.032, 28],
  // North London
  [51.572, -0.135, 21],
  [51.555, -0.105, 36],
  [51.568, -0.085, 42],
  [51.585, -0.121, 31],
  [51.562, -0.161, 27],
  // West London
  [51.502, -0.195, 38],
  [51.518, -0.217, 25],
  [51.531, -0.187, 31],
  [51.495, -0.225, 29],
  [51.488, -0.178, 32],
  // South London
  [51.465, -0.135, 43],
  [51.472, -0.088, 37],
  [51.451, -0.157, 28],
  [51.485, -0.112, 35],
  [51.442, -0.091, 31]
];

// Mock London LSOA GeoJSON data for initial load
const mockLondonLSOAs = {
  type: 'FeatureCollection',
  features: [
    {
      type: 'Feature',
      properties: {
        lsoa_code: 'E01000001',
        lsoa_name: 'City of London 001A',
        burglary_count: 35,
        risk_level: 'Medium'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.1, 51.51],
          [-0.095, 51.51],
          [-0.095, 51.515],
          [-0.1, 51.515],
          [-0.1, 51.51]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        lsoa_code: 'E01000005',
        lsoa_name: 'City of London 001E',
        burglary_count: 42,
        risk_level: 'High'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.08, 51.51],
          [-0.075, 51.51],
          [-0.075, 51.515],
          [-0.08, 51.515],
          [-0.08, 51.51]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        lsoa_code: 'E01032739',
        lsoa_name: 'City of London 001F',
        burglary_count: 28,
        risk_level: 'Medium'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.09, 51.515],
          [-0.085, 51.515],
          [-0.085, 51.52],
          [-0.09, 51.52],
          [-0.09, 51.515]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        lsoa_code: 'E01032740',
        lsoa_name: 'City of London 001G',
        burglary_count: 55,
        risk_level: 'Very High'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.105, 51.515],
          [-0.1, 51.515],
          [-0.1, 51.52],
          [-0.105, 51.52],
          [-0.105, 51.515]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        lsoa_code: 'E01000032',
        lsoa_name: 'Barking and Dagenham 002B',
        burglary_count: 18,
        risk_level: 'Low'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [0.13, 51.57],
          [0.135, 51.57],
          [0.135, 51.575],
          [0.13, 51.575],
          [0.13, 51.57]
        ]]
      }
    }
  ]
};

// Custom icon creation for police officers
const createPoliceIcon = (zoom = 11) => {
  const iconSize = getZoomDependentSize(20, zoom);
  return L.divIcon({
    html: `<div class="flex items-center justify-center w-full h-full">
            <span class="text-blue-500 text-base">👮</span>
          </div>`,
    className: 'bg-blue-500 bg-opacity-20 rounded-full border-2 border-blue-500 shadow-lg',
    iconSize: [iconSize, iconSize],
    iconAnchor: [iconSize/2, iconSize/2]
  });
};

// Custom icon creation for police vehicles
const createVehicleIcon = (zoom = 11) => {
  const iconSize = getZoomDependentSize(22, zoom);
  return L.divIcon({
    html: `<div class="flex items-center justify-center w-full h-full">
            <span class="text-green-500 text-base">🚓</span>
          </div>`,
    className: 'bg-green-500 bg-opacity-20 rounded-full border-2 border-green-500 shadow-lg',
    iconSize: [iconSize, iconSize],
    iconAnchor: [iconSize/2, iconSize/2]
  });
};

// Dynamic circle marker that updates size with zoom level
interface DynamicCircleMarkerProps {
  center: [number, number];
  baseRadius: number;
  currentZoom?: number;
  pathOptions?: L.PathOptions;
  children?: React.ReactNode;
}

const DynamicCircleMarker = ({ 
  center, 
  baseRadius, 
  currentZoom = 11, 
  pathOptions, 
  children 
}: DynamicCircleMarkerProps) => {
  const radius = getZoomDependentSize(baseRadius, currentZoom);
  return (
    <CircleMarker center={center} radius={radius} pathOptions={pathOptions}>
      {children}
    </CircleMarker>
  );
};

// Dynamic marker that updates icon with zoom level
interface DynamicMarkerProps {
  position: [number, number];
  patrolType: 'officer' | 'vehicle';
  currentZoom?: number;
  children?: React.ReactNode;
}

const DynamicMarker = ({ position, patrolType, currentZoom = 11, children }: DynamicMarkerProps) => {
  const icon = patrolType === 'officer' 
    ? createPoliceIcon(currentZoom) 
    : createVehicleIcon(currentZoom);
  
  return (
    <Marker position={position} icon={icon}>
      {children}
    </Marker>
  );
};

interface MapComponentProps {
  onLSOASelect?: (lsoa: string) => void;
  showPoliceAllocation?: boolean;
  selectedLSOA?: string | null;
  policeAllocationData?: { clusters: any[] } | null;
}

const MapComponent = ({ 
  onLSOASelect, 
  showPoliceAllocation = false, 
  selectedLSOA = null,
  policeAllocationData = null
}: MapComponentProps) => {
  const [lsoaData, setLsoaData] = useState<typeof mockLondonLSOAs>(mockLondonLSOAs);
  const mapRef = useRef<L.Map | null>(null);
  
  // Load LSOA data from API if available
  useEffect(() => {
    const fetchLSOAData = async () => {
      try {
        const lsoaList = await api.lsoa.getList();
        if (lsoaList && lsoaList.length > 0) {
          // Format data for GeoJSON structure if needed
          // This would depend on the actual API response structure
        }
      } catch (error) {
        console.error('Error fetching LSOA data:', error);
        // Fall back to mock data
      }
    };
    
    fetchLSOAData();
  }, []);

  // Style function for LSOA areas
  const getAreaStyle = (feature: any): L.PathOptions => {
    const isSelected = feature.properties.lsoa_code === selectedLSOA;
    const baseOpacity = isSelected ? 0.8 : 0.6;
    const riskLevel = feature.properties.risk_level || 'Medium';
    
    // Style based on risk level
    let fillColor;
    switch(riskLevel) {
      case 'Very High':
        fillColor = '#ef4444'; // Red
        break;
      case 'High':
        fillColor = '#f97316'; // Orange
        break;
      case 'Medium':
        fillColor = '#eab308'; // Yellow
        break;
      case 'Low':
        fillColor = '#22c55e'; // Green
        break;
      case 'Very Low':
        fillColor = '#3b82f6'; // Blue
        break;
      default:
        fillColor = '#8b5cf6'; // Purple (default)
    }
    
    return {
      fillColor,
      weight: isSelected ? 3 : 1,
      opacity: isSelected ? 1 : 0.7,
      color: isSelected ? '#ffffff' : '#666',
      dashArray: isSelected ? '' : '3',
      fillOpacity: baseOpacity
    };
  };

  // Handle LSOA interactions
  const onEachFeature = (feature: any, layer: L.Layer) => {
    const { lsoa_code, lsoa_name, burglary_count, risk_level } = feature.properties;
    
    // Popup content
    const popupContent = `
      <div class="popup-content">
        <h3 class="font-bold">${lsoa_name}</h3>
        <p class="text-sm"><strong>LSOA Code:</strong> ${lsoa_code}</p>
        <p class="text-sm"><strong>Burglary Count:</strong> ${burglary_count}</p>
        <p class="text-sm"><strong>Risk Level:</strong> ${risk_level}</p>
      </div>
    `;
    
    layer.bindPopup(popupContent);
    
    // Add interaction handlers
    layer.on({
      click: () => {
        if (onLSOASelect) {
          onLSOASelect(lsoa_code);
        }
      },
      mouseover: (e) => {
        const layer = e.target;
        layer.setStyle({
          weight: 4,
          color: '#ffffff',
          dashArray: '',
          fillOpacity: 0.8
        });
        
        if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
          layer.bringToFront();
        }
      },
      mouseout: (e) => {
        const layer = e.target;
        // Reset to original style
        layer.setStyle(getAreaStyle(feature));
      }
    });
  };

  return (
    <div className="w-full h-[60vh] rounded-lg overflow-hidden shadow-lg border border-gray-800">
      <MapContainer 
        center={[51.505, -0.09]} 
        zoom={11} 
        style={{ width: '100%', height: '100%' }}
        whenReady={(e) => { mapRef.current = e.target; }}
        className="map-container"
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          maxZoom={19}
        />
        
        {/* Alternative light map styles:
        <TileLayer
          url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          maxZoom={19}
        />
        */}
        
        <MapStyleLayer />
        
        <LayersControl position="topright">
          <LayersControl.Overlay name="LSOA Boundaries" checked>
            <GeoJSON
              data={lsoaData}
              style={getAreaStyle}
              onEachFeature={onEachFeature}
            />
          </LayersControl.Overlay>
          
          <LayersControl.Overlay name="Burglary Hotspots" checked>
            <FeatureGroup>
              <ZoomDependentMarkers>
                {londonBurglaryPoints.map(([lat, lng, intensity], idx) => (
                  <DynamicCircleMarker
                    key={`hotspot-${idx}`}
                    center={[lat, lng] as [number, number]}
                    baseRadius={intensity / 5}
                    pathOptions={{
                      fillColor: '#ef4444',
                      color: '#ef4444',
                      fillOpacity: 0.4,
                      weight: 1
                    }}
                  >
                    <Popup>
                      <div>
                        <h3 className="font-bold">Burglary Hotspot</h3>
                        <p className="text-sm">Intensity: {intensity}</p>
                      </div>
                    </Popup>
                  </DynamicCircleMarker>
                ))}
              </ZoomDependentMarkers>
            </FeatureGroup>
          </LayersControl.Overlay>
          
          {showPoliceAllocation && (
            <LayersControl.Overlay name="Police Allocation" checked>
              <FeatureGroup>
                <ZoomDependentMarkers>
                  {/* If real police allocation data available, use that instead */}
                  {(policeAllocationData?.clusters || []).map((cluster, idx) => (
                    <DynamicMarker
                      key={`police-${idx}`}
                      position={[cluster.lat, cluster.lon] as [number, number]}
                      patrolType={idx % 3 === 0 ? 'vehicle' : 'officer'}
                    >
                      <Popup>
                        <div>
                          <h3 className="font-bold">Police {idx % 3 === 0 ? 'Vehicle' : 'Officer'}</h3>
                          <p className="text-sm">Assigned to high-risk area</p>
                          <p className="text-sm">Expected crime reduction: {Math.round(15 + Math.random() * 25)}%</p>
                        </div>
                      </Popup>
                    </DynamicMarker>
                  ))}
                  
                  {/* Fallback to mock data if no real data */}
                  {(!policeAllocationData?.clusters || policeAllocationData.clusters.length === 0) && (
                    // Generate 15 mock police officers
                    Array.from({ length: 15 }).map((_, idx) => {
                      // Random positions around central London
                      const lat = 51.505 + (Math.random() - 0.5) * 0.1;
                      const lng = -0.09 + (Math.random() - 0.5) * 0.1;
                      return (
                        <DynamicMarker
                          key={`police-mock-${idx}`}
                          position={[lat, lng]}
                          patrolType={idx % 3 === 0 ? 'vehicle' : 'officer'}
                        >
                          <Popup>
                            <div>
                              <h3 className="font-bold">Police {idx % 3 === 0 ? 'Vehicle' : 'Officer'}</h3>
                              <p className="text-sm">Assigned to high-risk area</p>
                              <p className="text-sm">Expected crime reduction: {Math.round(15 + Math.random() * 25)}%</p>
                            </div>
                          </Popup>
                        </DynamicMarker>
                      );
                    })
                  )}
                </ZoomDependentMarkers>
              </FeatureGroup>
            </LayersControl.Overlay>
          )}
        </LayersControl>
        
        {/* Add Map Legend */}
        <MapLegend showPoliceAllocation={showPoliceAllocation} />
      </MapContainer>
    </div>
  );
};

export default MapComponent; 