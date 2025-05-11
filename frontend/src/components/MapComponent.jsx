import React, { useEffect, useState, useRef } from 'react';
import { MapContainer, TileLayer, GeoJSON, Marker, Popup, CircleMarker, LayersControl, FeatureGroup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for Leaflet default icon issue in React
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

// Helper component to set dark mode map style
const MapStyleLayer = () => {
  const map = useMap();
  
  useEffect(() => {
    // Apply dark mode styling to the map
    map.getContainer().style.background = '#111827';
    
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
const ZoomDependentMarkers = ({ children }) => {
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
  
  // Pass the current zoom level to all children
  return React.Children.map(children, child => {
    return React.cloneElement(child, { currentZoom: zoom });
  });
};

// Function to calculate marker size based on zoom level
const getZoomDependentSize = (baseSize, zoom) => {
  // Enhanced scaling factor with more aggressive scaling at higher zoom levels
  // Start scaling from zoom level 11 onwards
  if (zoom <= 11) return baseSize;
  if (zoom <= 13) return baseSize * (1 + (zoom - 11) * 0.25);
  if (zoom <= 15) return baseSize * (1.5 + (zoom - 13) * 0.4);
  return baseSize * (2.3 + (zoom - 15) * 0.5); // More dramatic scaling at highest zoom levels
};

// London coordinates for burglary heatmap points
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

// Mock London LSOA GeoJSON data
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
    },
    {
      type: 'Feature',
      properties: {
        lsoa_code: 'E01000012',
        lsoa_name: 'Central London 012C',
        burglary_count: 47,
        risk_level: 'High'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.115, 51.505],
          [-0.11, 51.505],
          [-0.11, 51.51],
          [-0.115, 51.51],
          [-0.115, 51.505]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        lsoa_code: 'E01000018',
        lsoa_name: 'Westminster 008A',
        burglary_count: 39,
        risk_level: 'Medium'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.125, 51.515],
          [-0.12, 51.515],
          [-0.12, 51.52],
          [-0.125, 51.52],
          [-0.125, 51.515]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        lsoa_code: 'E01000022',
        lsoa_name: 'Camden 005B',
        burglary_count: 56,
        risk_level: 'Very High'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.13, 51.535],
          [-0.125, 51.535],
          [-0.125, 51.54],
          [-0.13, 51.54],
          [-0.13, 51.535]
        ]]
      }
    }
  ]
};

// Mock police allocation points
const policeAllocations = [
  { position: [51.513, -0.092], officerId: 'A-137', coverage: 10, patrolType: 'Foot', effectivenessScore: 86 },
  { position: [51.517, -0.099], officerId: 'B-244', coverage: 12, patrolType: 'Vehicle', effectivenessScore: 92 },
  { position: [51.512, -0.082], officerId: 'C-089', coverage: 8, patrolType: 'Foot', effectivenessScore: 78 },
  { position: [51.505, -0.114], officerId: 'D-316', coverage: 15, patrolType: 'Vehicle', effectivenessScore: 89 },
  { position: [51.522, -0.105], officerId: 'E-421', coverage: 11, patrolType: 'Foot', effectivenessScore: 81 },
  { position: [51.573, 0.132], officerId: 'F-052', coverage: 6, patrolType: 'Community', effectivenessScore: 74 },
];

// Add police officer emoji icon with dynamic sizing
const createPoliceIcon = (zoom = 11) => {
  const size = getZoomDependentSize(20, zoom);
  return L.divIcon({
    html: '👮',
    className: 'police-marker',
    iconSize: [size, size],
    iconAnchor: [size/2, size/2]
  });
};

// Add police vehicle emoji icon with dynamic sizing
const createVehicleIcon = (zoom = 11) => {
  const size = getZoomDependentSize(22, zoom);
  return L.divIcon({
    html: '🚓',
    className: 'police-marker',
    iconSize: [size, size],
    iconAnchor: [size/2, size/2]
  });
};

// Initial icons (will be replaced by dynamic ones)
const policeIcon = createPoliceIcon();
const vehicleIcon = createVehicleIcon();

// DynamicCircleMarker component for zoom-dependent sizing
const DynamicCircleMarker = ({ center, baseRadius, currentZoom = 11, pathOptions, children }) => {
  // Calculate radius based on zoom level
  const radius = getZoomDependentSize(baseRadius, currentZoom);
  
  return (
    <CircleMarker
      center={center}
      radius={radius}
      pathOptions={pathOptions}
    >
      {children}
    </CircleMarker>
  );
};

// DynamicMarker component for zoom-dependent emoji icons
const DynamicMarker = ({ position, patrolType, currentZoom = 11, children }) => {
  // Create the appropriate icon based on patrol type and current zoom
  const icon = patrolType === 'Vehicle' ? createVehicleIcon(currentZoom) : createPoliceIcon(currentZoom);
  
  return (
    <Marker position={position} icon={icon}>
      {children}
    </Marker>
  );
};

const MapComponent = ({ onLSOASelect, showPoliceAllocation, selectedLSOA, policeAllocationData }) => {
  const mapRef = useRef(null);
  const [selectedLSOAData, setSelectedLSOAData] = useState(null);
  const [isLoadingPoliceData, setIsLoadingPoliceData] = useState(false);
  
  // Update loading state when police data changes
  useEffect(() => {
    if (showPoliceAllocation && !policeAllocationData) {
      setIsLoadingPoliceData(true);
    } else {
      setIsLoadingPoliceData(false);
    }
  }, [showPoliceAllocation, policeAllocationData]);
  
  // Style function for LSOA areas
  const getAreaStyle = (feature) => {
    const isSelected = selectedLSOA === feature.properties.lsoa_code;
    
    // Base styles
    const baseStyle = {
      weight: isSelected ? 3 : 1.5,
      opacity: isSelected ? 1 : 0.7,
      color: isSelected ? '#3b82f6' : '#6b7280',
      dashArray: isSelected ? '' : '3',
      fillOpacity: 0.6,
    };
    
    // Determine fill color based on burglary risk level
    let fillColor = '#047857'; // Low risk (default)
    
    if (feature.properties.risk_level === 'Medium') {
      fillColor = '#ca8a04';
    } else if (feature.properties.risk_level === 'High') {
      fillColor = '#b91c1c';
    } else if (feature.properties.risk_level === 'Very High') {
      fillColor = '#7f1d1d';
    }
    
    return {
      ...baseStyle,
      fillColor: isSelected ? '#3b82f6' : fillColor,
    };
  };
  
  const onEachFeature = (feature, layer) => {
    // Add click handler
    layer.on({
      click: () => {
        onLSOASelect(feature.properties.lsoa_code);
        setSelectedLSOAData(feature.properties);
        
        if (mapRef.current) {
          const map = mapRef.current;
          map.fitBounds(layer.getBounds());
        }
      }
    });
    
    // Add popup
    const popupContent = `
      <div>
        <h3 class="font-semibold">${feature.properties.lsoa_name}</h3>
        <p class="text-sm">Code: ${feature.properties.lsoa_code}</p>
        <p class="text-sm">Burglaries: ${feature.properties.burglary_count}</p>
        <p class="text-sm">Risk Level: ${feature.properties.risk_level}</p>
      </div>
    `;
    
    layer.bindPopup(popupContent);
  };
  
  return (
    <div className="dashboard-card rounded-lg overflow-hidden h-full flex flex-col">
      <div className="flex justify-between items-center p-3 border-b border-gray-800">
        <h3 className="text-base font-semibold text-white">
          {selectedLSOA ? `${selectedLSOAData?.lsoa_name} (${selectedLSOA})` : 'London Burglary Map'}
        </h3>
        {selectedLSOA && (
          <button 
            className="text-xs text-gray-400 bg-gray-800 px-2 py-1 rounded-md hover:bg-gray-700"
            onClick={() => onLSOASelect(null)}
          >
            Clear Selection
          </button>
        )}
      </div>
      
      <div className="flex-1 min-h-[400px] relative">
        {isLoadingPoliceData && (
          <div className="absolute top-2 right-2 z-[1000] bg-gray-800/80 text-white text-xs py-1 px-3 rounded-full flex items-center">
            <div className="w-3 h-3 border-2 border-t-blue-500 border-blue-500/20 rounded-full animate-spin mr-2"></div>
            Loading police units...
          </div>
        )}
        
        <MapContainer
          center={[51.505, -0.09]}
          zoom={11}
          scrollWheelZoom={true}
          style={{ height: "100%", width: "100%", background: "#111827", position: "absolute" }}
          ref={mapRef}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            className="dark-map-tiles"
          />
          
          <MapStyleLayer />
          
          <LayersControl position="topright">
            <LayersControl.Overlay name="LSOA Areas" checked>
              <GeoJSON 
                data={mockLondonLSOAs}
                style={getAreaStyle}
                onEachFeature={onEachFeature}
              />
            </LayersControl.Overlay>
            
            <LayersControl.Overlay name="Burglary Hotspots" checked>
              <ZoomDependentMarkers>
                <FeatureGroup>
                  {londonBurglaryPoints.map((point, index) => (
                    <DynamicCircleMarker 
                      key={index}
                      center={[point[0], point[1]]}
                      baseRadius={point[2] / 10}
                      pathOptions={{
                        fillColor: '#ef4444',
                        fillOpacity: 0.6,
                        color: '#b91c1c',
                        weight: 1
                      }}
                    >
                      <Popup>
                        <div>
                          <h3 className="font-semibold">Burglary Hotspot</h3>
                          <p className="text-sm">Incidents: {point[2]}</p>
                          <p className="text-sm">Location: {point[0].toFixed(4)}, {point[1].toFixed(4)}</p>
                        </div>
                      </Popup>
                    </DynamicCircleMarker>
                  ))}
                </FeatureGroup>
              </ZoomDependentMarkers>
            </LayersControl.Overlay>
            
            {showPoliceAllocation && policeAllocationData && (
              <LayersControl.Overlay name="Police Units" checked>
                <ZoomDependentMarkers>
                  <FeatureGroup>
                    {policeAllocationData.map((unit) => (
                      <React.Fragment key={unit.unit_id}>
                        {/* Police coverage circle */}
                        <DynamicCircleMarker
                          center={[unit.lat, unit.lon]}
                          baseRadius={unit.patrol_radius * 10}  
                          pathOptions={{
                            fillColor: '#3b82f6',
                            fillOpacity: 0.2,
                            color: '#2563eb',
                            weight: 1
                          }}
                        >
                          <Popup>
                            <div>
                              <h3 className="font-semibold">Police Unit #{unit.unit_id}</h3>
                              <p className="text-sm">Est. Burglaries: {unit.estimated_burglaries.toLocaleString()}</p>
                              <p className="text-sm">Patrol Radius: {unit.patrol_radius} km</p>
                              <p className="text-sm">Patrol Type: {unit.patrol_type}</p>
                              <p className="text-sm">Effectiveness: {unit.effectiveness_score}%</p>
                            </div>
                          </Popup>
                        </DynamicCircleMarker>
                        
                        {/* Police officer/vehicle emoji marker */}
                        <DynamicMarker
                          position={[unit.lat, unit.lon]}
                          patrolType={unit.patrol_type}
                        >
                          <Popup>
                            <div>
                              <h3 className="font-semibold">Unit #{unit.unit_id}</h3>
                              <p className="text-sm">Patrol Type: {unit.patrol_type}</p>
                              <p className="text-sm">Area Burglaries: {unit.estimated_burglaries.toLocaleString()}</p>
                            </div>
                          </Popup>
                        </DynamicMarker>
                      </React.Fragment>
                    ))}
                  </FeatureGroup>
                </ZoomDependentMarkers>
              </LayersControl.Overlay>
            )}
          </LayersControl>
        </MapContainer>
      </div>
    </div>
  );
};

export default MapComponent; 