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

// Custom police icon with glow effect
const policeIcon = new L.Icon({
  iconUrl: 'https://cdn-icons-png.flaticon.com/512/1680/1680070.png',
  iconSize: [24, 24],
  iconAnchor: [12, 24],
  popupAnchor: [0, -24],
  className: 'police-icon glow-blue',
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

// Mock London LSOA GeoJSON data with more realistic styling
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

// Mock police allocation points with improved accuracy and data
const policeAllocations = [
  { position: [51.513, -0.092], officerId: 'A-137', coverage: 10, patrolType: 'Foot', effectivenessScore: 86 },
  { position: [51.517, -0.099], officerId: 'B-244', coverage: 12, patrolType: 'Vehicle', effectivenessScore: 92 },
  { position: [51.512, -0.082], officerId: 'C-089', coverage: 8, patrolType: 'Foot', effectivenessScore: 78 },
  { position: [51.505, -0.114], officerId: 'D-316', coverage: 15, patrolType: 'Vehicle', effectivenessScore: 89 },
  { position: [51.522, -0.105], officerId: 'E-421', coverage: 11, patrolType: 'Foot', effectivenessScore: 81 },
  { position: [51.573, 0.132], officerId: 'F-052', coverage: 6, patrolType: 'Community', effectivenessScore: 74 },
];

const MapComponent = ({ onLSOASelect, showPoliceAllocation, selectedLSOA }) => {
  const mapRef = useRef(null);
  const geoJsonLayerRef = useRef(null);
  const [highlightedArea, setHighlightedArea] = useState(null);
  const [mapMode, setMapMode] = useState('default'); // default, heatmap, risk
  const [isMapLoading, setIsMapLoading] = useState(true);
  
  // Style function for GeoJSON areas
  const getAreaStyle = (feature) => {
    const isSelected = selectedLSOA === feature.properties.lsoa_code;
    const isHighlighted = highlightedArea === feature.properties.lsoa_code;
    const count = feature.properties.burglary_count;
    const riskLevel = feature.properties.risk_level;
    
    // Color schemes based on map mode
    let fillColor;
    
    if (mapMode === 'risk') {
      // Risk-based coloring
      switch(riskLevel) {
        case 'Very High': fillColor = '#ef4444'; break; // red
        case 'High': fillColor = '#f97316'; break; // orange
        case 'Medium': fillColor = '#facc15'; break; // yellow
        case 'Low': fillColor = '#22c55e'; break; // green
        default: fillColor = '#3b82f6'; // blue
      }
    } else {
      // Intensity-based coloring (default and heatmap modes)
      const intensity = Math.min(1, count / 60);
      fillColor = isSelected 
        ? '#4338ca' 
        : `rgba(${Math.round(255 * intensity)}, ${Math.round(100 * (1 - intensity) + 50 * intensity)}, ${Math.round(255 * (1 - intensity))}, 0.7)`;
    }
    
    return {
      fillColor: fillColor,
      weight: isSelected || isHighlighted ? 3 : 1,
      opacity: 1,
      color: isSelected ? '#6366f1' : isHighlighted ? '#3b82f6' : '#6b7280',
      dashArray: isSelected ? '' : isHighlighted ? '5,5' : '3,5',
      fillOpacity: isSelected ? 0.7 : isHighlighted ? 0.6 : 0.5
    };
  };

  // Event handlers for GeoJSON interactions
  const onEachFeature = (feature, layer) => {
    const lsoaCode = feature.properties.lsoa_code;
    const lsoaName = feature.properties.lsoa_name;
    const burglaryCount = feature.properties.burglary_count;
    const riskLevel = feature.properties.risk_level;
    
    // Create a custom popup with styled HTML
    const popupContent = `
      <div class="bg-gray-800 p-3 rounded-md border border-gray-700 text-white">
        <h3 class="text-lg font-semibold">${lsoaName}</h3>
        <div class="mt-2 grid grid-cols-2 gap-2 text-sm">
          <div>
            <span class="text-gray-400">LSOA:</span> 
            <span class="font-mono">${lsoaCode}</span>
          </div>
          <div>
            <span class="text-gray-400">Burglaries:</span> 
            <span class="font-bold text-${
              riskLevel === 'Very High' ? 'red' : 
              riskLevel === 'High' ? 'orange' : 
              riskLevel === 'Medium' ? 'yellow' : 'green'
            }-400">${burglaryCount}</span>
          </div>
          <div class="col-span-2">
            <span class="text-gray-400">Risk Level:</span> 
            <span class="px-2 py-0.5 rounded-full bg-${
              riskLevel === 'Very High' ? 'red' : 
              riskLevel === 'High' ? 'orange' : 
              riskLevel === 'Medium' ? 'yellow' : 'green'
            }-700/50 text-${
              riskLevel === 'Very High' ? 'red' : 
              riskLevel === 'High' ? 'orange' : 
              riskLevel === 'Medium' ? 'yellow' : 'green'
            }-400">${riskLevel}</span>
          </div>
        </div>
        <div class="mt-3 text-center">
          <button class="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-white text-xs select-area-btn">
            Select Area
          </button>
        </div>
      </div>
    `;
    
    const popup = L.popup({
      closeButton: false,
      className: 'custom-popup'
    }).setContent(popupContent);
    
    layer.bindPopup(popup);
    
    layer.on({
      mouseover: () => {
        setHighlightedArea(lsoaCode);
        layer.openPopup();
      },
      mouseout: () => {
        setHighlightedArea(null);
        layer.closePopup();
      },
      click: () => {
        onLSOASelect(lsoaCode);
      }
    });
    
    // Add event listener to the popup content after it's added to the DOM
    layer.on('popupopen', () => {
      const selectBtn = document.querySelector('.select-area-btn');
      if (selectBtn) {
        selectBtn.addEventListener('click', () => {
          onLSOASelect(lsoaCode);
          layer.closePopup();
        });
      }
    });
  };

  // Reset the GeoJSON style when props change
  useEffect(() => {
    if (geoJsonLayerRef.current) {
      geoJsonLayerRef.current.setStyle(getAreaStyle);
    }
  }, [selectedLSOA, highlightedArea, mapMode]);

  // Add effect to handle map loading state
  useEffect(() => {
    // Simulate map loading completion
    const timer = setTimeout(() => {
      setIsMapLoading(false);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="dashboard-card overflow-hidden">
      <div className="flex items-center justify-between card-header px-4 py-3">
        <h3 className="text-white text-lg font-semibold flex items-center">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M12 1.586l-4 4v12.828l4-4V1.586zM3.707 3.293A1 1 0 002 4v10a1 1 0 00.293.707L6 18.414V5.586L3.707 3.293zM17.707 5.293L14 1.586v12.828l2.293 2.293A1 1 0 0018 16V6a1 1 0 00-.293-.707z" clipRule="evenodd" />
          </svg>
          Predictive Map
        </h3>
        
        <div className="flex items-center space-x-3">
          <button 
            className={`px-2 py-1 rounded text-xs ${mapMode === 'default' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'}`}
            onClick={() => setMapMode('default')}
          >
            Default
          </button>
          <button 
            className={`px-2 py-1 rounded text-xs ${mapMode === 'heatmap' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'}`}
            onClick={() => setMapMode('heatmap')}
          >
            Heatmap
          </button>
          <button 
            className={`px-2 py-1 rounded text-xs ${mapMode === 'risk' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'}`}
            onClick={() => setMapMode('risk')}
          >
            Risk Levels
          </button>
        </div>
      </div>
      
      <div className="map-container">
        {isMapLoading ? (
          <div className="h-full w-full flex flex-col items-center justify-center bg-gray-900">
            <div className="w-12 h-12 border-4 border-t-blue-500 border-blue-500/20 rounded-full animate-spin mb-4"></div>
            <p className="text-gray-400">Loading map data...</p>
          </div>
        ) : (
          <MapContainer 
            center={[51.52, -0.09]} 
            zoom={13} 
            scrollWheelZoom={true}
            style={{ height: '100%', width: '100%' }}
            ref={mapRef}
            attributionControl={false}
            zoomControl={false}
          >
            <MapStyleLayer />
            
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
              className={mapMode === 'heatmap' ? 'low-opacity-tiles' : ''}
            />
            
            {/* Custom Heatmap using CircleMarkers */}
            {mapMode === 'heatmap' && (
              <FeatureGroup>
                {londonBurglaryPoints.map((point, idx) => (
                  <CircleMarker
                    key={`heatpoint-${idx}`}
                    center={[point[0], point[1]]}
                    radius={Math.sqrt(point[2]) * 1.5}
                    fillColor={
                      point[2] > 45 ? '#ef4444' :
                      point[2] > 35 ? '#f97316' :
                      point[2] > 25 ? '#facc15' : 
                      '#3b82f6'
                    }
                    fillOpacity={0.7}
                    stroke={false}
                  />
                ))}
                {londonBurglaryPoints.map((point, idx) => (
                  <CircleMarker
                    key={`heatglow-${idx}`}
                    center={[point[0], point[1]]}
                    radius={Math.sqrt(point[2]) * 3}
                    fillColor={
                      point[2] > 45 ? '#ef4444' :
                      point[2] > 35 ? '#f97316' :
                      point[2] > 25 ? '#facc15' : 
                      '#3b82f6'
                    }
                    fillOpacity={0.2}
                    stroke={false}
                  />
                ))}
              </FeatureGroup>
            )}
            
            <LayersControl position="bottomright">
              <LayersControl.BaseLayer checked name="Dark">
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                  url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                />
              </LayersControl.BaseLayer>
              
              <LayersControl.BaseLayer name="Streets">
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  className="inverted-tiles"
                />
              </LayersControl.BaseLayer>
              
              <LayersControl.Overlay checked name="LSOA Boundaries">
                <GeoJSON 
                  data={mockLondonLSOAs}
                  style={getAreaStyle}
                  onEachFeature={onEachFeature}
                  ref={geoJsonLayerRef}
                />
              </LayersControl.Overlay>
              
              {mapMode !== 'heatmap' && (
                <LayersControl.Overlay checked={mapMode === 'default'} name="Burglary Hotspots">
                  <FeatureGroup>
                    {londonBurglaryPoints.map((spot, index) => (
                      <CircleMarker
                        key={`hotspot-${index}`}
                        center={[spot[0], spot[1]]}
                        radius={Math.sqrt(spot[2]) * 0.7}
                        fillColor="rgba(255, 120, 0, 0.8)"
                        color="#f97316"
                        weight={1}
                        opacity={0.8}
                        fillOpacity={0.6}
                      >
                        <Popup className="custom-popup">
                          <div className="bg-gray-800 p-2 rounded-md border border-gray-700 text-white">
                            <strong>Burglary Hotspot</strong>
                            <p className="text-sm">{spot[2]} incidents reported</p>
                            <div className="text-xs text-gray-400 mt-1">Long: {spot[1].toFixed(4)}, Lat: {spot[0].toFixed(4)}</div>
                          </div>
                        </Popup>
                      </CircleMarker>
                    ))}
                  </FeatureGroup>
                </LayersControl.Overlay>
              )}
              
              {/* Police allocation only shown when toggle is active */}
              {showPoliceAllocation && (
                <LayersControl.Overlay checked name="Optimal Police Allocation">
                  <FeatureGroup>
                    {policeAllocations.map((officer, index) => (
                      <React.Fragment key={`officer-${index}`}>
                        <Marker
                          position={officer.position}
                          icon={policeIcon}
                        >
                          <Popup className="custom-popup">
                            <div className="bg-gray-800 p-3 rounded-md border border-gray-700 text-white">
                              <h3 className="text-blue-400 font-semibold">Officer {officer.officerId}</h3>
                              <div className="grid grid-cols-2 gap-2 mt-2 text-sm">
                                <div>
                                  <span className="text-gray-400">Patrol:</span> 
                                  <span className="ml-1">{officer.patrolType}</span>
                                </div>
                                <div>
                                  <span className="text-gray-400">Coverage:</span> 
                                  <span className="ml-1 font-semibold">{officer.coverage} incidents</span>
                                </div>
                                <div className="col-span-2">
                                  <span className="text-gray-400">Effectiveness:</span> 
                                  <span className="ml-1 text-green-400">{officer.effectivenessScore}%</span>
                                </div>
                              </div>
                              <div className="mt-2 text-xs text-gray-400">
                                Allocated using K-means clustering
                              </div>
                            </div>
                          </Popup>
                        </Marker>
                        <CircleMarker
                          center={officer.position}
                          radius={officer.coverage * 1.5}
                          fillColor="rgba(59, 130, 246, 0.2)"
                          color="rgba(59, 130, 246, 0.8)"
                          weight={2}
                          dashArray="5,5"
                          opacity={0.7}
                          fillOpacity={0.1}
                        />
                      </React.Fragment>
                    ))}
                  </FeatureGroup>
                </LayersControl.Overlay>
              )}
            </LayersControl>
            
            {/* Enhanced map controls */}
            <div className="leaflet-control leaflet-bar custom-controls" style={{ position: 'absolute', top: '10px', right: '10px', zIndex: 1000 }}>
              <button className="bg-gray-800 hover:bg-gray-700 text-white w-8 h-8 flex items-center justify-center rounded-md border border-gray-700 mb-1"
                onClick={() => mapRef.current && mapRef.current.zoomIn()}>
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                </svg>
              </button>
              <button className="bg-gray-800 hover:bg-gray-700 text-white w-8 h-8 flex items-center justify-center rounded-md border border-gray-700"
                onClick={() => mapRef.current && mapRef.current.zoomOut()}>
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18 12H6" />
                </svg>
              </button>
            </div>
            
            {/* Map Legend */}
            <div className="leaflet-bottom leaflet-left" style={{ zIndex: 1000, margin: '0 0 30px 10px' }}>
              <div className="leaflet-control leaflet-bar bg-gray-800 p-3 rounded-md shadow-lg border border-gray-700">
                <h4 className="font-semibold text-sm mb-2 text-white">{
                  mapMode === 'risk' ? 'Risk Levels' : 
                  mapMode === 'heatmap' ? 'Burglary Heatmap' : 
                  'Burglary Intensity'
                }</h4>
                <div className="heatmap-legend mb-1">
                  <div className="heatmap-legend-gradient"></div>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="heatmap-label">Low</span>
                  <span className="heatmap-label">Medium</span>
                  <span className="heatmap-label">High</span>
                </div>
                
                {showPoliceAllocation && (
                  <div className="mt-3 pt-3 border-t border-gray-700">
                    <div className="flex items-center mb-1 text-white text-sm">
                      <div className="h-3 w-3 rounded-full bg-blue-500 mr-2"></div>
                      <span>Police Officer</span>
                    </div>
                    <div className="flex items-center text-white text-sm">
                      <div className="h-3 w-3 border-2 border-dashed border-blue-500 rounded-full mr-2"></div>
                      <span>Coverage Area</span>
                    </div>
                  </div>
                )}
                
                <div className="mt-3 text-xs text-gray-400">
                  Click on an area to view detailed data
                </div>
              </div>
            </div>
          </MapContainer>
        )}
      </div>

      {/* Selected area info - only show if an area is selected */}
      {selectedLSOA && (
        <div className="p-3 bg-gray-800 border-t border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-400 mr-2" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M5.05 4.05a7 7 0 119.9 9.9L10 18.9l-4.95-4.95a7 7 0 010-9.9zM10 11a2 2 0 100-4 2 2 0 000 4z" clipRule="evenodd" />
              </svg>
              <span className="text-white font-medium">Selected: </span>
              <span className="text-blue-400 ml-1">{selectedLSOA}</span>
            </div>
            <button 
              className="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded"
              onClick={() => onLSOASelect(null)}
            >
              Clear
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default MapComponent; 