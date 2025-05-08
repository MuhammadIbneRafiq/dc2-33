import React, { useEffect, useState, useRef } from 'react';
import { MapContainer, TileLayer, GeoJSON, Marker, Popup, CircleMarker, LayersControl, FeatureGroup } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for Leaflet default icon issue in React
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

// Custom police icon
const policeIcon = new L.Icon({
  iconUrl: 'https://cdn-icons-png.flaticon.com/512/1680/1680070.png',
  iconSize: [32, 32],
  iconAnchor: [16, 32],
  popupAnchor: [0, -32],
});

// Mock London LSOA GeoJSON data (simplified for this example)
const mockLondonLSOAs = {
  type: 'FeatureCollection',
  features: [
    {
      type: 'Feature',
      properties: {
        lsoa_code: 'E01000001',
        lsoa_name: 'City of London 001A',
        burglary_count: 35
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
        burglary_count: 42
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
        burglary_count: 28
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
        burglary_count: 55
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
        burglary_count: 18
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

// Mock burglary hotspots
const burglaryHotspots = [
  { position: [51.512, -0.09], count: 45 },
  { position: [51.513, -0.1], count: 38 },
  { position: [51.518, -0.102], count: 52 },
  { position: [51.514, -0.085], count: 29 },
  { position: [51.511, -0.077], count: 33 },
  { position: [51.572, 0.135], count: 21 },
];

// Mock police allocation points (using k-means results)
const policeAllocations = [
  { position: [51.513, -0.092], officerId: 1, coverage: 10 },
  { position: [51.517, -0.099], officerId: 2, coverage: 12 },
  { position: [51.512, -0.082], officerId: 3, coverage: 8 },
  { position: [51.573, 0.132], officerId: 4, coverage: 6 },
];

const MapComponent = ({ onLSOASelect, showPoliceAllocation, selectedLSOA }) => {
  const mapRef = useRef(null);
  const geoJsonLayerRef = useRef(null);
  const [highlightedArea, setHighlightedArea] = useState(null);

  // Style function for GeoJSON areas
  const getAreaStyle = (feature) => {
    const isSelected = selectedLSOA === feature.properties.lsoa_code;
    const isHighlighted = highlightedArea === feature.properties.lsoa_code;
    
    // Get intensity based on burglary count
    const intensity = Math.min(1, feature.properties.burglary_count / 60);
    
    return {
      fillColor: isSelected ? '#4338ca' : `rgb(${Math.round(255 * intensity)}, 0, ${Math.round(255 * (1 - intensity))})`,
      weight: isSelected || isHighlighted ? 3 : 1,
      opacity: 1,
      color: isSelected ? '#4338ca' : isHighlighted ? '#2563eb' : '#666',
      dashArray: isSelected ? '' : '3',
      fillOpacity: isSelected ? 0.6 : isHighlighted ? 0.5 : 0.4
    };
  };

  // Event handlers for GeoJSON interactions
  const onEachFeature = (feature, layer) => {
    const lsoaCode = feature.properties.lsoa_code;
    const lsoaName = feature.properties.lsoa_name;
    const burglaryCount = feature.properties.burglary_count;
    
    layer.on({
      mouseover: () => {
        setHighlightedArea(lsoaCode);
        layer.bindTooltip(`${lsoaName}<br/>Burglaries: ${burglaryCount}`).openTooltip();
      },
      mouseout: () => {
        setHighlightedArea(null);
        layer.closeTooltip();
      },
      click: () => {
        onLSOASelect(lsoaCode);
      }
    });
  };

  // Reset the GeoJSON style when props change
  useEffect(() => {
    if (geoJsonLayerRef.current) {
      geoJsonLayerRef.current.setStyle(getAreaStyle);
    }
  }, [selectedLSOA, highlightedArea]);

  return (
    <div className="map-container">
      <MapContainer 
        center={[51.52, -0.09]} 
        zoom={13} 
        scrollWheelZoom={true}
        style={{ height: '100%', width: '100%' }}
        ref={mapRef}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        
        <LayersControl position="topright">
          <LayersControl.BaseLayer checked name="OpenStreetMap">
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
          </LayersControl.BaseLayer>
          
          <LayersControl.BaseLayer name="Satellite">
            <TileLayer
              attribution='&copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
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
          
          <LayersControl.Overlay checked name="Burglary Hotspots">
            <FeatureGroup>
              {burglaryHotspots.map((spot, index) => (
                <CircleMarker
                  key={`hotspot-${index}`}
                  center={spot.position}
                  radius={Math.sqrt(spot.count) * 0.8}
                  fillColor="#ff7800"
                  color="#000"
                  weight={1}
                  opacity={1}
                  fillOpacity={0.6}
                >
                  <Popup>
                    <div>
                      <strong>Burglary Hotspot</strong>
                      <p>{spot.count} incidents reported</p>
                    </div>
                  </Popup>
                </CircleMarker>
              ))}
            </FeatureGroup>
          </LayersControl.Overlay>
          
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
                      <Popup>
                        <div>
                          <strong>Police Officer #{officer.officerId}</strong>
                          <p>Assigned to this location based on K-means clustering</p>
                          <p>Coverage: {officer.coverage} burglary incidents</p>
                        </div>
                      </Popup>
                    </Marker>
                    <CircleMarker
                      center={officer.position}
                      radius={officer.coverage * 1.5}
                      fillColor="#4338ca"
                      color="#4338ca"
                      weight={1}
                      opacity={0.4}
                      fillOpacity={0.1}
                    />
                  </React.Fragment>
                ))}
              </FeatureGroup>
            </LayersControl.Overlay>
          )}
        </LayersControl>
        
        {/* Map Legend */}
        <div className="leaflet-bottom leaflet-left" style={{ zIndex: 1000 }}>
          <div className="leaflet-control leaflet-bar bg-white p-2 shadow-md">
            <h4 className="font-semibold text-sm mb-1">Burglary Intensity</h4>
            <div className="heatmap-legend">
              <div className="heatmap-legend-gradient"></div>
              <span className="heatmap-label">Low</span>
              <span className="heatmap-label ml-auto">High</span>
            </div>
            {showPoliceAllocation && (
              <>
                <div className="mt-2 flex items-center">
                  <div className="h-3 w-3 rounded-full bg-blue-700 mr-1"></div>
                  <span className="text-xs">Police allocation</span>
                </div>
              </>
            )}
          </div>
        </div>
      </MapContainer>
    </div>
  );
};

export default MapComponent; 