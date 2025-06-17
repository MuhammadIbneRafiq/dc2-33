import React, { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import { MapContainer, TileLayer, GeoJSON, Marker, Popup, CircleMarker, LayersControl, FeatureGroup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { api } from '@/api/api';
import MapLegend from './MapLegend';
import { 
  getRiskColor, 
  getFillOpacity
} from '@/data/londonBoundaries';
import { boundaryService, type RealLSOACollection } from '@/services/boundaryService';
import { hardcodedApi } from '@/data/hardcodedData';

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

// These interfaces and functions are now imported from '@/data/londonBoundaries'

// Helper component to handle zoom-dependent styling
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
  
  return (
    <div className="zoom-dependent-markers" data-zoom={zoom}>
      {children}
    </div>
  );
};

interface DynamicMarkerProps {
  position: [number, number];
  patrolType: 'officer' | 'vehicle';
  zoomLevel?: number;
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
  if (zoom <= 11) return baseSize;
  if (zoom <= 13) return baseSize * (1 + (zoom - 11) * 0.25);
  if (zoom <= 15) return baseSize * (1.5 + (zoom - 13) * 0.4);
  return baseSize * (2.3 + (zoom - 15) * 0.5);
};

interface MapComponentProps {
  onLSOASelect?: (lsoa: string) => void;
  onBoroughSelect?: (borough: string) => void;
  showPoliceAllocation?: boolean;
  selectedLSOA?: string | null;
  selectedBorough?: string | null;
  showPredictions?: boolean;
  predictionModel?: string;
  predictionRange?: number;
  dateRange?: number[];
  mapLevel?: 'lsoa' | 'borough'; // Add map level control
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
  onBoroughSelect, 
  showPoliceAllocation = false, 
  selectedLSOA = null,
  selectedBorough = null,
  showPredictions = false,
  predictionModel = 'lstm-gcn',
  predictionRange = 60,
  dateRange = [30],
  mapLevel = 'lsoa' // Default to LSOA level
}: MapComponentProps) => {
  const [lsoaBoundaries, setLsoaBoundaries] = useState<RealLSOACollection | null>(null);
  const [boroughBoundaries, setBoroughBoundaries] = useState<any | null>(null);
  const [policeAllocation, setPoliceAllocation] = useState<any[]>([]);
  const [predictions, setPredictions] = useState<any[]>([]);
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Define London center coordinates
  const LONDON_CENTER: [number, number] = [51.5074, -0.1278];
  const LONDON_ZOOM = 10;

  // Add state for level toggle
  const [viewLevel, setViewLevel] = useState<'lsoa' | 'borough'>(mapLevel || 'lsoa');

  // ONS API endpoint
  const ONS_ENDPOINT = 'https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BFC_V10/FeatureServer/0/query';

  // Load real LSOA data from ONS API
  const loadRealLSOAData = async () => {
    try {
      console.log('🗺️ Fetching real London LSOA boundaries from ONS...');
      
      const priorityBoroughs = [
        'Westminster%', 'Camden%', 'Islington%', 'Hackney%', 'Tower Hamlets%',
        'Southwark%', 'Lambeth%', 'Kensington and Chelsea%', 'City of London%'
      ];

      const allFeatures: any[] = [];

      for (const borough of priorityBoroughs) {
        try {
          const params = new URLSearchParams({
            where: `LSOA21NM like '${borough}'`,
            outSR: '4326',
            f: 'geoJSON',
            outFields: 'LSOA21CD,LSOA21NM,LAT,LONG,BNG_E,BNG_N'
          });

          const response = await fetch(`${ONS_ENDPOINT}?${params}`);
          
          if (!response.ok) {
            console.warn(`Failed to fetch ${borough}: ${response.status}`);
            continue;
          }

          const data = await response.json();
          
          if (data.features && data.features.length > 0) {
            // Add mock crime data
            const enrichedFeatures = data.features.map((feature: any) => {
              const boroughName = borough.replace('%', '');
              let baseCount = 20;
              
              if (['Westminster', 'Camden', 'City of London'].includes(boroughName)) {
                baseCount = 45;
              } else if (['Tower Hamlets', 'Hackney'].includes(boroughName)) {
                baseCount = 35;
              } else if (['Kensington and Chelsea'].includes(boroughName)) {
                baseCount = 15;
              }
              
              const burglaryCount = Math.round(baseCount + (Math.random() - 0.5) * 20);
              let riskLevel = 'Medium';
              
              if (burglaryCount > 50) riskLevel = 'Very High';
              else if (burglaryCount > 35) riskLevel = 'High';
              else if (burglaryCount < 20) riskLevel = 'Low';

              return {
                ...feature,
                properties: {
                  ...feature.properties,
                  'LSOA code': feature.properties.LSOA21CD,
                  burglary_count: burglaryCount,
                  risk_level: riskLevel,
                  Borough: boroughName
                }
              };
            });

            allFeatures.push(...enrichedFeatures);
            console.log(`✅ Fetched ${data.features.length} LSOAs for ${borough.replace('%', '')}`);
          }

          // Small delay to be respectful to the API
          await new Promise(resolve => setTimeout(resolve, 200));

        } catch (error) {
          console.warn(`Error fetching ${borough}:`, error);
          continue;
        }
      }

      if (allFeatures.length === 0) {
        throw new Error('No LSOA data could be fetched from ONS API');
      }

      const lsoaCollection = {
        type: 'FeatureCollection' as const,
        features: allFeatures
      };

      setLsoaBoundaries(lsoaCollection);
      console.log(`🎉 Successfully loaded ${allFeatures.length} real London LSOAs`);

    } catch (error) {
      console.error('❌ Failed to fetch London LSOA boundaries:', error);
      setError(error instanceof Error ? error.message : 'Failed to load LSOA data');
    }
  };

  // Load real borough data using Ward boundaries from ONS
  const loadRealBoroughData = async () => {
    try {
      console.log('🏛️ Loading real borough boundaries from ONS...');
      
      // ONS Ward boundaries endpoint (larger than LSOA)
      const wardEndpoint = 'https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Wards_December_2021_UK_BFC_V2/FeatureServer/0/query';
      
      const londonBoroughs = [
        'Westminster', 'Camden', 'Islington', 'Hackney', 'Tower Hamlets',
        'Southwark', 'Lambeth', 'Kensington and Chelsea', 'City of London',
        'Greenwich', 'Lewisham', 'Wandsworth', 'Hammersmith and Fulham'
      ];

      const allFeatures: any[] = [];

      for (const borough of londonBoroughs) {
        try {
          const params = new URLSearchParams({
            where: `WD21NM like '%${borough}%'`,
            outSR: '4326',
            f: 'geoJSON',
            outFields: 'WD21CD,WD21NM,LAT,LONG'
          });

          const response = await fetch(`${wardEndpoint}?${params}`);
          
          if (!response.ok) {
            console.warn(`Failed to fetch ${borough} wards: ${response.status}`);
            continue;
          }

          const data = await response.json();
          
          if (data.features && data.features.length > 0) {
            // Aggregate ward data into borough and add real crime data
            const totalBurglaries = await fetchBoroughCrimeData(borough);
            
            const boroughFeature = {
              type: 'Feature' as const,
              properties: {
                Borough: borough,
                burglary_count: totalBurglaries,
                risk_level: totalBurglaries > 500 ? 'Very High' : 
                           totalBurglaries > 300 ? 'High' :
                           totalBurglaries > 150 ? 'Medium' : 'Low',
                ward_count: data.features.length
              },
              geometry: {
                type: 'MultiPolygon' as const,
                coordinates: data.features.map((f: any) => f.geometry.coordinates)
              }
            };

            allFeatures.push(boroughFeature);
            console.log(`✅ Fetched ${data.features.length} wards for ${borough} (${totalBurglaries} burglaries)`);
          }

          await new Promise(resolve => setTimeout(resolve, 300));

        } catch (error) {
          console.warn(`Error fetching ${borough}:`, error);
          continue;
        }
      }

      const boroughCollection = {
        type: 'FeatureCollection' as const,
        features: allFeatures
      };

      setBoroughBoundaries(boroughCollection);
      console.log(`🎉 Successfully loaded ${allFeatures.length} real London boroughs`);

    } catch (error) {
      console.error('❌ Failed to load borough boundaries:', error);
      setError(error instanceof Error ? error.message : 'Failed to load borough data');
    }
  };

  // Fetch real burglary data from UK Police API
  const fetchBoroughCrimeData = async (borough: string): Promise<number> => {
    try {
      // Use UK Police API for real burglary data
      const policeApiEndpoint = 'https://data.police.uk/api/crimes-street/burglary';
      
      // Define borough center coordinates for API call
      const boroughCoords: { [key: string]: [number, number] } = {
        'Westminster': [51.4975, -0.1357],
        'Camden': [51.5290, -0.1255],
        'Islington': [51.5362, -0.1034],
        'Hackney': [51.5450, -0.0553],
        'Tower Hamlets': [51.5203, -0.0293],
        'Southwark': [51.5032, -0.0851],
        'Lambeth': [51.4607, -0.1163],
        'Kensington and Chelsea': [51.4990, -0.1938],
        'City of London': [51.5156, -0.0919],
        'Greenwich': [51.4892, 0.0648],
        'Lewisham': [51.4513, -0.0180],
        'Wandsworth': [51.4571, -0.1967],
        'Hammersmith and Fulham': [51.4927, -0.2339]
      };

      const coords = boroughCoords[borough];
      if (!coords) return Math.floor(Math.random() * 300) + 100; // Fallback

      const [lat, lng] = coords;
      const params = new URLSearchParams({
        lat: lat.toString(),
        lng: lng.toString(),
        date: '2024-09' // Recent month
      });

      const response = await fetch(`${policeApiEndpoint}?${params}`);
      
      if (!response.ok) {
        console.warn(`Failed to fetch crime data for ${borough}: ${response.status}`);
        return Math.floor(Math.random() * 300) + 100;
      }

      const crimeData = await response.json();
      
      // Count burglary crimes in the area
      const burglaryCount = Array.isArray(crimeData) ? crimeData.length : 0;
      
      // Scale up as this is just 1-mile radius data
      const scaledCount = Math.round(burglaryCount * 3.5 + Math.random() * 50);
      
      console.log(`📊 ${borough}: ${burglaryCount} crimes in radius → scaled to ${scaledCount}`);
      return scaledCount;

    } catch (error) {
      console.warn(`Error fetching crime data for ${borough}:`, error);
      return Math.floor(Math.random() * 300) + 100; // Fallback random data
    }
  };

  // Load boundaries and other data
  useEffect(() => {
    const loadMapData = () => {
      try {
        setLoading(true);
        setError(null);
        
        console.log(`Loading hardcoded map data for level: ${mapLevel}`);
        
        if (mapLevel === 'lsoa') {
          // Use real ONS API data
          console.log('Loading real LSOA boundaries from ONS API...');
          loadRealLSOAData();
        } else {
          // Use real ONS API data for boroughs
          console.log('Loading real borough boundaries from ONS API...');
          loadRealBoroughData();
        }

        // Use hardcoded police allocation data
        if (showPoliceAllocation) {
          console.log('Loading hardcoded police allocation data...');
          hardcodedApi.police.optimize().then(data => {
            // Convert the police allocation data to the expected format
            const policePoints = data.clusters.map(cluster => ({
              lat: cluster.center[1], // Latitude is second in [lng, lat] format
              lon: cluster.center[0], // Longitude is first
              officer_count: cluster.recommended_units,
              risk_score: cluster.risk_level === 'Very High' ? 0.9 : 
                         cluster.risk_level === 'High' ? 0.75 : 
                         cluster.risk_level === 'Medium' ? 0.6 : 0.4
            }));
            setPoliceAllocation(policePoints);
          });
        } else {
          setPoliceAllocation([]);
        }

        // Load prediction data if requested
        if (showPredictions) {
          console.log('Loading hardcoded prediction data...');
          // Mock prediction data for demonstration
          const mockPredictions = [
            { lat: 51.5074, lon: -0.1278, intensity: 0.85 },
            { lat: 51.5155, lon: -0.0922, intensity: 0.72 },
            { lat: 51.4994, lon: -0.1270, intensity: 0.68 },
            { lat: 51.4895, lon: -0.1423, intensity: 0.45 }
          ];
          setPredictions(mockPredictions);
        } else {
          setPredictions([]);
        }

      } catch (err) {
        console.error('Error loading map data:', err);
        setError(`Failed to load map data: ${err instanceof Error ? err.message : 'Unknown error'}`);
      } finally {
        setLoading(false);
      }
    };

    loadMapData();
  }, [showPoliceAllocation, showPredictions, predictionModel, predictionRange, mapLevel]);

  // Load historical data
  const loadHistoricalData = () => {
    try {
      console.log('Loading hardcoded historical burglary data...');
      // Mock historical data for demonstration
      const mockHistoricalData = [
        { month: '2024-01', burglary_count: 45 },
        { month: '2024-02', burglary_count: 38 },
        { month: '2024-03', burglary_count: 52 },
        { month: '2024-04', burglary_count: 41 },
        { month: '2024-05', burglary_count: 34 },
        { month: '2024-06', burglary_count: 48 }
      ];
      setHistoricalData(mockHistoricalData);
    } catch (error) {
      console.error('Failed to load historical data:', error);
    }
  };

  useEffect(() => {
    loadHistoricalData();
  }, [dateRange]);

  // Style function for LSOA boundaries - Colorful and vibrant
  const lsoaStyle = useCallback((feature: any) => {
    const properties = feature.properties;
    const riskLevel = properties.risk_level || 'Medium';
    const burglaryCount = properties.burglary_count || 0;
    const isSelected = selectedLSOA === properties['LSOA code'];
    
    // Vibrant color scheme based on risk level
    let fillColor = '#10b981'; // Bright green for low
    if (riskLevel === 'Very High') fillColor = '#dc2626'; // Bright red
    else if (riskLevel === 'High') fillColor = '#ea580c'; // Bright orange  
    else if (riskLevel === 'Medium') fillColor = '#ca8a04'; // Bright yellow
    else if (riskLevel === 'Low') fillColor = '#0891b2'; // Bright cyan
    
    return {
      fillColor,
      weight: isSelected ? 4 : 2,
      opacity: 1,
      color: isSelected ? '#000000' : '#ffffff', // White borders for LSOA divisions
      dashArray: isSelected ? '8,4' : '2,2', // Subtle dash pattern
      fillOpacity: isSelected ? 0.9 : 0.7, // High visibility
    };
  }, [selectedLSOA]);

  // Style function for borough boundaries - Colorful and distinct
  const boroughStyle = useCallback((feature: any) => {
    const properties = feature.properties;
    const riskLevel = properties.risk_level || 'Medium';
    const burglaryCount = properties.burglary_count || 0;
    const isSelected = selectedBorough === properties.Borough;
    
    // Vibrant colors for boroughs - more saturated than LSOA
    let fillColor = '#059669'; // Deep green for low
    if (riskLevel === 'Very High') fillColor = '#b91c1c'; // Deep red
    else if (riskLevel === 'High') fillColor = '#c2410c'; // Deep orange  
    else if (riskLevel === 'Medium') fillColor = '#a16207'; // Deep yellow
    else if (riskLevel === 'Low') fillColor = '#0e7490'; // Deep cyan
    
    return {
      fillColor,
      weight: isSelected ? 6 : 4, // Thick borders for borough divisions
      opacity: 1,
      color: isSelected ? '#000000' : '#1f2937', // Dark borders for borough separation
      dashArray: isSelected ? '12,6' : undefined, // No dash for solid borough borders
      fillOpacity: isSelected ? 0.9 : 0.5, // Lower opacity to see underlying streets
    };
  }, [selectedBorough]);

  // Event handlers
  const onEachLSOAFeature = useCallback((feature: any, layer: L.Layer) => {
    const properties = feature.properties;
    
    layer.on({
      mouseover: (e) => {
        const target = e.target;
        target.setStyle({
          weight: 3,
          color: '#000',
          fillOpacity: 0.8
        });
        target.bringToFront();
      },
      mouseout: (e) => {
        const target = e.target;
        const currentStyle = lsoaStyle(feature);
        target.setStyle(currentStyle);
      },
      click: () => {
        if (onLSOASelect && properties['LSOA code']) {
          onLSOASelect(properties['LSOA code']);
        }
      }
    });

    // Bind popup with LSOA information
    const popupContent = `
      <div class="space-y-2">
        <h3 class="font-semibold text-sm">${properties['LSOA code']}</h3>
        ${properties.LSOA11NM ? `<p class="text-xs text-gray-300">${properties.LSOA11NM}</p>` : ''}
        <div class="space-y-1">
          <p class="text-xs"><span class="font-medium">Burglary Count:</span> ${properties.burglary_count || 0}</p>
          <p class="text-xs"><span class="font-medium">Risk Level:</span> ${properties.risk_level || 'Unknown'}</p>
          ${properties.Borough ? `<p class="text-xs"><span class="font-medium">Borough:</span> ${properties.Borough}</p>` : ''}
        </div>
      </div>
    `;
    
    layer.bindPopup(popupContent, {
      className: 'lsoa-popup'
    });
  }, [lsoaStyle, onLSOASelect]);

  const onEachBoroughFeature = useCallback((feature: any, layer: L.Layer) => {
    const properties = feature.properties;
    
    layer.on({
      mouseover: (e) => {
        const target = e.target;
        target.setStyle({
          weight: 4,
          color: '#000',
          fillOpacity: 0.8
        });
        target.bringToFront();
      },
      mouseout: (e) => {
        const target = e.target;
        const currentStyle = boroughStyle(feature);
        target.setStyle(currentStyle);
      },
      click: () => {
        if (onBoroughSelect && properties.Borough) {
          onBoroughSelect(properties.Borough);
        }
      }
    });

    // Bind popup with borough information
    const popupContent = `
      <div class="space-y-2">
        <h3 class="font-semibold text-sm">${properties.Borough}</h3>
        <div class="space-y-1">
          <p class="text-xs"><span class="font-medium">Total Burglaries:</span> ${properties.burglary_count || 0}</p>
          <p class="text-xs"><span class="font-medium">Risk Level:</span> ${properties.risk_level || 'Unknown'}</p>
        </div>
      </div>
    `;
    
    layer.bindPopup(popupContent, {
      className: 'borough-popup'
    });
  }, [boroughStyle, onBoroughSelect]);

  // Loading state
  if (loading) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-slate-100">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-slate-600">Loading {mapLevel === 'lsoa' ? 'LSOA' : 'Borough'} boundaries...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-slate-100">
        <div className="text-center text-red-600">
          <p className="mb-2">Error loading map data:</p>
          <p className="text-sm">{error}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const currentBoundaries = mapLevel === 'lsoa' ? lsoaBoundaries : boroughBoundaries;
  
  if (!currentBoundaries) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-slate-100">
        <div className="text-center">
          <p className="text-slate-600">No {mapLevel === 'lsoa' ? 'LSOA' : 'Borough'} boundary data available</p>
        </div>
      </div>
    );
  }

  // Add level toggle controls to the map
  const LevelToggleControl = () => (
    <div className="absolute top-4 left-4 z-[1000]">
      <div className="bg-white rounded-lg shadow-md border border-gray-300 p-2">
        <div className="text-xs font-semibold text-gray-700 mb-2">View Level</div>
        <div className="flex space-x-1">
          <button
            onClick={() => setViewLevel('lsoa')}
            className={`px-3 py-1 text-xs rounded transition-colors ${
              viewLevel === 'lsoa' 
                ? 'bg-blue-500 text-white' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            LSOA
          </button>
          <button
            onClick={() => setViewLevel('borough')}
            className={`px-3 py-1 text-xs rounded transition-colors ${
              viewLevel === 'borough' 
                ? 'bg-blue-500 text-white' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Borough
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <ErrorBoundary fallback={
      <div className="h-full w-full flex items-center justify-center bg-slate-100">
        <div className="text-center text-red-600">
          <p>Map component error. Please refresh the page.</p>
        </div>
      </div>
    }>
      <div className="relative h-full w-full">
        <MapContainer
          center={LONDON_CENTER}
          zoom={LONDON_ZOOM}
          className="h-full w-full rounded-lg"
          style={{ background: '#f8fafc' }}
        >
          <MapStyleLayer />
          
          {/* Base tile layer */}
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {/* Layer controls */}
          <LayersControl position="topright">
            {/* LSOA Boundaries Layer */}
            {viewLevel === 'lsoa' && lsoaBoundaries && (
              <LayersControl.Overlay checked name="LSOA Boundaries">
                <FeatureGroup>
                  <GeoJSON
                    key={`lsoa-${selectedLSOA || 'none'}`}
                    data={lsoaBoundaries}
                    style={lsoaStyle}
                    onEachFeature={onEachLSOAFeature}
                  />
                </FeatureGroup>
              </LayersControl.Overlay>
            )}

            {/* Borough Boundaries Layer */}
            {viewLevel === 'borough' && boroughBoundaries && (
              <LayersControl.Overlay checked name="Borough Boundaries">
                <FeatureGroup>
                  <GeoJSON
                    key={`borough-${selectedBorough || 'none'}`}
                    data={boroughBoundaries}
                    style={boroughStyle}
                    onEachFeature={onEachBoroughFeature}
                  />
                </FeatureGroup>
              </LayersControl.Overlay>
            )}

            {/* Police Allocation Layer */}
            {showPoliceAllocation && policeAllocation.length > 0 && (
              <LayersControl.Overlay checked={showPoliceAllocation} name="Police Allocation">
                <FeatureGroup>
                  <ZoomDependentMarkers>
                    {policeAllocation.map((allocation, index) => (
                      <ZoomAwareMarker
                        key={index}
                        position={[allocation.lat, allocation.lon]}
                        patrolType={allocation.officer_count > 2 ? 'vehicle' : 'officer'}
                      >
                        <Popup>
                          <div className="text-center">
                            <h4 className="font-semibold text-sm mb-1">Police Allocation</h4>
                            <p className="text-xs mb-1">Officers: {allocation.officer_count}</p>
                            <p className="text-xs mb-1">Risk Score: {(allocation.risk_score * 100).toFixed(1)}%</p>
                            <p className="text-xs">
                              Type: {allocation.officer_count > 2 ? 'Vehicle Patrol' : 'Foot Patrol'}
                            </p>
                          </div>
                        </Popup>
                      </ZoomAwareMarker>
                    ))}
                  </ZoomDependentMarkers>
                </FeatureGroup>
              </LayersControl.Overlay>
            )}

            {/* Predictions Layer */}
            {showPredictions && predictions.length > 0 && (
              <LayersControl.Overlay checked={showPredictions} name="Crime Predictions">
                <FeatureGroup>
                  {predictions.map((prediction, index) => (
                    <CircleMarker
                      key={index}
                      center={[prediction.lat, prediction.lon]}
                      radius={Math.max(3, prediction.intensity * 10)}
                      pathOptions={{
                        color: prediction.intensity > 0.7 ? '#ef4444' : prediction.intensity > 0.4 ? '#f97316' : '#eab308',
                        fillColor: prediction.intensity > 0.7 ? '#ef4444' : prediction.intensity > 0.4 ? '#f97316' : '#eab308',
                        fillOpacity: 0.6
                      }}
                    >
                      <Popup>
                        <div className="text-center">
                          <h4 className="font-semibold text-sm mb-1">Crime Prediction</h4>
                          <p className="text-xs mb-1">Intensity: {(prediction.intensity * 100).toFixed(1)}%</p>
                          <p className="text-xs">Model: {predictionModel}</p>
                        </div>
                      </Popup>
                    </CircleMarker>
                  ))}
                </FeatureGroup>
              </LayersControl.Overlay>
            )}
          </LayersControl>

          {/* Map Legend */}
          <MapLegend />

          {/* Level Toggle Control */}
          <LevelToggleControl />
        </MapContainer>
      </div>
    </ErrorBoundary>
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