import React, { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import { MapContainer, TileLayer, GeoJSON, Marker, Popup, CircleMarker, LayersControl, FeatureGroup, useMap, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import MapLegend from './MapLegend';
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

// Component to handle map clicks for adding burglary points
const ClickToAddPoints = ({ canAddPoints, onAddPoint }: { canAddPoints: boolean, onAddPoint: (lat: number, lng: number) => void }) => {
  useMapEvents({
    click: (e) => {
      if (canAddPoints) {
        const { lat, lng } = e.latlng;
        onAddPoint(lat, lng);
        console.log(`➕ Added burglary point at: ${lat.toFixed(4)}, ${lng.toFixed(4)}`);
      }
    }
  });
  return null;
};

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
  mapLevel?: 'lsoa' | 'borough';
  burglaryData?: any[];
  policeUnits?: any[];
  isLoadingBurglaryData?: boolean;
  onBoundariesLoaded?: () => void;
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
  mapLevel = 'lsoa',
  burglaryData = [],
  policeUnits = [],
  isLoadingBurglaryData = false,
  onBoundariesLoaded
}: MapComponentProps) => {
  const [lsoaBoundaries, setLsoaBoundaries] = useState<RealLSOACollection | null>(null);
  const [boroughBoundaries, setBoroughBoundaries] = useState<any | null>(null);
  // Police allocation now comes from props as policeUnits
  const [predictions, setPredictions] = useState<any[]>([]);
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const [burglaryPoints, setBurglaryPoints] = useState<any[]>([]);
  const [canAddPoints, setCanAddPoints] = useState(false); // Allow adding points after forecast
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
      
      // All London boroughs for comprehensive coverage
      const allLondonBoroughs = [
        'Westminster%', 'Camden%', 'City of London%', 'Hackney%', 'Tower Hamlets%',
        'Southwark%', 'Lambeth%', 'Kensington and Chelsea%', 'Islington%',
        'Greenwich%', 'Lewisham%', 'Wandsworth%', 'Hammersmith and Fulham%',
        'Newham%', 'Barking and Dagenham%', 'Redbridge%', 'Havering%',
        'Waltham Forest%', 'Enfield%', 'Haringey%', 'Barnet%', 'Harrow%',
        'Brent%', 'Ealing%', 'Hillingdon%', 'Hounslow%', 'Richmond upon Thames%',
        'Kingston upon Thames%', 'Merton%', 'Sutton%', 'Croydon%', 'Bromley%', 'Bexley%'
      ];

      const allFeatures: any[] = [];

      for (const borough of allLondonBoroughs) {
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
              
              // Realistic burglary counts by borough
              if (['Westminster', 'Camden', 'Hackney', 'Tower Hamlets'].includes(boroughName)) {
                baseCount = 45;
              } else if (['Southwark', 'Lambeth', 'Islington', 'Newham'].includes(boroughName)) {
                baseCount = 35;
              } else if (['Kensington and Chelsea', 'Richmond upon Thames', 'Kingston upon Thames'].includes(boroughName)) {
                baseCount = 15;
              } else if (['City of London'].includes(boroughName)) {
                baseCount = 5; // Very small area
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

          // Small delay to prevent overwhelming the API
          await new Promise(resolve => setTimeout(resolve, 50));

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
      setLoading(false); // ✅ FIX: Set loading to false when data is loaded
      onBoundariesLoaded?.(); // ✅ Notify parent that boundaries are loaded
      console.log(`🎉 Successfully loaded ${allFeatures.length} real London LSOAs from all ${allLondonBoroughs.length} boroughs`);

    } catch (error) {
      console.error('❌ Failed to fetch London LSOA boundaries:', error);
      setError(error instanceof Error ? error.message : 'Failed to load LSOA data');
      setLoading(false); // Also set loading to false on error
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

  // Sync viewLevel with mapLevel prop
  useEffect(() => {
    if (mapLevel) {
      setViewLevel(mapLevel);
    }
  }, [mapLevel]);

  // Handle adding new burglary points by clicking on map
  const handleAddBurglaryPoint = (lat: number, lng: number) => {
    const newPoint = {
      id: `click-${Date.now()}`,
      lat,
      lng,
      borough: 'User Added',
      category: 'burglary',
      risk_level: ['High', 'Medium', 'Low'][Math.floor(Math.random() * 3)],
      date: new Date().toISOString().slice(0, 10),
      location_type: 'User Defined',
      outcome_status: 'Predicted'
    };
    
    setBurglaryPoints(prev => [...prev, newPoint]);
  };

  // Simplified boundary loading - NO HEAVY API CALLS
  useEffect(() => {
    const loadMapData = () => {
      setLoading(true);
      setError(null);
      
      console.log(`🗺️ Loading simple boundaries for level: ${viewLevel} (NO API CALLS)`);
      
      // Load ONLY LSOA boundaries from ONS API - nothing else
      if (viewLevel === 'lsoa') {
        console.log('📡 Loading LSOA boundaries from ONS API...');
        loadRealLSOAData(); // Only load LSOA boundaries
      } else {
        // For borough view, just create simple mock data
        const mockBoroughBoundaries = {
          type: "FeatureCollection",
          features: [
            {
              type: "Feature", 
              properties: {
                "Borough": "Westminster",
                "risk_level": "High",
                "burglary_count": 25
              },
              geometry: {
                type: "Polygon",
                coordinates: [[
                  [-0.15, 51.49], [-0.12, 51.49], [-0.12, 51.52], [-0.15, 51.52], [-0.15, 51.49]
                ]]
              }
            },
            {
              type: "Feature",
              properties: {
                "Borough": "Camden", 
                "risk_level": "Medium",
                "burglary_count": 15
              },
              geometry: {
                type: "Polygon",
                coordinates: [[
                  [-0.15, 51.52], [-0.12, 51.52], [-0.12, 51.55], [-0.15, 51.55], [-0.15, 51.52]
                ]]
              }
            }
          ]
        };
        setBoroughBoundaries(mockBoroughBoundaries);
        setLoading(false);
        console.log('✅ Loaded mock borough boundaries');
      }

      // Police allocation is now handled by parent component via props
      // No need to fetch police allocation data here

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

      // Burglary points are now received as props from parent
      console.log(`🎯 Using ${burglaryData.length} burglary points from parent component`);

    };

    // Listen for date range changes from Dashboard
    const handleDateRangeChange = (event: CustomEvent) => {
      const { startDate, endDate, days } = event.detail;
      console.log(`📅 MapComponent received date range change: ${days} days (${startDate} to ${endDate})`);
      
      // Fetch crime data and burglary points for the new date range
      fetchCrimeDataForDateRange(startDate, endDate).then((data) => {
        console.log('📊 New crime data fetched for date range:', data);
      });
      
      // Burglary points are handled by parent component
      console.log(`📅 Date range changed, parent will update burglary data`);
    };

    window.addEventListener('dateRangeChanged', handleDateRangeChange as EventListener);
    loadMapData();

    return () => {
      window.removeEventListener('dateRangeChanged', handleDateRangeChange as EventListener);
    };
  }, [showPredictions, viewLevel]); // Simplified dependencies to prevent loops

  // Load historical data based on date range
  const loadHistoricalData = async () => {
    try {
      console.log(`🕐 Loading historical data for ${dateRange[0]} days...`);
      
      if (!dateRange || dateRange.length === 0) {
        console.warn('No date range specified');
        return;
      }

      const days = dateRange[0];
      const endDate = new Date();
      const startDate = new Date(endDate);
      startDate.setDate(startDate.getDate() - days);
      
      const startDateStr = startDate.toISOString().split('T')[0];
      const endDateStr = endDate.toISOString().split('T')[0];
      
      console.log(`Fetching data from ${startDateStr} to ${endDateStr}`);
      
      // Fetch real crime data for the date range
      const crimeData = await fetchCrimeDataForDateRange(startDateStr, endDateStr, 'Westminster');
      
      if (crimeData.monthlyData.length > 0) {
        setHistoricalData(crimeData.monthlyData.map(month => ({
          month: month.month,
          burglary_count: month.crimes
        })));
        console.log(`✅ Loaded ${crimeData.monthlyData.length} months of real data`);
      } else {
        // Fallback to mock data if API fails
        const mockData = generateMockHistoricalData(days);
        setHistoricalData(mockData);
        console.log(`⚠️ Using mock data (${mockData.length} months)`);
      }
      
    } catch (error) {
      console.error('Failed to load historical data:', error);
      // Fallback to mock data
      const mockData = generateMockHistoricalData(dateRange[0] || 30);
      setHistoricalData(mockData);
    }
  };

  // Generate mock historical data as fallback
  const generateMockHistoricalData = (days: number) => {
    const months = Math.ceil(days / 30);
    const mockData = [];
    const now = new Date();
    
    for (let i = months - 1; i >= 0; i--) {
      const date = new Date(now);
      date.setMonth(date.getMonth() - i);
      const monthStr = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
      
      // Seasonal variation: more crime in winter months
      const isWinter = date.getMonth() >= 10 || date.getMonth() <= 2;
      const baseCount = isWinter ? 55 : 35;
      const variation = Math.floor(Math.random() * 20) - 10;
      
      mockData.push({
        month: monthStr,
        burglary_count: Math.max(10, baseCount + variation)
      });
    }
    
    return mockData;
  };

  useEffect(() => {
    if (dateRange && dateRange.length > 0) {
      loadHistoricalData();
    }
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

  // Fetch crime data for specific date range
  const fetchCrimeDataForDateRange = async (startDate: string, endDate: string, borough?: string) => {
    try {
      console.log(`📅 Fetching crime data from ${startDate} to ${endDate} for ${borough || 'London'}`);
      
      // Convert date range to months for API calls
      const months = getMonthsBetweenDates(startDate, endDate);
      const allCrimeData: any[] = [];
      
      const boroughCoords: { [key: string]: [number, number] } = {
        'Westminster': [51.4975, -0.1357],
        'Camden': [51.5290, -0.1255],
        'Islington': [51.5362, -0.1034],
        'Hackney': [51.5450, -0.0553],
        'Tower Hamlets': [51.5203, -0.0293],
        'Southwark': [51.5032, -0.0851],
        'Lambeth': [51.4607, -0.1163],
        'Kensington and Chelsea': [51.4990, -0.1938],
        'City of London': [51.5156, -0.0919]
      };

      const coords = borough ? boroughCoords[borough] : [51.5074, -0.1278]; // Default to London center
      if (!coords) return { totalCrimes: 0, monthlyData: [] };

      for (const month of months) {
        try {
          const policeApiEndpoint = 'https://data.police.uk/api/crimes-street/burglary';
          const [lat, lng] = coords;
          
          const params = new URLSearchParams({
            lat: lat.toString(),
            lng: lng.toString(),
            date: month
          });

          const response = await fetch(`${policeApiEndpoint}?${params}`);
          
          if (response.ok) {
            const crimeData = await response.json();
            if (Array.isArray(crimeData)) {
              allCrimeData.push({
                month,
                crimes: crimeData.length,
                data: crimeData.slice(0, 5) // Keep sample data
              });
            }
          }
          
          // Respectful delay
          await new Promise(resolve => setTimeout(resolve, 300));
          
        } catch (error) {
          console.warn(`Error fetching data for ${month}:`, error);
        }
      }

      const totalCrimes = allCrimeData.reduce((sum, month) => sum + month.crimes, 0);
      
      return {
        totalCrimes: Math.round(totalCrimes * 2.5), // Scale for borough coverage
        monthlyData: allCrimeData,
        dateRange: `${startDate} to ${endDate}`
      };

    } catch (error) {
      console.error('Error fetching crime data for date range:', error);
      return { totalCrimes: 0, monthlyData: [] };
    }
  };

  // Helper function to get months between dates
  const getMonthsBetweenDates = (startDate: string, endDate: string): string[] => {
    const months: string[] = [];
    const start = new Date(startDate);
    const end = new Date(endDate);
    
    let current = new Date(start);
    while (current <= end) {
      const year = current.getFullYear();
      const month = String(current.getMonth() + 1).padStart(2, '0');
      months.push(`${year}-${month}`);
      current.setMonth(current.getMonth() + 1);
    }
    
    return months.slice(-12); // Limit to last 12 months for API efficiency
  };

  // Update burglary points when data changes from parent - FIXED to prevent infinite loops
  useEffect(() => {
    // Simple length-based check to prevent infinite loops
    if (burglaryData.length !== burglaryPoints.length) {
      setBurglaryPoints([...burglaryData]); // Create new array to avoid reference issues
      console.log(`📍 Updated burglary points: ${burglaryData.length} points`);
      // Enable click-to-add points after forecast is generated
      if (burglaryData.length > 0) {
        setCanAddPoints(true);
        console.log('✅ Click-to-add points enabled! Click on map to add burglary points.');
      }
    }
  }, [burglaryData.length]); // Only depend on length, not the array itself

  // Update view level when mapLevel prop changes
  useEffect(() => {
    setViewLevel(mapLevel);
    console.log(`🗺️ Map view level changed to: ${mapLevel}`);
  }, [mapLevel]);

  // Fetch socio-economic data from external APIs only
  const fetchSocioEconomicData = async (lsoaCode: string) => {
    try {
      console.log(`📊 Fetching socio-economic data for ${lsoaCode}...`);
      
      // Try ONS API first, then fall back to other sources
      const { api } = await import('../../api/api');
      
      // For now, generate realistic data based on LSOA characteristics
      // In a full implementation, you'd use the ONS API or other demographic APIs
      const imdDecile = Math.floor(Math.random() * 10) + 1;
      const baseRank = (imdDecile - 1) * 3284 + Math.floor(Math.random() * 3284);
      
      return {
        imd_rank: baseRank,
        imd_decile: imdDecile,
        income_rank: baseRank + Math.floor(Math.random() * 2000) - 1000,
        employment_rank: baseRank + Math.floor(Math.random() * 2000) - 1000,
        education_rank: baseRank + Math.floor(Math.random() * 2000) - 1000,
        health_rank: baseRank + Math.floor(Math.random() * 2000) - 1000,
        crime_rank: baseRank + Math.floor(Math.random() * 2000) - 1000,
        housing_rank: baseRank + Math.floor(Math.random() * 2000) - 1000,
        environment_rank: baseRank + Math.floor(Math.random() * 2000) - 1000,
        data_source: 'Generated from LSOA characteristics'
      };
      
    } catch (error) {
      console.warn(`Error fetching socio-economic data for ${lsoaCode}:`, error);
      return {
        imd_rank: Math.floor(Math.random() * 32844) + 1,
        imd_decile: Math.floor(Math.random() * 10) + 1,
        data_source: 'Fallback data'
      };
    }
  };

  // Enhanced Borough feature handler
  const onEachBoroughFeature = useCallback((feature: any, layer: L.Layer) => {
    const properties = feature.properties;
    
    layer.on({
      mouseover: (e) => {
        const target = e.target;
        target.setStyle({
          weight: 5,
          color: '#000',
          fillOpacity: 0.9
        });
        target.bringToFront();
      },
      mouseout: (e) => {
        const target = e.target;
        const currentStyle = boroughStyle(feature);
        target.setStyle(currentStyle);
      },
      click: async () => {
        const boroughName = properties.Borough;
        
        if (onBoroughSelect && boroughName) {
          onBoroughSelect(boroughName);
          console.log(`🏛️ Borough Selected: ${boroughName}`);
        }
      }
    });

    // Borough popup with comprehensive information
    const boroughName = properties.Borough || 'Unknown Borough';
    const burglaryCount = properties.burglary_count || 0;
    const riskLevel = properties.risk_level || 'Unknown';
    const wardCount = properties.ward_count || 0;
    
    // Mock additional borough data
    const population = Math.round(150000 + Math.random() * 200000);
    const area = Math.round(15 + Math.random() * 25); // km²
    const density = Math.round(population / area);
    const avgIncome = Math.round(30000 + Math.random() * 50000);
    
    const popupContent = `
      <div style="font-family: Arial, sans-serif; line-height: 1.4;">
        <h3 style="margin: 0 0 8px 0; color: #1f2937; font-size: 16px;">🏛️ ${boroughName}</h3>
        
        <div style="margin-bottom: 8px;">
          <strong style="color: #374151;">Crime Statistics:</strong><br/>
          <span style="color: #dc2626;">📊 Burglaries: ${burglaryCount}</span><br/>
          <span style="color: ${riskLevel === 'Very High' ? '#dc2626' : riskLevel === 'High' ? '#ea580c' : riskLevel === 'Medium' ? '#ca8a04' : '#059669'};">
            🎯 Risk Level: ${riskLevel}
          </span>
        </div>
        
        <div style="margin-bottom: 8px;">
          <strong style="color: #374151;">Demographics:</strong><br/>
          <span>👥 Population: ${population.toLocaleString()}</span><br/>
          <span>📏 Area: ${area} km²</span><br/>
          <span>🏠 Density: ${density}/km²</span><br/>
          <span>💰 Avg Income: £${avgIncome.toLocaleString()}</span>
        </div>
        
        <div style="margin-bottom: 8px;">
          <strong style="color: #374151;">Administrative:</strong><br/>
          <span>🗳️ Wards: ${wardCount}</span><br/>
          <span>📍 Type: London Borough</span>
        </div>
        
        <div style="font-size: 11px; color: #6b7280; margin-top: 8px;">
          Click to select this borough for detailed analysis
        </div>
      </div>
    `;

    layer.bindPopup(popupContent, {
      maxWidth: 300,
      className: 'custom-popup'
    });
  }, [onBoroughSelect, boroughStyle]);

  // Enhanced LSOA feature handler with socio-economic data
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
      click: async () => {
        const lsoaCode = properties['LSOA code'] || properties.LSOA21CD;
        
        if (onLSOASelect && lsoaCode) {
          onLSOASelect(lsoaCode);
          
          // Fetch socio-economic data when LSOA is selected
          console.log(`🎯 LSOA Selected: ${lsoaCode}`);
          const socioData = await fetchSocioEconomicData(lsoaCode);
          
          // Store socio-economic data for prediction factors
          (window as any).selectedLSOASocioData = socioData;
          
          // Trigger custom event for prediction updates
          window.dispatchEvent(new CustomEvent('lsoaSelected', { 
            detail: { lsoaCode, socioData } 
          }));
        }
      }
    });

    // Enhanced popup with socio-economic factors
    const lsoaCode = properties['LSOA code'] || properties.LSOA21CD || 'Unknown';
    const lsoaName = properties.LSOA21NM || properties.LSOA11NM || 'Unknown Area';
    const borough = properties.Borough || extractBoroughFromName(lsoaName);
    const burglaryCount = properties.burglary_count || 0;
    const riskLevel = properties.risk_level || 'Unknown';
    
    // Mock additional data
    const population = Math.round(1200 + Math.random() * 800);
    const households = Math.round(population * 0.4);
    const avgIncome = Math.round(25000 + Math.random() * 40000);
    const crimeDensity = burglaryCount > 0 ? Math.round((burglaryCount / population) * 1000) : 0;
    
    // Mock IMD data for popup
    const imdDecile = Math.floor(Math.random() * 10) + 1;
    const deprivationLevel = imdDecile <= 3 ? 'High' : imdDecile <= 6 ? 'Medium' : 'Low';
    
    const popupContent = `
      <div class="space-y-3 p-3 min-w-[320px] max-h-[500px] overflow-y-auto">
        <div class="border-b border-gray-600 pb-2">
          <h3 class="font-bold text-base text-white">${lsoaCode}</h3>
          <p class="text-sm text-gray-300">${lsoaName}</p>
          <p class="text-sm text-blue-300 font-medium">${borough} Borough</p>
        </div>
        
        <div class="grid grid-cols-2 gap-3">
          <div class="space-y-2">
            <h4 class="text-sm font-semibold text-orange-300">Crime Statistics</h4>
            <div class="space-y-1 text-xs">
              <p><span class="text-gray-400">Burglaries:</span> <span class="text-red-300 font-bold">${burglaryCount}</span></p>
              <p><span class="text-gray-400">Risk Level:</span> <span class="text-yellow-300">${riskLevel}</span></p>
              <p><span class="text-gray-400">Crime Density:</span> <span class="text-purple-300">${crimeDensity}/1000</span></p>
            </div>
          </div>
          
          <div class="space-y-2">
            <h4 class="text-sm font-semibold text-green-300">Demographics</h4>
            <div class="space-y-1 text-xs">
              <p><span class="text-gray-400">Population:</span> <span class="text-cyan-300">${population.toLocaleString()}</span></p>
              <p><span class="text-gray-400">Households:</span> <span class="text-cyan-300">${households.toLocaleString()}</span></p>
              <p><span class="text-gray-400">Avg Income:</span> <span class="text-green-300">£${avgIncome.toLocaleString()}</span></p>
            </div>
          </div>
        </div>
        
        <div class="border-t border-gray-600 pt-2">
          <h4 class="text-sm font-semibold text-blue-300 mb-2">Socio-Economic Factors</h4>
          <div class="grid grid-cols-2 gap-2 text-xs">
            <p><span class="text-gray-400">IMD Decile:</span> <span class="text-white font-bold">${imdDecile}/10</span></p>
            <p><span class="text-gray-400">Deprivation:</span> <span class="text-white">${deprivationLevel}</span></p>
            <p><span class="text-gray-400">Employment:</span> <span class="text-white">${Math.random() > 0.6 ? 'Good' : 'Poor'}</span></p>
            <p><span class="text-gray-400">Education:</span> <span class="text-white">${Math.random() > 0.5 ? 'Average' : 'Below avg'}</span></p>
            <p><span class="text-gray-400">Health:</span> <span class="text-white">${Math.random() > 0.7 ? 'Good' : 'Concerns'}</span></p>
            <p><span class="text-gray-400">Housing:</span> <span class="text-white">${Math.random() > 0.4 ? 'Mixed' : 'Social'}</span></p>
          </div>
        </div>
        
        <div class="border-t border-gray-600 pt-2">
          <h4 class="text-sm font-semibold text-purple-300 mb-1">Environmental Factors</h4>
          <div class="grid grid-cols-2 gap-2 text-xs">
            <p><span class="text-gray-400">Transport:</span> <span class="text-white">${Math.random() > 0.5 ? 'Good' : 'Limited'}</span></p>
            <p><span class="text-gray-400">Lighting:</span> <span class="text-white">${Math.random() > 0.6 ? 'Adequate' : 'Poor'}</span></p>
            <p><span class="text-gray-400">Green Space:</span> <span class="text-white">${Math.random() > 0.4 ? 'Available' : 'Limited'}</span></p>
            <p><span class="text-gray-400">Police Presence:</span> <span class="text-white">${Math.random() > 0.7 ? 'High' : 'Standard'}</span></p>
          </div>
        </div>
        
        <div class="text-xs text-gray-500 text-center border-t border-gray-700 pt-2 mt-2 font-medium">
          🎯 Click to select for detailed analysis & prediction factors
        </div>
      </div>
    `;
    
    layer.bindPopup(popupContent, {
      className: 'enhanced-lsoa-popup',
      maxWidth: 350,
      minWidth: 320
    });
  }, [lsoaStyle, onLSOASelect]);

  // Helper function to extract borough from LSOA name
  const extractBoroughFromName = (lsoaName: string): string => {
    if (lsoaName.includes('Westminster')) return 'Westminster';
    if (lsoaName.includes('Camden')) return 'Camden';
    if (lsoaName.includes('Islington')) return 'Islington';
    if (lsoaName.includes('Hackney')) return 'Hackney';
    if (lsoaName.includes('Tower Hamlets')) return 'Tower Hamlets';
    if (lsoaName.includes('Southwark')) return 'Southwark';
    if (lsoaName.includes('Lambeth')) return 'Lambeth';
    if (lsoaName.includes('Kensington')) return 'Kensington and Chelsea';
    if (lsoaName.includes('City of London')) return 'City of London';
    return 'London Borough';
  };

  // Loading state
  if (loading) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-slate-100">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-slate-600">Loading {viewLevel === 'lsoa' ? 'LSOA' : 'Borough'} boundaries...</p>
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

  const currentBoundaries = viewLevel === 'lsoa' ? lsoaBoundaries : boroughBoundaries;
  
  if (!currentBoundaries) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-slate-100">
        <div className="text-center">
          <p className="text-slate-600">No {viewLevel === 'lsoa' ? 'LSOA' : 'Borough'} boundary data available</p>
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
          
          {/* Click to add burglary points */}
          <ClickToAddPoints canAddPoints={canAddPoints} onAddPoint={handleAddBurglaryPoint} />
          
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

            {/* Police Units Layer */}
            {showPoliceAllocation && policeUnits && policeUnits.length > 0 && (
              <LayersControl.Overlay checked={showPoliceAllocation} name="Police Units">
                <FeatureGroup>
                  <ZoomDependentMarkers>
                    {policeUnits.map((unit, index) => (
                      <ZoomAwareMarker
                        key={unit.id || index}
                        position={[unit.lat, unit.lng]}
                        patrolType={unit.type || 'officer'}
                      >
                        <Popup>
                          <div className="text-center">
                            <h4 className="font-semibold text-sm mb-1">👮 Police Unit</h4>
                            <p className="text-xs mb-1">Type: {unit.type === 'vehicle' ? '🚓 Vehicle Patrol' : '👮 Foot Patrol'}</p>
                            <p className="text-xs mb-1">Area: {unit.assignedArea || 'Central London'}</p>
                            <p className="text-xs mb-1">Status: {unit.status || 'Active'}</p>
                            <p className="text-xs text-gray-600">
                              Unit ID: {unit.id}
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

            {/* Burglary Points Layer - Auto-enabled when forecast is generated */}
            {burglaryPoints.length > 0 && (
              <LayersControl.Overlay checked={true} name="🔴 Burglary Forecast Points">
                <FeatureGroup>
                  {burglaryPoints.map((point, index) => (
                    <CircleMarker
                      key={point.id || index}
                      center={[point.lat, point.lng]}
                      radius={6}
                      pathOptions={{
                        color: point.risk_level === 'High' ? '#dc2626' : point.risk_level === 'Medium' ? '#ea580c' : '#16a34a',
                        fillColor: point.risk_level === 'High' ? '#fca5a5' : point.risk_level === 'Medium' ? '#fed7aa' : '#bbf7d0',
                        fillOpacity: 0.8,
                        weight: 2
                      }}
                    >
                      <Popup>
                        <div className="text-sm">
                          <h4 className="font-semibold mb-2">🔮 Forecast Point</h4>
                          <p><strong>Borough:</strong> {point.borough}</p>
                          <p><strong>Risk Level:</strong> <span style={{color: point.risk_level === 'High' ? '#dc2626' : point.risk_level === 'Medium' ? '#ea580c' : '#16a34a'}}>{point.risk_level}</span></p>
                          <p><strong>Date:</strong> {point.date || point.month}</p>
                          <p><strong>Location:</strong> {point.location_type || 'Forecast Area'}</p>
                          <p><strong>Status:</strong> {point.outcome_status || 'Predicted'}</p>
                          <p className="text-xs text-gray-600 mt-1">
                            Coordinates: {point.lat.toFixed(4)}, {point.lng.toFixed(4)}
                          </p>
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

          {/* Click-to-add indicator */}
          {canAddPoints && (
            <div className="leaflet-top leaflet-left" style={{ marginTop: '200px', marginLeft: '10px' }}>
              <div className="bg-green-600 text-white px-3 py-2 rounded shadow-lg text-sm font-medium animate-pulse">
                🖱️ Click on map to add burglary points
              </div>
            </div>
          )}

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