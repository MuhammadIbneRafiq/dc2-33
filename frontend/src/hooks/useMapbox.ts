import { useState, useEffect, useRef, useCallback } from 'react';
import mapboxgl from 'mapbox-gl';
// import { getCrimeForecastData } from '../api/backendService';
import { 
  crimeDataToGeoJSON, 
  highRiskPointsToGeoJSON, 
  policeDataToGeoJSON,
  addHeatmapLayer,
  addPointsLayer,
  addPoliceMarkersLayer,
  createCrimePointPopupHTML,
  createPoliceMarkerPopupHTML
} from '../utils/mapUtils';

// Default demo token - shouldn't be used in production
const DEMO_MAPBOX_TOKEN = 'pk.eyJ1IjoiZGVtby1hcGkta2V5IiwiYSI6ImNsMnpxenkwZTBmZjMza3RhbzMwYjVvZGQifQ.qibivFHf6bzU4Wo-oLFEfg';

export const useMapbox = (
  onSelectLSOA: (lsoa: string) => void,
  showPoliceAllocation: boolean,
  policeData: any[] | null
) => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [crimeData, setCrimeData] = useState<any[] | null>(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [tokenInput, setTokenInput] = useState('');
  const [tokenSubmitted, setTokenSubmitted] = useState(false);
  const [effectiveToken, setEffectiveToken] = useState(DEMO_MAPBOX_TOKEN);
  
  const handleTokenSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    if (tokenInput) {
      setEffectiveToken(tokenInput);
      setTokenSubmitted(true);
    }
  }, [tokenInput]);
  
  // Initialize map
  useEffect(() => {
    if (mapContainer.current && !map.current) {
      mapboxgl.accessToken = effectiveToken;
      
      try {
        map.current = new mapboxgl.Map({
          container: mapContainer.current,
          style: 'mapbox://styles/mapbox/dark-v11',
          center: [-0.118092, 51.509865], // London coordinates
          zoom: 10,
          pitch: 30,
          bearing: -17.6,
          antialias: true
        });

        map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');
        map.current.addControl(new mapboxgl.GeolocateControl({
          positionOptions: { enableHighAccuracy: true },
          trackUserLocation: true
        }));
        
        map.current.on('load', () => {
          setMapLoaded(true);
        });
        
      } catch (error) {
        console.error("Error initializing map:", error);
      }
    }
    
    return () => {
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, [effectiveToken]);
  
  // Load crime data
  useEffect(() => {
    const loadCrimeData = async () => {
      try {
        // const data = await getCrimeForecastData();
        setCrimeData(await mockCrimeForecastData());
      } catch (error) {
        console.error("Failed to load crime forecast data:", error);
      }
    };
    
    if (mapLoaded) {
      loadCrimeData();
    }
  }, [mapLoaded]);
  
  // Add crime data to map
  useEffect(() => {
    if (!map.current || !mapLoaded || !crimeData || crimeData.length === 0) return;
    
    // Remove existing sources and layers if they exist
    if (map.current.getSource('crime-hotspots')) {
      map.current.removeLayer('crime-heat');
      map.current.removeSource('crime-hotspots');
    }
    
    if (map.current.getSource('crime-points')) {
      map.current.removeLayer('crime-points');
      map.current.removeSource('crime-points');
    }
    
    // Convert data to GeoJSON
    const geojsonHotspots = crimeDataToGeoJSON(crimeData);
    const geojsonPoints = highRiskPointsToGeoJSON(crimeData);
    
    // Add layers
    addHeatmapLayer(map.current, geojsonHotspots);
    addPointsLayer(map.current, geojsonPoints);
    
    // Add click event to show LSOA code
    map.current.on('click', 'crime-points', (e) => {
      if (e.features && e.features[0]) {
        const properties = e.features[0].properties;
        if (properties && properties.lsoa_code) {
          onSelectLSOA(properties.lsoa_code);
          
          new mapboxgl.Popup()
            .setLngLat(e.lngLat)
            .setHTML(createCrimePointPopupHTML(properties))
            .addTo(map.current!);
        }
      }
    });
    
    // Change cursor on hover
    map.current.on('mouseenter', 'crime-points', () => {
      if (map.current) map.current.getCanvas().style.cursor = 'pointer';
    });
    
    map.current.on('mouseleave', 'crime-points', () => {
      if (map.current) map.current.getCanvas().style.cursor = '';
    });
    
  }, [crimeData, mapLoaded, onSelectLSOA]);
  
  // Add police allocation to map
  useEffect(() => {
    if (!map.current || !mapLoaded || !policeData) return;
    
    // Remove existing police markers if they exist
    if (map.current.getSource('police-units')) {
      map.current.removeLayer('police-markers');
      map.current.removeSource('police-units');
    }
    
    if (showPoliceAllocation) {
      // Convert police data to GeoJSON
      const geojsonPolice = policeDataToGeoJSON(policeData);
      
      // Add police markers layer
      addPoliceMarkersLayer(map.current, geojsonPolice);
      
      // Add click event for police units
      map.current.on('click', 'police-markers', (e) => {
        if (e.features && e.features[0]) {
          const properties = e.features[0].properties;
          if (properties) {
            new mapboxgl.Popup()
              .setLngLat(e.lngLat)
              .setHTML(createPoliceMarkerPopupHTML(properties))
              .addTo(map.current!);
          }
        }
      });
      
      // Change cursor on hover for police markers
      map.current.on('mouseenter', 'police-markers', () => {
        if (map.current) map.current.getCanvas().style.cursor = 'pointer';
      });
      
      map.current.on('mouseleave', 'police-markers', () => {
        if (map.current) map.current.getCanvas().style.cursor = '';
      });
    }
    
  }, [showPoliceAllocation, policeData, mapLoaded, onSelectLSOA]);

  return {
    mapContainer,
    tokenInput,
    setTokenInput,
    tokenSubmitted,
    handleTokenSubmit,
    needsToken: !tokenSubmitted && DEMO_MAPBOX_TOKEN.includes('demo-api-key')
  };
};

// Add a mockCrimeForecastData function to simulate data
const mockCrimeForecastData = async () => {
  return Array.from({ length: 50 }, () => {
    const latitude = 51.5 + (Math.random() * 0.2) - 0.1;
    const longitude = -0.12 + (Math.random() * 0.2) - 0.1;
    const intensity = Math.random();
    const burglary_risk = Math.floor(Math.random() * 100) + 1;
    const lsoa_code = `E${String(Math.floor(Math.random() * 90000) + 10000).padStart(5, '0')}`;
    return {
      latitude,
      longitude,
      intensity,
      burglary_risk,
      lsoa_code,
      trend: Math.random() > 0.5 ? 'up' : 'down',
      last_month_incidents: Math.floor(Math.random() * 30),
    };
  });
};
