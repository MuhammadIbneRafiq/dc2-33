import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet icon issue
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

interface LSOAProperties {
  LSOA21CD: string;
  LSOA21NM: string;
  LAT: number;
  LONG: number;
  burglary_count?: number;
  risk_level?: string;
}

interface LSOAFeature {
  type: 'Feature';
  properties: LSOAProperties;
  geometry: any;
}

interface LSOAData {
  type: 'FeatureCollection';
  features: LSOAFeature[];
}

const RealBoundaryMap = () => {
  const [lsoaData, setLsoaData] = useState<LSOAData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const LONDON_CENTER: [number, number] = [51.5074, -0.1278];

  // Risk color mapping
  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'Very High': return '#ef4444';
      case 'High': return '#f97316';
      case 'Medium': return '#eab308';
      case 'Low': return '#84cc16';
      default: return '#94a3b8';
    }
  };

  const getFillOpacity = (riskLevel: string) => {
    switch (riskLevel) {
      case 'Very High': return 0.25;
      case 'High': return 0.2;
      case 'Medium': return 0.15;
      case 'Low': return 0.1;
      default: return 0.05;
    }
  };

  // Fetch real London LSOA data
  useEffect(() => {
    const fetchLondonLSOAs = async () => {
      setLoading(true);
      setError(null);

      try {
        console.log('🗺️ Fetching real London LSOA boundaries from ONS...');

        const priorityBoroughs = [
          'Westminster%', 'Camden%', 'Islington%', 'Hackney%', 'Tower Hamlets%',
          'Southwark%', 'Lambeth%', 'Kensington and Chelsea%', 'City of London%'
        ];

        const allFeatures: LSOAFeature[] = [];

        for (const borough of priorityBoroughs) {
          try {
            const params = new URLSearchParams({
              where: `LSOA21NM like '${borough}'`,
              outSR: '4326',
              f: 'geoJSON',
              outFields: 'LSOA21CD,LSOA21NM,LAT,LONG'
            });

            const endpoint = 'https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BFC_V10/FeatureServer/0/query';
            
            const response = await fetch(`${endpoint}?${params}`);
            
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
                    burglary_count: burglaryCount,
                    risk_level: riskLevel
                  }
                };
              });

              allFeatures.push(...enrichedFeatures);
              console.log(`✅ Fetched ${data.features.length} LSOAs for ${boroughName}`);
            }

            // Small delay to be respectful to the API
            await new Promise(resolve => setTimeout(resolve, 300));

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

        setLsoaData(lsoaCollection);
        console.log(`🎉 Successfully loaded ${allFeatures.length} real London LSOAs`);

      } catch (err) {
        console.error('❌ Failed to fetch London LSOA boundaries:', err);
        setError(err instanceof Error ? err.message : 'Failed to load boundary data');
      } finally {
        setLoading(false);
      }
    };

    fetchLondonLSOAs();
  }, []);

  // Style function for LSOA boundaries
  const lsoaStyle = (feature: LSOAFeature) => {
    const riskLevel = feature.properties.risk_level || 'Medium';
    
    return {
      fillColor: getRiskColor(riskLevel),
      weight: 1,
      opacity: 0.8,
      color: '#fff',
      fillOpacity: getFillOpacity(riskLevel),
    };
  };

  // Event handlers for LSOA features
  const onEachLSOAFeature = (feature: LSOAFeature, layer: L.Layer) => {
    const properties = feature.properties;
    
    layer.on({
      mouseover: (e) => {
        const target = e.target;
        target.setStyle({
          weight: 2,
          color: '#000',
          fillOpacity: 0.5
        });
        target.bringToFront();
      },
      mouseout: (e) => {
        const target = e.target;
        target.setStyle(lsoaStyle(feature));
      }
    });

    // Popup content
    const popupContent = `
      <div style="font-family: system-ui; color: white; background: rgba(30, 41, 59, 0.95); padding: 12px; border-radius: 8px;">
        <h3 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 600;">${properties.LSOA21CD}</h3>
        ${properties.LSOA21NM ? `<p style="margin: 0 0 4px 0; font-size: 12px; opacity: 0.8;">${properties.LSOA21NM}</p>` : ''}
        <p style="margin: 0; font-size: 12px;"><strong>Burglary Count:</strong> ${properties.burglary_count || 0}</p>
        <p style="margin: 0; font-size: 12px;"><strong>Risk Level:</strong> ${properties.risk_level || 'Unknown'}</p>
      </div>
    `;
    
    layer.bindPopup(popupContent);
  };

  if (loading) {
    return (
      <div className="h-screen w-full flex items-center justify-center bg-gray-100">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600 text-lg">Loading real London LSOA boundaries...</p>
          <p className="text-gray-500 text-sm">Fetching data from ONS Open Geography Portal</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-screen w-full flex items-center justify-center bg-gray-100">
        <div className="text-center text-red-600">
          <p className="text-lg font-semibold">Failed to load boundary data</p>
          <p className="text-sm">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-full relative">
      <MapContainer 
        center={LONDON_CENTER} 
        zoom={11} 
        style={{ height: '100%', width: '100%' }}
        preferCanvas={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        
        {lsoaData && (
          <GeoJSON
            key={`lsoa-${lsoaData.features.length}`}
            data={lsoaData}
            style={lsoaStyle}
            onEachFeature={onEachLSOAFeature}
          />
        )}
      </MapContainer>

      {/* Legend */}
      <div className="absolute top-4 right-4 bg-white bg-opacity-90 p-4 rounded-lg shadow-lg z-1000">
        <h3 className="text-sm font-semibold mb-2">Crime Risk Level</h3>
        <div className="space-y-1">
          {['Very High', 'High', 'Medium', 'Low'].map(level => (
            <div key={level} className="flex items-center space-x-2">
              <div 
                className="w-4 h-4 rounded border border-gray-300"
                style={{ backgroundColor: getRiskColor(level), opacity: getFillOpacity(level) + 0.5 }}
              />
              <span className="text-xs">{level}</span>
            </div>
          ))}
        </div>
        <div className="mt-3 pt-2 border-t border-gray-200">
          <p className="text-xs text-gray-600">
            Data: ONS Open Geography Portal
          </p>
          <p className="text-xs text-gray-600">
            {lsoaData?.features.length || 0} LSOAs loaded
          </p>
        </div>
      </div>
    </div>
  );
};

export default RealBoundaryMap; 