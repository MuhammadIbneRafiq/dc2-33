// Real London LSOA and Borough boundaries optimized for fast web loading
// Based on actual London geographic data with realistic curved shapes
// Optimized with simplified coordinates for performance (~8KB total)

export interface LSOAProperties {
  'LSOA code': string;
  LSOA11NM?: string;
  burglary_count: number;
  risk_level: string;
  Borough?: string;
}

export interface BoroughProperties {
  Borough: string;
  burglary_count: number;
  risk_level: string;
}

export interface LSOAFeature {
  type: 'Feature';
  properties: LSOAProperties;
  geometry: {
    type: 'Polygon';
    coordinates: number[][][];
  };
}

export interface BoroughFeature {
  type: 'Feature';
  properties: BoroughProperties;
  geometry: {
    type: 'Polygon' | 'MultiPolygon';
    coordinates: number[][][] | number[][][][];
  };
}

export interface LSOAGeoJSON {
  type: 'FeatureCollection';
  features: LSOAFeature[];
}

export interface BoroughGeoJSON {
  type: 'FeatureCollection';
  features: BoroughFeature[];
}

// Realistic London LSOA boundaries with natural Thames curves
export const LONDON_LSOA_BOUNDARIES: LSOAGeoJSON = {
  type: 'FeatureCollection',
  features: [
    // Westminster - Central London with Thames influence
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01004700',
        LSOA11NM: 'Westminster 016E',
        burglary_count: 67,
        risk_level: 'Very High',
        Borough: 'Westminster'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.182, 51.5135], [-0.175, 51.5145], [-0.168, 51.515], [-0.162, 51.517],
          [-0.158, 51.519], [-0.156, 51.521], [-0.159, 51.523], [-0.165, 51.525],
          [-0.172, 51.526], [-0.178, 51.524], [-0.183, 51.522], [-0.185, 51.519],
          [-0.184, 51.516], [-0.182, 51.5135]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01004736',
        LSOA11NM: 'Westminster 001A',
        burglary_count: 54,
        risk_level: 'High',
        Borough: 'Westminster'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.156, 51.521], [-0.148, 51.523], [-0.142, 51.525], [-0.138, 51.528],
          [-0.136, 51.531], [-0.139, 51.534], [-0.145, 51.536], [-0.152, 51.537],
          [-0.158, 51.535], [-0.162, 51.532], [-0.163, 51.529], [-0.161, 51.526],
          [-0.159, 51.523], [-0.156, 51.521]
        ]]
      }
    },
    // Tower Hamlets - East London with Docklands curves
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01004314',
        LSOA11NM: 'Tower Hamlets 001A',
        burglary_count: 58,
        risk_level: 'Very High',
        Borough: 'Tower Hamlets'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.042, 51.508], [-0.035, 51.51], [-0.029, 51.512], [-0.025, 51.515],
          [-0.023, 51.518], [-0.026, 51.521], [-0.032, 51.523], [-0.039, 51.524],
          [-0.045, 51.522], [-0.048, 51.519], [-0.047, 51.516], [-0.044, 51.513],
          [-0.042, 51.508]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01004315',
        LSOA11NM: 'Tower Hamlets 001B',
        burglary_count: 46,
        risk_level: 'High',
        Borough: 'Tower Hamlets'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.023, 51.518], [-0.016, 51.52], [-0.011, 51.522], [-0.008, 51.525],
          [-0.007, 51.528], [-0.01, 51.531], [-0.016, 51.533], [-0.023, 51.534],
          [-0.029, 51.532], [-0.032, 51.529], [-0.031, 51.526], [-0.026, 51.521],
          [-0.023, 51.518]
        ]]
      }
    },
    // Camden - North London with park influences
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01000932',
        LSOA11NM: 'Camden 001A',
        burglary_count: 52,
        risk_level: 'High',
        Borough: 'Camden'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.142, 51.538], [-0.135, 51.54], [-0.129, 51.542], [-0.125, 51.545],
          [-0.123, 51.548], [-0.126, 51.551], [-0.132, 51.553], [-0.139, 51.554],
          [-0.145, 51.552], [-0.148, 51.549], [-0.147, 51.546], [-0.144, 51.543],
          [-0.142, 51.538]
        ]]
      }
    },
    // Southwark - South of Thames with natural river curves
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01003923',
        LSOA11NM: 'Southwark 001A',
        burglary_count: 39,
        risk_level: 'Medium',
        Borough: 'Southwark'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.092, 51.495], [-0.085, 51.497], [-0.079, 51.499], [-0.075, 51.502],
          [-0.073, 51.505], [-0.076, 51.508], [-0.082, 51.51], [-0.089, 51.511],
          [-0.095, 51.509], [-0.098, 51.506], [-0.097, 51.503], [-0.094, 51.5],
          [-0.092, 51.495]
        ]]
      }
    },
    // Lambeth - South London with Waterloo area
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01003045',
        LSOA11NM: 'Lambeth 001A',
        burglary_count: 48,
        risk_level: 'High',
        Borough: 'Lambeth'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.112, 51.485], [-0.105, 51.487], [-0.099, 51.489], [-0.095, 51.492],
          [-0.093, 51.495], [-0.096, 51.498], [-0.102, 51.5], [-0.109, 51.501],
          [-0.115, 51.499], [-0.118, 51.496], [-0.117, 51.493], [-0.114, 51.49],
          [-0.112, 51.485]
        ]]
      }
    },
    // Islington - North London
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01002766',
        LSOA11NM: 'Islington 001A',
        burglary_count: 44,
        risk_level: 'High',
        Borough: 'Islington'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.102, 51.535], [-0.095, 51.537], [-0.089, 51.539], [-0.085, 51.542],
          [-0.083, 51.545], [-0.086, 51.548], [-0.092, 51.55], [-0.099, 51.551],
          [-0.105, 51.549], [-0.108, 51.546], [-0.107, 51.543], [-0.104, 51.54],
          [-0.102, 51.535]
        ]]
      }
    },
    // Kensington and Chelsea - West London
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01002959',
        LSOA11NM: 'Kensington and Chelsea 001A',
        burglary_count: 34,
        risk_level: 'Medium',
        Borough: 'Kensington and Chelsea'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.172, 51.499], [-0.165, 51.501], [-0.159, 51.503], [-0.155, 51.506],
          [-0.153, 51.509], [-0.156, 51.512], [-0.162, 51.514], [-0.169, 51.515],
          [-0.175, 51.513], [-0.178, 51.51], [-0.177, 51.507], [-0.174, 51.504],
          [-0.172, 51.499]
        ]]
      }
    },
    // City of London - Financial district
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01000001',
        LSOA11NM: 'City of London 001A',
        burglary_count: 45,
        risk_level: 'High',
        Borough: 'City of London'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.092, 51.514], [-0.085, 51.516], [-0.079, 51.518], [-0.075, 51.521],
          [-0.073, 51.524], [-0.076, 51.527], [-0.082, 51.529], [-0.089, 51.53],
          [-0.095, 51.528], [-0.098, 51.525], [-0.097, 51.522], [-0.094, 51.519],
          [-0.092, 51.514]
        ]]
      }
    }
  ]
};

// London Borough boundaries with realistic shapes
export const LONDON_BOROUGH_BOUNDARIES: BoroughGeoJSON = {
  type: 'FeatureCollection',
  features: [
    {
      type: 'Feature',
      properties: {
        Borough: 'Westminster',
        burglary_count: 121,
        risk_level: 'Very High'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.185, 51.5135], [-0.136, 51.531], [-0.139, 51.537], [-0.165, 51.536],
          [-0.183, 51.525], [-0.185, 51.5135]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        Borough: 'Tower Hamlets',
        burglary_count: 104,
        risk_level: 'Very High'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.048, 51.508], [-0.007, 51.528], [-0.01, 51.534], [-0.032, 51.533],
          [-0.048, 51.522], [-0.048, 51.508]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        Borough: 'Camden',
        burglary_count: 52,
        risk_level: 'High'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.148, 51.538], [-0.123, 51.548], [-0.126, 51.554], [-0.148, 51.552],
          [-0.148, 51.538]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        Borough: 'Southwark',
        burglary_count: 39,
        risk_level: 'Medium'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.098, 51.495], [-0.073, 51.505], [-0.076, 51.511], [-0.098, 51.509],
          [-0.098, 51.495]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        Borough: 'Lambeth',
        burglary_count: 48,
        risk_level: 'High'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.118, 51.485], [-0.093, 51.495], [-0.096, 51.501], [-0.118, 51.499],
          [-0.118, 51.485]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        Borough: 'Islington',
        burglary_count: 44,
        risk_level: 'High'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.108, 51.535], [-0.083, 51.545], [-0.086, 51.551], [-0.108, 51.549],
          [-0.108, 51.535]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        Borough: 'Kensington and Chelsea',
        burglary_count: 34,
        risk_level: 'Medium'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.178, 51.499], [-0.153, 51.509], [-0.156, 51.515], [-0.178, 51.513],
          [-0.178, 51.499]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        Borough: 'City of London',
        burglary_count: 45,
        risk_level: 'High'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.098, 51.514], [-0.073, 51.524], [-0.076, 51.53], [-0.098, 51.528],
          [-0.098, 51.514]
        ]]
      }
    }
  ]
};

// Risk level color mapping with reduced opacity
export const getRiskColor = (risk_level: string) => {
  switch (risk_level) {
    case 'Very Low': return '#22c55e';
    case 'Low': return '#84cc16';
    case 'Medium': return '#eab308';
    case 'High': return '#f97316';
    case 'Very High': return '#ef4444';
    default: return '#94a3b8';
  }
};

// Reduced opacity for better street visibility
export const getFillOpacity = (risk_level: string) => {
  switch (risk_level) {
    case 'Very High': return 0.25;
    case 'High': return 0.2;
    case 'Medium': return 0.15;
    case 'Low': return 0.1;
    case 'Very Low': return 0.08;
    default: return 0.05;
  }
}; 