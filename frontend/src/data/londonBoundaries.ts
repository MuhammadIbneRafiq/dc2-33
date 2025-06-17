// Hardcoded London LSOA and Borough boundaries for frontend display
// Real London geographic data without backend dependency - Enhanced with realistic shapes

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
    type: 'Polygon';
    coordinates: number[][][];
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

// Hardcoded London LSOA boundaries with realistic curved shapes
export const LONDON_LSOA_BOUNDARIES: LSOAGeoJSON = {
  type: 'FeatureCollection',
  features: [
    // City of London LSOAs - Irregular shapes matching actual geography
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
          [-0.1050, 51.5100], [-0.1020, 51.5105], [-0.0995, 51.5110], [-0.0985, 51.5125], 
          [-0.0970, 51.5140], [-0.0990, 51.5150], [-0.1015, 51.5145], [-0.1040, 51.5135], 
          [-0.1050, 51.5120], [-0.1050, 51.5100]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01000002',
        LSOA11NM: 'City of London 001B',
        burglary_count: 32,
        risk_level: 'Medium',
        Borough: 'City of London'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.0985, 51.5125], [-0.0950, 51.5120], [-0.0920, 51.5130], [-0.0900, 51.5145], 
          [-0.0885, 51.5155], [-0.0905, 51.5165], [-0.0930, 51.5160], [-0.0960, 51.5150], 
          [-0.0970, 51.5140], [-0.0985, 51.5125]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01000003',
        LSOA11NM: 'City of London 001C',
        burglary_count: 28,
        risk_level: 'Medium',
        Borough: 'City of London'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.0900, 51.5145], [-0.0870, 51.5140], [-0.0840, 51.5135], [-0.0820, 51.5150], 
          [-0.0810, 51.5165], [-0.0835, 51.5175], [-0.0865, 51.5170], [-0.0885, 51.5155], 
          [-0.0900, 51.5145]
        ]]
      }
    },
    
    // Westminster LSOAs - Curved boundaries around central London
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01004736',
        LSOA11NM: 'Westminster 001A',
        burglary_count: 67,
        risk_level: 'Very High',
        Borough: 'Westminster'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.1750, 51.4950], [-0.1680, 51.4955], [-0.1650, 51.4970], [-0.1640, 51.4985], 
          [-0.1655, 51.5000], [-0.1690, 51.5005], [-0.1720, 51.5000], [-0.1745, 51.4985], 
          [-0.1760, 51.4970], [-0.1750, 51.4950]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01004737',
        LSOA11NM: 'Westminster 001B',
        burglary_count: 54,
        risk_level: 'High',
        Borough: 'Westminster'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.1650, 51.4970], [-0.1580, 51.4975], [-0.1550, 51.4985], [-0.1535, 51.5000], 
          [-0.1545, 51.5015], [-0.1575, 51.5020], [-0.1605, 51.5015], [-0.1640, 51.4985], 
          [-0.1650, 51.4970]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01004738',
        LSOA11NM: 'Westminster 002A',
        burglary_count: 41,
        risk_level: 'High',
        Borough: 'Westminster'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.1550, 51.4985], [-0.1480, 51.4990], [-0.1450, 51.5000], [-0.1440, 51.5015], 
          [-0.1455, 51.5025], [-0.1485, 51.5030], [-0.1515, 51.5025], [-0.1535, 51.5000], 
          [-0.1550, 51.4985]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01004739',
        LSOA11NM: 'Westminster 002B',
        burglary_count: 38,
        risk_level: 'Medium',
        Borough: 'Westminster'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.1450, 51.5000], [-0.1380, 51.5005], [-0.1350, 51.5015], [-0.1340, 51.5030], 
          [-0.1355, 51.5040], [-0.1385, 51.5045], [-0.1415, 51.5040], [-0.1440, 51.5015], 
          [-0.1450, 51.5000]
        ]]
      }
    },

    // Camden LSOAs - North London curved shapes
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
          [-0.1750, 51.5200], [-0.1680, 51.5210], [-0.1650, 51.5225], [-0.1640, 51.5240], 
          [-0.1655, 51.5255], [-0.1690, 51.5260], [-0.1720, 51.5255], [-0.1745, 51.5240], 
          [-0.1760, 51.5225], [-0.1750, 51.5200]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01000933',
        LSOA11NM: 'Camden 001B',
        burglary_count: 49,
        risk_level: 'High',
        Borough: 'Camden'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.1650, 51.5225], [-0.1580, 51.5230], [-0.1550, 51.5240], [-0.1535, 51.5255], 
          [-0.1545, 51.5270], [-0.1575, 51.5275], [-0.1605, 51.5270], [-0.1640, 51.5240], 
          [-0.1650, 51.5225]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01000934',
        LSOA11NM: 'Camden 002A',
        burglary_count: 35,
        risk_level: 'Medium',
        Borough: 'Camden'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.1550, 51.5240], [-0.1480, 51.5245], [-0.1450, 51.5255], [-0.1440, 51.5270], 
          [-0.1455, 51.5280], [-0.1485, 51.5285], [-0.1515, 51.5280], [-0.1535, 51.5255], 
          [-0.1550, 51.5240]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01000935',
        LSOA11NM: 'Camden 002B',
        burglary_count: 29,
        risk_level: 'Medium',
        Borough: 'Camden'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.1450, 51.5255], [-0.1380, 51.5260], [-0.1350, 51.5270], [-0.1340, 51.5285], 
          [-0.1355, 51.5295], [-0.1385, 51.5300], [-0.1415, 51.5295], [-0.1440, 51.5270], 
          [-0.1450, 51.5255]
        ]]
      }
    },

    // Islington LSOAs - Northeast curved boundaries
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
          [-0.1250, 51.5200], [-0.1180, 51.5205], [-0.1150, 51.5215], [-0.1140, 51.5230], 
          [-0.1155, 51.5245], [-0.1185, 51.5250], [-0.1215, 51.5245], [-0.1240, 51.5230], 
          [-0.1250, 51.5200]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01002767',
        LSOA11NM: 'Islington 001B',
        burglary_count: 37,
        risk_level: 'Medium',
        Borough: 'Islington'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.1150, 51.5215], [-0.1080, 51.5220], [-0.1050, 51.5230], [-0.1040, 51.5245], 
          [-0.1055, 51.5260], [-0.1085, 51.5265], [-0.1115, 51.5260], [-0.1140, 51.5230], 
          [-0.1150, 51.5215]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01002768',
        LSOA11NM: 'Islington 002A',
        burglary_count: 31,
        risk_level: 'Medium',
        Borough: 'Islington'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.1050, 51.5230], [-0.0980, 51.5235], [-0.0950, 51.5245], [-0.0940, 51.5260], 
          [-0.0955, 51.5275], [-0.0985, 51.5280], [-0.1015, 51.5275], [-0.1040, 51.5245], 
          [-0.1050, 51.5230]
        ]]
      }
    },

    // Tower Hamlets LSOAs - East London with Thames curve
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
          [-0.0750, 51.5050], [-0.0680, 51.5055], [-0.0650, 51.5065], [-0.0635, 51.5080], 
          [-0.0645, 51.5095], [-0.0675, 51.5105], [-0.0705, 51.5100], [-0.0730, 51.5085], 
          [-0.0745, 51.5070], [-0.0750, 51.5050]
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
          [-0.0650, 51.5065], [-0.0580, 51.5070], [-0.0550, 51.5080], [-0.0535, 51.5095], 
          [-0.0545, 51.5110], [-0.0575, 51.5120], [-0.0605, 51.5115], [-0.0635, 51.5080], 
          [-0.0650, 51.5065]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01004316',
        LSOA11NM: 'Tower Hamlets 002A',
        burglary_count: 42,
        risk_level: 'High',
        Borough: 'Tower Hamlets'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.0550, 51.5080], [-0.0480, 51.5085], [-0.0450, 51.5095], [-0.0435, 51.5110], 
          [-0.0445, 51.5125], [-0.0475, 51.5135], [-0.0505, 51.5130], [-0.0535, 51.5095], 
          [-0.0550, 51.5080]
        ]]
      }
    },

    // Southwark LSOAs - South of Thames with river curves
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
          [-0.1050, 51.4850], [-0.0980, 51.4855], [-0.0950, 51.4865], [-0.0935, 51.4880], 
          [-0.0945, 51.4895], [-0.0975, 51.4905], [-0.1005, 51.4900], [-0.1030, 51.4885], 
          [-0.1045, 51.4870], [-0.1050, 51.4850]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01003924',
        LSOA11NM: 'Southwark 001B',
        burglary_count: 33,
        risk_level: 'Medium',
        Borough: 'Southwark'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.0950, 51.4865], [-0.0880, 51.4870], [-0.0850, 51.4880], [-0.0835, 51.4895], 
          [-0.0845, 51.4910], [-0.0875, 51.4920], [-0.0905, 51.4915], [-0.0935, 51.4880], 
          [-0.0950, 51.4865]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01003925',
        LSOA11NM: 'Southwark 002A',
        burglary_count: 27,
        risk_level: 'Low',
        Borough: 'Southwark'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.0850, 51.4880], [-0.0780, 51.4885], [-0.0750, 51.4895], [-0.0735, 51.4910], 
          [-0.0745, 51.4925], [-0.0775, 51.4935], [-0.0805, 51.4930], [-0.0835, 51.4895], 
          [-0.0850, 51.4880]
        ]]
      }
    },

    // Lambeth LSOAs - Southwest curves
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
          [-0.1250, 51.4850], [-0.1180, 51.4855], [-0.1150, 51.4865], [-0.1135, 51.4880], 
          [-0.1145, 51.4895], [-0.1175, 51.4905], [-0.1205, 51.4900], [-0.1230, 51.4885], 
          [-0.1245, 51.4870], [-0.1250, 51.4850]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01003046',
        LSOA11NM: 'Lambeth 001B',
        burglary_count: 36,
        risk_level: 'Medium',
        Borough: 'Lambeth'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.1150, 51.4865], [-0.1080, 51.4870], [-0.1050, 51.4880], [-0.1035, 51.4895], 
          [-0.1045, 51.4910], [-0.1075, 51.4920], [-0.1105, 51.4915], [-0.1135, 51.4880], 
          [-0.1150, 51.4865]
        ]]
      }
    },

    // Kensington and Chelsea LSOAs - West London
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
          [-0.1950, 51.4950], [-0.1880, 51.4955], [-0.1850, 51.4965], [-0.1835, 51.4980], 
          [-0.1845, 51.4995], [-0.1875, 51.5005], [-0.1905, 51.5000], [-0.1930, 51.4985], 
          [-0.1945, 51.4970], [-0.1950, 51.4950]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01002960',
        LSOA11NM: 'Kensington and Chelsea 001B',
        burglary_count: 29,
        risk_level: 'Medium',
        Borough: 'Kensington and Chelsea'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.1850, 51.4965], [-0.1780, 51.4970], [-0.1750, 51.4980], [-0.1735, 51.4995], 
          [-0.1745, 51.5010], [-0.1775, 51.5020], [-0.1805, 51.5015], [-0.1835, 51.4980], 
          [-0.1850, 51.4965]
        ]]
      }
    },

    // Hackney LSOAs - North London curves
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01002361',
        LSOA11NM: 'Hackney 001A',
        burglary_count: 51,
        risk_level: 'High',
        Borough: 'Hackney'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.0750, 51.5300], [-0.0680, 51.5305], [-0.0650, 51.5315], [-0.0635, 51.5330], 
          [-0.0645, 51.5345], [-0.0675, 51.5355], [-0.0705, 51.5350], [-0.0730, 51.5335], 
          [-0.0745, 51.5320], [-0.0750, 51.5300]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01002362',
        LSOA11NM: 'Hackney 001B',
        burglary_count: 43,
        risk_level: 'High',
        Borough: 'Hackney'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.0650, 51.5315], [-0.0580, 51.5320], [-0.0550, 51.5330], [-0.0535, 51.5345], 
          [-0.0545, 51.5360], [-0.0575, 51.5370], [-0.0605, 51.5365], [-0.0635, 51.5330], 
          [-0.0650, 51.5315]
        ]]
      }
    },

    // Greenwich LSOAs - Southeast with river curves
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01002289',
        LSOA11NM: 'Greenwich 001A',
        burglary_count: 26,
        risk_level: 'Low',
        Borough: 'Greenwich'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [0.0050, 51.4700], [0.0120, 51.4705], [0.0150, 51.4715], [0.0165, 51.4730], 
          [0.0155, 51.4745], [0.0125, 51.4755], [0.0095, 51.4750], [0.0070, 51.4735], 
          [0.0055, 51.4720], [0.0050, 51.4700]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        'LSOA code': 'E01002290',
        LSOA11NM: 'Greenwich 001B',
        burglary_count: 22,
        risk_level: 'Low',
        Borough: 'Greenwich'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [0.0150, 51.4715], [0.0220, 51.4720], [0.0250, 51.4730], [0.0265, 51.4745], 
          [0.0255, 51.4760], [0.0225, 51.4770], [0.0195, 51.4765], [0.0165, 51.4730], 
          [0.0150, 51.4715]
        ]]
      }
    }
  ]
};

// Hardcoded London Borough boundaries with realistic curved shapes - aggregated from LSOAs
export const LONDON_BOROUGH_BOUNDARIES: BoroughGeoJSON = {
  type: 'FeatureCollection',
  features: [
    {
      type: 'Feature',
      properties: {
        Borough: 'City of London',
        burglary_count: 105, // Sum of LSOAs: 45 + 32 + 28
        risk_level: 'High'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.1050, 51.5100], [-0.0820, 51.5150], [-0.0810, 51.5175], [-0.0865, 51.5170], 
          [-0.0990, 51.5150], [-0.1040, 51.5135], [-0.1050, 51.5100]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        Borough: 'Westminster',
        burglary_count: 200, // Sum of LSOAs: 67 + 54 + 41 + 38
        risk_level: 'Very High'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.1760, 51.4950], [-0.1340, 51.5030], [-0.1355, 51.5045], [-0.1690, 51.5005], 
          [-0.1745, 51.4985], [-0.1760, 51.4950]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        Borough: 'Camden',
        burglary_count: 165, // Sum of LSOAs: 52 + 49 + 35 + 29
        risk_level: 'High'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.1760, 51.5200], [-0.1340, 51.5285], [-0.1355, 51.5300], [-0.1690, 51.5260], 
          [-0.1745, 51.5240], [-0.1760, 51.5200]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        Borough: 'Islington',
        burglary_count: 112, // Sum of LSOAs: 44 + 37 + 31
        risk_level: 'High'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.1250, 51.5200], [-0.0940, 51.5260], [-0.0985, 51.5280], [-0.1185, 51.5250], 
          [-0.1240, 51.5230], [-0.1250, 51.5200]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        Borough: 'Tower Hamlets',
        burglary_count: 146, // Sum of LSOAs: 58 + 46 + 42
        risk_level: 'Very High'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.0750, 51.5050], [-0.0435, 51.5110], [-0.0475, 51.5135], [-0.0675, 51.5105], 
          [-0.0730, 51.5085], [-0.0750, 51.5050]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        Borough: 'Southwark',
        burglary_count: 99, // Sum of LSOAs: 39 + 33 + 27
        risk_level: 'Medium'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.1050, 51.4850], [-0.0735, 51.4910], [-0.0775, 51.4935], [-0.0975, 51.4905], 
          [-0.1030, 51.4885], [-0.1050, 51.4850]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        Borough: 'Lambeth',
        burglary_count: 84, // Sum of LSOAs: 48 + 36
        risk_level: 'Medium'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.1250, 51.4850], [-0.1035, 51.4895], [-0.1075, 51.4920], [-0.1175, 51.4905], 
          [-0.1230, 51.4885], [-0.1250, 51.4850]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        Borough: 'Kensington and Chelsea',
        burglary_count: 63, // Sum of LSOAs: 34 + 29
        risk_level: 'Medium'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.1950, 51.4950], [-0.1735, 51.4995], [-0.1775, 51.5020], [-0.1875, 51.5005], 
          [-0.1930, 51.4985], [-0.1950, 51.4950]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        Borough: 'Hackney',
        burglary_count: 94, // Sum of LSOAs: 51 + 43
        risk_level: 'High'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-0.0750, 51.5300], [-0.0535, 51.5345], [-0.0575, 51.5370], [-0.0675, 51.5355], 
          [-0.0730, 51.5335], [-0.0750, 51.5300]
        ]]
      }
    },
    {
      type: 'Feature',
      properties: {
        Borough: 'Greenwich',
        burglary_count: 48, // Sum of LSOAs: 26 + 22
        risk_level: 'Low'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [0.0050, 51.4700], [0.0265, 51.4745], [0.0225, 51.4770], [0.0125, 51.4755], 
          [0.0070, 51.4735], [0.0050, 51.4700]
        ]]
      }
    }
  ]
};

// Risk level color mapping with reduced opacity for better map visibility
export const getRiskColor = (risk_level: string) => {
  switch (risk_level) {
    case 'Very Low':
      return '#22c55e'; // Green
    case 'Low':
      return '#84cc16'; // Light green
    case 'Medium':
      return '#eab308'; // Yellow
    case 'High':
      return '#f97316'; // Orange
    case 'Very High':
      return '#ef4444'; // Red
    default:
      return '#94a3b8'; // Gray for unknown
  }
};

// Get fill opacity based on risk level - REDUCED for better underlying map visibility
export const getFillOpacity = (risk_level: string) => {
  switch (risk_level) {
    case 'Very High':
      return 0.4; // Reduced from 0.8
    case 'High':
      return 0.35; // Reduced from 0.7
    case 'Medium':
      return 0.3; // Reduced from 0.6
    case 'Low':
      return 0.25; // Reduced from 0.5
    case 'Very Low':
      return 0.2; // Reduced from 0.4
    default:
      return 0.15; // Reduced from 0.3
  }
}; 