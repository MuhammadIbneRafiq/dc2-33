// Comprehensive hardcoded data service for frontend-only operation
// No backend dependencies - All data is realistic and London-focused

export interface TimeSeriesPoint {
  date: string;
  value: number;
  burglary_count: number;
}

export interface ForecastData {
  forecast: number[];
  dates: string[];
  confidence_interval: {
    lower: number[];
    upper: number[];
  };
}

export interface LSOAWellbeingData {
  lsoa_code: string;
  borough: string;
  population: number;
  deprivation_score: number;
  crime_rate: number;
  unemployment_rate: number;
  average_income: number;
  education_score: number;
  health_score: number;
  housing_score: number;
  safety_score: number;
  community_cohesion: number;
}

export interface PoliceAllocationData {
  clusters: Array<{
    id: string;
    center: [number, number];
    risk_level: string;
    recommended_units: number;
    patrol_type: 'foot' | 'vehicle';
    effectiveness: number;
  }>;
  summary: {
    total_units: number;
    coverage_percentage: number;
    average_response_time: number;
    high_risk_areas: number;
  };
}

export interface BurglaryTimeSeries {
  time_series: TimeSeriesPoint[];
  total_count: number;
  average_per_month: number;
  trend: 'increasing' | 'decreasing' | 'stable';
}

// Generate realistic time series data for London burglaries
const generateTimeSeriesData = (lsoa_code?: string, days: number = 90): BurglaryTimeSeries => {
  const data: TimeSeriesPoint[] = [];
  const startDate = new Date();
  startDate.setDate(startDate.getDate() - days);
  
  // Base crime rate varies by LSOA - higher in central London
  const baseRate = lsoa_code ? 
    (lsoa_code.includes('E01004') || lsoa_code.includes('E01000001') ? 8 : // Westminster/City
     lsoa_code.includes('E01004314') ? 12 : // Tower Hamlets
     lsoa_code.includes('E01002361') ? 9 : // Hackney
     5) : 7; // Default
  
  let totalCount = 0;
  
  for (let i = 0; i < days; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    
    // Add seasonal and weekly patterns
    const dayOfWeek = date.getDay();
    const month = date.getMonth();
    
    // Weekend effect (higher crime Friday-Sunday)
    const weekendMultiplier = [0.8, 0.7, 0.9, 0.9, 1.0, 1.3, 1.4][dayOfWeek];
    
    // Seasonal effect (higher in winter months)
    const seasonalMultiplier = [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.4][month];
    
    // Random variation
    const randomFactor = 0.5 + Math.random();
    
    const burglaryCount = Math.round(baseRate * weekendMultiplier * seasonalMultiplier * randomFactor);
    totalCount += burglaryCount;
    
    data.push({
      date: date.toISOString().split('T')[0],
      value: burglaryCount,
      burglary_count: burglaryCount
    });
  }
  
  return {
    time_series: data,
    total_count: totalCount,
    average_per_month: Math.round((totalCount / days) * 30),
    trend: totalCount > (days * baseRate * 0.9) ? 'increasing' : 
           totalCount < (days * baseRate * 0.8) ? 'decreasing' : 'stable'
  };
};

// Generate realistic forecast data using SARIMA-like patterns
const generateForecastData = (lsoa_code: string): ForecastData => {
  const forecastPeriods = 30; // 30 days ahead
  const baseRate = lsoa_code.includes('E01004') ? 8 : 
                   lsoa_code.includes('E01004314') ? 12 : 
                   lsoa_code.includes('E01002361') ? 9 : 5;
  
  const forecast: number[] = [];
  const dates: string[] = [];
  const lower: number[] = [];
  const upper: number[] = [];
  
  const startDate = new Date();
  
  for (let i = 1; i <= forecastPeriods; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    
    // Trend component (slight increase over time)
    const trend = baseRate + (i * 0.02);
    
    // Seasonal component (weekly pattern)
    const dayOfWeek = date.getDay();
    const weekendMultiplier = [0.8, 0.7, 0.9, 0.9, 1.0, 1.3, 1.4][dayOfWeek];
    
    // Add some noise for realism
    const noise = (Math.random() - 0.5) * 2;
    
    const predicted = Math.max(0, Math.round(trend * weekendMultiplier + noise));
    const confidence = predicted * 0.3; // 30% confidence interval
    
    forecast.push(predicted);
    dates.push(date.toISOString().split('T')[0]);
    lower.push(Math.max(0, Math.round(predicted - confidence)));
    upper.push(Math.round(predicted + confidence));
  }
  
  return {
    forecast,
    dates,
    confidence_interval: { lower, upper }
  };
};

// Generate realistic LSOA wellbeing data
const generateLSOAWellbeingData = (lsoa_code: string): LSOAWellbeingData => {
  // Map LSOA codes to borough names
  const boroughMap: { [key: string]: string } = {
    'E01000001': 'City of London',
    'E01000002': 'City of London',
    'E01000003': 'City of London',
    'E01004736': 'Westminster',
    'E01004737': 'Westminster',
    'E01004738': 'Westminster',
    'E01004739': 'Westminster',
    'E01000932': 'Camden',
    'E01000933': 'Camden',
    'E01000934': 'Camden',
    'E01000935': 'Camden',
    'E01002766': 'Islington',
    'E01002767': 'Islington',
    'E01002768': 'Islington',
    'E01004314': 'Tower Hamlets',
    'E01004315': 'Tower Hamlets',
    'E01004316': 'Tower Hamlets',
    'E01003923': 'Southwark',
    'E01003924': 'Southwark',
    'E01003925': 'Southwark',
    'E01003045': 'Lambeth',
    'E01003046': 'Lambeth',
    'E01002959': 'Kensington and Chelsea',
    'E01002960': 'Kensington and Chelsea',
    'E01002361': 'Hackney',
    'E01002362': 'Hackney',
    'E01002289': 'Greenwich',
    'E01002290': 'Greenwich'
  };
  
  const borough = boroughMap[lsoa_code] || 'Unknown';
  
  // Generate realistic data based on borough characteristics
  const isAffluentArea = ['City of London', 'Kensington and Chelsea'].includes(borough);
  const isCentralArea = ['Westminster', 'Camden'].includes(borough);
  const isEastLondon = ['Tower Hamlets', 'Hackney'].includes(borough);
  
  return {
    lsoa_code,
    borough,
    population: 1200 + Math.floor(Math.random() * 800), // 1200-2000
    deprivation_score: isAffluentArea ? 15 + Math.random() * 20 : 
                      isCentralArea ? 35 + Math.random() * 30 :
                      isEastLondon ? 55 + Math.random() * 35 : 
                      40 + Math.random() * 30,
    crime_rate: isAffluentArea ? 12 + Math.random() * 8 : 
               isCentralArea ? 25 + Math.random() * 15 :
               isEastLondon ? 35 + Math.random() * 20 : 
               22 + Math.random() * 12,
    unemployment_rate: isAffluentArea ? 2 + Math.random() * 3 : 
                      isCentralArea ? 6 + Math.random() * 4 :
                      isEastLondon ? 12 + Math.random() * 8 : 
                      8 + Math.random() * 5,
    average_income: isAffluentArea ? 65000 + Math.random() * 35000 : 
                   isCentralArea ? 45000 + Math.random() * 20000 :
                   isEastLondon ? 28000 + Math.random() * 12000 : 
                   35000 + Math.random() * 15000,
    education_score: isAffluentArea ? 85 + Math.random() * 10 : 
                    isCentralArea ? 75 + Math.random() * 15 :
                    isEastLondon ? 60 + Math.random() * 20 : 
                    70 + Math.random() * 15,
    health_score: isAffluentArea ? 80 + Math.random() * 15 : 
                 isCentralArea ? 70 + Math.random() * 20 :
                 isEastLondon ? 55 + Math.random() * 25 : 
                 65 + Math.random() * 20,
    housing_score: isAffluentArea ? 90 + Math.random() * 8 : 
                  isCentralArea ? 65 + Math.random() * 20 :
                  isEastLondon ? 45 + Math.random() * 25 : 
                  60 + Math.random() * 20,
    safety_score: isAffluentArea ? 85 + Math.random() * 10 : 
                 isCentralArea ? 60 + Math.random() * 20 :
                 isEastLondon ? 45 + Math.random() * 25 : 
                 65 + Math.random() * 20,
    community_cohesion: isAffluentArea ? 75 + Math.random() * 20 : 
                       isCentralArea ? 60 + Math.random() * 25 :
                       isEastLondon ? 50 + Math.random() * 30 : 
                       65 + Math.random() * 25
  };
};

// Generate realistic police allocation data
const generatePoliceAllocationData = (): PoliceAllocationData => {
  // London police stations and high-crime areas
  const londonHotspots = [
    { center: [-0.1276, 51.5074], risk: 'Very High', area: 'Westminster' }, // Central London
    { center: [-0.0759, 51.5051], risk: 'High', area: 'Tower Hamlets' }, // Canary Wharf area
    { center: [-0.1174, 51.5200], risk: 'High', area: 'Camden' }, // Camden Market
    { center: [-0.1599, 51.4975], risk: 'Medium', area: 'Westminster' }, // Victoria
    { center: [-0.0615, 51.5155], risk: 'High', area: 'Tower Hamlets' }, // Bethnal Green
    { center: [-0.1057, 51.4879], risk: 'Medium', area: 'Southwark' }, // London Bridge
  ];
  
  const clusters = londonHotspots.map((spot, index) => ({
    id: `cluster_${index + 1}`,
    center: spot.center as [number, number],
    risk_level: spot.risk,
    recommended_units: spot.risk === 'Very High' ? 8 : spot.risk === 'High' ? 5 : 3,
    patrol_type: (spot.risk === 'Very High' || Math.random() > 0.6) ? 'vehicle' : 'foot' as 'foot' | 'vehicle',
    effectiveness: 75 + Math.random() * 20 // 75-95%
  }));
  
  const totalUnits = clusters.reduce((sum, cluster) => sum + cluster.recommended_units, 0);
  
  return {
    clusters,
    summary: {
      total_units: totalUnits,
      coverage_percentage: 68 + Math.random() * 15, // 68-83%
      average_response_time: 8 + Math.random() * 8, // 8-16 minutes
      high_risk_areas: clusters.filter(c => c.risk_level === 'Very High' || c.risk_level === 'High').length
    }
  };
};

// Hardcoded API service that mimics backend responses
export const hardcodedApi = {
  burglary: {
    getTimeSeries: async (params: { lsoa_code?: string; days?: number }): Promise<BurglaryTimeSeries> => {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 300 + Math.random() * 500));
      return generateTimeSeriesData(params.lsoa_code, params.days);
    },
    
    getForecast: async (params: { lsoa_code: string }): Promise<ForecastData> => {
      await new Promise(resolve => setTimeout(resolve, 400 + Math.random() * 600));
      return generateForecastData(params.lsoa_code);
    }
  },
  
  lsoa: {
    getWellbeingData: async (lsoa_code: string): Promise<LSOAWellbeingData> => {
      await new Promise(resolve => setTimeout(resolve, 200 + Math.random() * 300));
      return generateLSOAWellbeingData(lsoa_code);
    }
  },
  
  police: {
    optimize: async (): Promise<PoliceAllocationData> => {
      await new Promise(resolve => setTimeout(resolve, 800 + Math.random() * 700));
      return generatePoliceAllocationData();
    }
  }
};

// Mock additional chart data for analytics
export const generateAnalyticsData = () => ({
  crimeByType: {
    labels: ['Burglary', 'Theft', 'Violence', 'Drug Offenses', 'Public Order', 'Other'],
    data: [342, 567, 234, 123, 189, 98]
  },
  
  crimeByHour: {
    labels: Array.from({length: 24}, (_, i) => `${i}:00`),
    data: [12, 8, 5, 3, 2, 4, 8, 15, 22, 18, 16, 20, 25, 28, 32, 35, 38, 42, 45, 41, 35, 28, 22, 16]
  },
  
  crimeTrend: {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    data: [156, 142, 168, 178, 165, 189, 201, 195, 187, 176, 163, 158]
  },
  
  policeEffectiveness: {
    vehiclePatrols: 87,
    footPatrols: 92,
    responseTime: 12.3,
    solvedCases: 78
  }
}); 