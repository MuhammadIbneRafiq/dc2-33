// API service for connecting to the Flask backend

// Base URL for API requests
const API_BASE_URL = 'http://localhost:5000/api';

// Fallback data for when the API is not available
const MOCK_DATA = {
  lsoa: {
    list: [
      'E01000001', 'E01000005', 'E01032739', 'E01032740', 'E01000032'
    ],
    wellbeing: {
      name: 'London LSOA',
      risk_level: 'Medium',
      imd_rank: 5243,
      crime_score: 0.28,
      income_score: 0.31,
      employment_score: 0.25,
      health_score: 0.22,
      education_score: 0.19,
      housing_score: 0.35,
      living_environment_score: 0.33
    }
  },
  burglary: {
    timeSeries: {
      months: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
      counts: [35, 28, 32, 30, 25, 27, 31, 29, 36, 39, 33, 29]
    },
    forecast: {
      forecast: [33, 31, 34, 30, 28, 26, 32, 34, 35, 39, 31, 28],
      accuracy: "92.5%",
      aic: 876.32
    },
    correlation: {
      income: -0.72,
      employment: -0.65,
      health: -0.54,
      education: -0.48,
      housing: 0.67,
      living_environment: 0.71
    }
  },
  police: {
    optimize: {
      clusters: [
        { lat: 51.512, lon: -0.09, officer_count: 3, risk_score: 0.85 },
        { lat: 51.513, lon: -0.1, officer_count: 2, risk_score: 0.75 },
        { lat: 51.518, lon: -0.102, officer_count: 4, risk_score: 0.92 },
        { lat: 51.514, lon: -0.085, officer_count: 1, risk_score: 0.68 },
        { lat: 51.511, lon: -0.077, officer_count: 2, risk_score: 0.72 }
      ]
    }
  },
  emmie: {
    scores: {
      effect: 0.78,
      mechanism: 0.65,
      moderation: 0.72,
      implementation: 0.83,
      economic: 0.69,
      overall: 0.74
    }
  }
};

// Generic error handling function with mock data fallback
const handleApiError = (error: unknown, endpoint: string, mockData: any) => {
  console.error('API Error:', error);
  console.warn(`Using mock data for ${endpoint}`);
  return mockData;
};

// Check if the Flask backend is available
let isBackendAvailable = false;
fetch(`${API_BASE_URL}/lsoa/list`, { method: 'HEAD' })
  .then(response => {
    isBackendAvailable = response.ok;
    console.log(`Backend availability: ${isBackendAvailable ? 'Connected' : 'Not available'}`);
  })
  .catch(() => {
    console.warn('Backend server is not available. Using mock data fallbacks.');
    isBackendAvailable = false;
  });

// Generic fetch function with error handling and mock data fallback
const fetchApi = async (endpoint: string, mockDataPath: any, options: RequestInit = {}) => {
  if (!isBackendAvailable) {
    console.log(`Using mock data for ${endpoint}`);
    return mockDataPath;
  }
  
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `Request failed with status ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    return handleApiError(error, endpoint, mockDataPath);
  }
};

// API endpoints
export const api = {
  // LSOA endpoints
  lsoa: {
    getList: () => fetchApi('/lsoa/list', MOCK_DATA.lsoa.list),
    getWellbeingData: (lsoaCode: string) => fetchApi(`/imd/lsoa/${lsoaCode}`, MOCK_DATA.lsoa.wellbeing),
  },

  // Burglary data endpoints
  burglary: {
    getTimeSeries: (params?: { lsoa_code?: string }) => {
      const queryParams = params?.lsoa_code ? `?lsoa_code=${params.lsoa_code}` : '';
      return fetchApi(`/burglary/time-series${queryParams}`, MOCK_DATA.burglary.timeSeries);
    },
    getForecast: (params?: { lsoa_code?: string }) => {
      const queryParams = params?.lsoa_code ? `?lsoa_code=${params.lsoa_code}` : '';
      return fetchApi(`/burglary/forecast${queryParams}`, MOCK_DATA.burglary.forecast);
    },
    getCorrelation: () => fetchApi('/burglary/correlation', MOCK_DATA.burglary.correlation),
  },

  // Police allocation endpoints
  police: {
    optimize: (params?: { clusters?: number }) => {
      const queryParams = params?.clusters ? `?clusters=${params.clusters}` : '';
      return fetchApi(`/police/optimize${queryParams}`, MOCK_DATA.police.optimize);
    },
  },

  // EMMIE scores endpoints
  emmie: {
    getScores: (params?: { lsoa_code?: string }) => {
      const queryParams = params?.lsoa_code ? `?lsoa_code=${params.lsoa_code}` : '';
      return fetchApi(`/emmie/scores${queryParams}`, MOCK_DATA.emmie.scores);
    },
  },
};

export default api; 