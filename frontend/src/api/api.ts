// API service for external APIs only - UK Police API, ONS API, etc.
// NO backend, NO mock data - only real external APIs

// UK Police API endpoints
const POLICE_API_BASE = 'https://data.police.uk/api';
const ONS_API_BASE = 'https://api.beta.ons.gov.uk/v1';

// Helper function to fetch from UK Police API
const fetchPoliceAPI = async (endpoint: string) => {
  try {
    const response = await fetch(`${POLICE_API_BASE}${endpoint}`);
    if (!response.ok) {
      throw new Error(`Police API error: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Error fetching from Police API: ${endpoint}`, error);
    throw error;
  }
};

// Helper function to fetch from ONS API
const fetchONSAPI = async (endpoint: string) => {
  try {
    const response = await fetch(`${ONS_API_BASE}${endpoint}`);
    if (!response.ok) {
      throw new Error(`ONS API error: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Error fetching from ONS API: ${endpoint}`, error);
    throw error;
  }
};

// Get burglary crimes for a specific location and date
export const getBurglaryData = async (lat: number, lng: number, date?: string) => {
  const dateParam = date || new Date().toISOString().slice(0, 7); // YYYY-MM format
  return fetchPoliceAPI(`/crimes-street/burglary?lat=${lat}&lng=${lng}&date=${dateParam}`);
};

// Get all crimes for a specific location and date
export const getCrimeData = async (lat: number, lng: number, date?: string) => {
  const dateParam = date || new Date().toISOString().slice(0, 7);
  return fetchPoliceAPI(`/crimes-street/all-crime?lat=${lat}&lng=${lng}&date=${dateParam}`);
};

// Get crime data for multiple months
export const getCrimeDataRange = async (lat: number, lng: number, months: string[]) => {
  const promises = months.map(month => 
    fetchPoliceAPI(`/crimes-street/burglary?lat=${lat}&lng=${lng}&date=${month}`)
  );
  return Promise.all(promises);
};

// Get police forces
export const getPoliceForces = async () => {
  return fetchPoliceAPI('/forces');
};

// Get specific police force details
export const getPoliceForce = async (forceId: string) => {
  return fetchPoliceAPI(`/forces/${forceId}`);
};

// Get neighbourhoods for a force
export const getNeighbourhoods = async (forceId: string) => {
  return fetchPoliceAPI(`/forces/${forceId}/neighbourhoods`);
};

// Get neighbourhood details
export const getNeighbourhood = async (forceId: string, neighbourhoodId: string) => {
  return fetchPoliceAPI(`/${forceId}/${neighbourhoodId}`);
};

// Get outcome data for crimes
export const getCrimeOutcomes = async (crimeId: string) => {
  return fetchPoliceAPI(`/outcomes-for-crime/${crimeId}`);
};

// Generate months array for date ranges
export const generateMonthsArray = (startDate: string, endDate: string): string[] => {
  const start = new Date(startDate);
  const end = new Date(endDate);
  const months: string[] = [];
  
  const current = new Date(start.getFullYear(), start.getMonth(), 1);
  const endMonth = new Date(end.getFullYear(), end.getMonth(), 1);
  
  while (current <= endMonth) {
    const year = current.getFullYear();
    const month = String(current.getMonth() + 1).padStart(2, '0');
    months.push(`${year}-${month}`);
    current.setMonth(current.getMonth() + 1);
  }
  
  return months.slice(-12); // Limit to last 12 months for performance
};

// London borough coordinates for API calls
export const LONDON_BOROUGHS = [
  { name: 'Westminster', coords: [51.4975, -0.1357] as [number, number] },
  { name: 'Camden', coords: [51.5290, -0.1255] as [number, number] },
  { name: 'Islington', coords: [51.5362, -0.1034] as [number, number] },
  { name: 'Hackney', coords: [51.5450, -0.0553] as [number, number] },
  { name: 'Tower Hamlets', coords: [51.5203, -0.0293] as [number, number] },
  { name: 'Southwark', coords: [51.5032, -0.0851] as [number, number] },
  { name: 'Lambeth', coords: [51.4607, -0.1163] as [number, number] },
  { name: 'Kensington and Chelsea', coords: [51.4990, -0.1938] as [number, number] },
  { name: 'City of London', coords: [51.5156, -0.0919] as [number, number] },
  { name: 'Hammersmith and Fulham', coords: [51.4927, -0.2339] as [number, number] },
  { name: 'Wandsworth', coords: [51.4571, -0.1909] as [number, number] },
  { name: 'Merton', coords: [51.4098, -0.2108] as [number, number] },
  { name: 'Kingston upon Thames', coords: [51.4120, -0.3006] as [number, number] },
  { name: 'Richmond upon Thames', coords: [51.4613, -0.3037] as [number, number] },
  { name: 'Hounslow', coords: [51.4673, -0.3611] as [number, number] },
  { name: 'Hillingdon', coords: [51.5441, -0.4760] as [number, number] },
  { name: 'Ealing', coords: [51.5130, -0.3089] as [number, number] },
  { name: 'Brent', coords: [51.5588, -0.2817] as [number, number] },
  { name: 'Harrow', coords: [51.5898, -0.3346] as [number, number] },
  { name: 'Barnet', coords: [51.6252, -0.2000] as [number, number] },
  { name: 'Enfield', coords: [51.6523, -0.0799] as [number, number] },
  { name: 'Haringey', coords: [51.5906, -0.1119] as [number, number] },
  { name: 'Waltham Forest', coords: [51.5886, -0.0118] as [number, number] },
  { name: 'Redbridge', coords: [51.5590, 0.0741] as [number, number] },
  { name: 'Havering', coords: [51.5812, 0.2120] as [number, number] },
  { name: 'Barking and Dagenham', coords: [51.5607, 0.1557] as [number, number] },
  { name: 'Newham', coords: [51.5077, 0.0469] as [number, number] },
  { name: 'Greenwich', coords: [51.4892, 0.0648] as [number, number] },
  { name: 'Bexley', coords: [51.4549, 0.1505] as [number, number] },
  { name: 'Bromley', coords: [51.4039, 0.0144] as [number, number] },
  { name: 'Croydon', coords: [51.3762, -0.0982] as [number, number] },
  { name: 'Sutton', coords: [51.3618, -0.1945] as [number, number] },
  { name: 'Lewisham', coords: [51.4452, -0.0209] as [number, number] }
];

// Get burglary data for all London boroughs
export const getLondonBurglaryData = async (months: string[]) => {
  const allData: any[] = [];
  
  for (const borough of LONDON_BOROUGHS) {
    for (const month of months.slice(-3)) { // Last 3 months for performance
      try {
        const [lat, lng] = borough.coords;
        const crimeData = await getBurglaryData(lat, lng, month);
        
        if (Array.isArray(crimeData)) {
          crimeData.forEach((crime: any, index: number) => {
            if (crime.location && crime.location.latitude && crime.location.longitude) {
              allData.push({
                id: `${borough.name}-${month}-${index}`,
                lat: parseFloat(crime.location.latitude),
                lng: parseFloat(crime.location.longitude),
                borough: borough.name,
                month: month,
                category: crime.category,
                location_type: crime.location_type || 'Unknown',
                outcome_status: crime.outcome_status?.category || 'Under investigation',
                date: crime.month || month,
                street: crime.location.street?.name || 'Unknown Street'
              });
            }
          });
        }
        
        // Respectful delay to avoid rate limiting
        await new Promise(resolve => setTimeout(resolve, 200));
        
      } catch (error) {
        console.warn(`Error fetching data for ${borough.name} ${month}:`, error);
      }
    }
  }
  
  return allData;
};

// API object for easy access
export const api = {
  police: {
    getBurglaryData,
    getCrimeData,
    getCrimeDataRange,
    getForces: getPoliceForces,
    getForce: getPoliceForce,
    getNeighbourhoods,
    getNeighbourhood,
    getCrimeOutcomes,
    getLondonBurglaryData
  },
  utils: {
    generateMonthsArray,
    LONDON_BOROUGHS
  }
};

export default api; 