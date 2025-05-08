// Service to communicate with the Flask backend
const API_BASE_URL = 'http://localhost:5000/api';

/**
 * Get time series data for a specific LSOA or all of London
 * 
 * @param {string} lsoa_code - Optional LSOA code to filter by
 * @returns {Promise} - Promise that resolves to time series data
 */
export const getBurglaryTimeSeries = async (lsoa_code = null) => {
  try {
    const url = `${API_BASE_URL}/burglary/time-series${lsoa_code ? `?lsoa_code=${lsoa_code}` : ''}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    const data = await response.json();
    return data.time_series;
  } catch (error) {
    console.error('Error fetching burglary time series:', error);
    throw error;
  }
};

/**
 * Get ARIMA forecast for future burglary counts
 * 
 * @param {string} lsoa_code - Optional LSOA code to filter by
 * @param {number} periods - Number of periods to forecast
 * @returns {Promise} - Promise that resolves to forecast data
 */
export const getBurglaryForecast = async (lsoa_code = null, periods = 6) => {
  try {
    const url = `${API_BASE_URL}/burglary/forecast?periods=${periods}${lsoa_code ? `&lsoa_code=${lsoa_code}` : ''}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching burglary forecast:', error);
    throw error;
  }
};

/**
 * Get optimized police allocation
 * 
 * @param {number} units - Number of police units to allocate
 * @param {string} lsoa_code - Optional LSOA code to filter by
 * @returns {Promise} - Promise that resolves to police allocation data
 */
export const getPoliceAllocation = async (units = 8, lsoa_code = null) => {
  try {
    const url = `${API_BASE_URL}/police/optimize?units=${units}${lsoa_code ? `&lsoa_code=${lsoa_code}` : ''}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    const data = await response.json();
    return data.police_allocation;
  } catch (error) {
    console.error('Error fetching police allocation:', error);
    throw error;
  }
};

/**
 * Get IMD (Indices of Multiple Deprivation) data for a specific LSOA
 * 
 * @param {string} lsoa_code - LSOA code
 * @returns {Promise} - Promise that resolves to IMD data
 */
export const getLsoaWellbeingData = async (lsoa_code) => {
  try {
    const url = `${API_BASE_URL}/imd/lsoa/${lsoa_code}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      if (response.status === 404) {
        return null; // LSOA not found
      }
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error(`Error fetching IMD data for LSOA ${lsoa_code}:`, error);
    throw error;
  }
};

/**
 * Get EMMIE framework scores
 * 
 * @returns {Promise} - Promise that resolves to EMMIE framework scores
 */
export const getEmmieScores = async () => {
  try {
    const url = `${API_BASE_URL}/emmie/scores`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching EMMIE scores:', error);
    throw error;
  }
};

/**
 * Get list of all LSOAs with names
 * 
 * @returns {Promise} - Promise that resolves to list of LSOAs
 */
export const getLsoaList = async () => {
  try {
    const url = `${API_BASE_URL}/lsoa/list`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    const data = await response.json();
    return data.lsoas;
  } catch (error) {
    console.error('Error fetching LSOA list:', error);
    throw error;
  }
};

/**
 * Get GeoJSON for London LSOAs with burglary data
 * 
 * @returns {Promise} - Promise that resolves to GeoJSON data
 */
export const getLsoaGeoJson = async () => {
  try {
    const url = `${API_BASE_URL}/geojson/lsoa`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching LSOA GeoJSON:', error);
    throw error;
  }
}; 