import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { getBurglaryTimeSeries, getBurglaryForecast } from '../api/backendService';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const TimeSeriesForecasting = ({ selectedLSOA }) => {
  const [forecastData, setForecastData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [timeRange, setTimeRange] = useState('3m');
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        // Determine how many months of historical data to fetch based on selected time range
        const monthsToShow = timeRange === '3m' ? 3 : timeRange === '6m' ? 6 : timeRange === '1y' ? 12 : 36;
        
        // Fetch historical time series data
        const timeSeriesData = await getBurglaryTimeSeries(selectedLSOA);
        
        // Determine how many forecast periods to request
        const forecastPeriods = 6;
        
        // Fetch forecast data
        const forecast = await getBurglaryForecast(selectedLSOA, forecastPeriods);
        
        // Process data for visualization
        const processedData = processDataForVisualization(timeSeriesData, forecast, monthsToShow);
        setForecastData(processedData);
        
        setIsLoading(false);
      } catch (error) {
        console.error("Error loading forecast data:", error);
        setError("Failed to load forecast data. Please try again later.");
        setIsLoading(false);
      }
    };
    
    loadData();
  }, [selectedLSOA, timeRange]);

  const processDataForVisualization = (timeSeriesData, forecastData, monthsToShow) => {
    // If there's no data, return null
    if (!timeSeriesData || timeSeriesData.length === 0) {
      return null;
    }
    
    // Sort time series data by date
    const sortedTimeSeries = [...timeSeriesData].sort((a, b) => new Date(a.date) - new Date(b.date));
    
    // Take only the last X months based on timeRange
    const recentTimeSeries = sortedTimeSeries.slice(-monthsToShow);
    
    // Prepare labels and data arrays
    const labels = [];
    const historicalData = [];
    const forecastValues = [];
    const upperBound = [];
    const lowerBound = [];
    
    // Add historical data
    recentTimeSeries.forEach(point => {
      // Extract month and year from date (assumed format: YYYY-MM or YYYY-MM-DD)
      const dateStr = point.date;
      const date = new Date(dateStr);
      const monthYear = date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
      
      labels.push(monthYear);
      historicalData.push(point.burglary_count);
      forecastValues.push(null);
      upperBound.push(null);
      lowerBound.push(null);
    });
    
    // Add forecast data if available
    if (forecastData && forecastData.forecast) {
      // Get the last date from historical data
      const lastDate = new Date(recentTimeSeries[recentTimeSeries.length - 1].date);
      
      // Add forecasted points
      for (let i = 0; i < forecastData.forecast.length; i++) {
        // Calculate next month date
        const forecastDate = new Date(lastDate);
        forecastDate.setMonth(forecastDate.getMonth() + i + 1);
        const monthYear = forecastDate.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
        
        labels.push(`${monthYear} (forecast)`);
        historicalData.push(null);
        forecastValues.push(forecastData.forecast[i]);
        upperBound.push(forecastData.upper_bound[i]);
        lowerBound.push(forecastData.lower_bound[i]);
      }
    }
    
    return {
      labels,
      historicalData,
      forecastData: forecastValues,
      confidenceUpperBound: upperBound,
      confidenceLowerBound: lowerBound
    };
  };

  const handleRangeChange = (range) => {
    setTimeRange(range);
  };

  if (isLoading) {
    return (
      <div className="dashboard-card h-64 flex items-center justify-center">
        <div className="flex flex-col items-center">
          <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
          <p className="mt-3 text-gray-400">Loading forecast data...</p>
        </div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="dashboard-card h-64 flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-500 text-xl mb-2">⚠️</div>
          <p className="text-gray-300">{error}</p>
        </div>
      </div>
    );
  }
  
  if (!forecastData) {
    return (
      <div className="dashboard-card h-64 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-400">No data available for this area.</p>
          <p className="text-sm text-gray-500">Please select a different LSOA or time range.</p>
        </div>
      </div>
    );
  }
  
  const chartData = {
    labels: forecastData.labels || [],
    datasets: [
      {
        label: 'Historical Burglary Counts',
        data: forecastData.historicalData || [],
        borderColor: 'rgba(59, 130, 246, 1)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        pointBackgroundColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 2,
        tension: 0.3,
        fill: false,
      },
      {
        label: 'ARIMA Forecast',
        data: forecastData.forecastData || [],
        borderColor: 'rgba(139, 92, 246, 1)',
        backgroundColor: 'rgba(139, 92, 246, 0.1)',
        pointBackgroundColor: 'rgba(139, 92, 246, 1)',
        borderWidth: 2,
        borderDash: [5, 5],
        tension: 0.3,
        fill: false,
      },
      {
        label: 'Upper Confidence Bound',
        data: forecastData.confidenceUpperBound || [],
        borderColor: 'rgba(139, 92, 246, 0.3)',
        backgroundColor: 'transparent',
        pointBackgroundColor: 'transparent',
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
      },
      {
        label: 'Lower Confidence Bound',
        data: forecastData.confidenceLowerBound || [],
        borderColor: 'rgba(139, 92, 246, 0.3)',
        backgroundColor: 'rgba(139, 92, 246, 0.1)',
        pointBackgroundColor: 'transparent',
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.3,
        fill: '+1', // Fill between this dataset and the one above
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: 'rgba(209, 213, 219, 1)',
          font: {
            size: 11
          }
        }
      },
      title: {
        display: true,
        text: selectedLSOA 
          ? `ARIMA Forecast for ${selectedLSOA}` 
          : 'London-wide Burglary Forecast',
        color: 'rgba(255, 255, 255, 0.9)',
        font: {
          size: 16,
          weight: 'bold'
        }
      },
      tooltip: {
        intersect: false,
        mode: 'index',
        backgroundColor: 'rgba(17, 24, 39, 0.9)',
        titleColor: 'rgba(255, 255, 255, 0.9)',
        bodyColor: 'rgba(209, 213, 219, 1)',
        borderColor: 'rgba(75, 85, 99, 0.3)',
        borderWidth: 1,
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(75, 85, 99, 0.2)'
        },
        border: {
          color: 'rgba(75, 85, 99, 0.5)'
        },
        ticks: {
          color: 'rgba(209, 213, 219, 1)'
        },
        title: {
          display: true,
          text: 'Number of Burglaries',
          color: 'rgba(209, 213, 219, 1)'
        }
      },
      x: {
        grid: {
          color: 'rgba(75, 85, 99, 0.2)'
        },
        border: {
          color: 'rgba(75, 85, 99, 0.5)'
        },
        ticks: {
          color: 'rgba(209, 213, 219, 1)'
        }
      }
    }
  };

  // Determine if forecast shows an increase or decrease
  const forecastTrend = () => {
    if (!forecastData?.forecastData) return 'unknown';
    
    const forecastValues = forecastData.forecastData.filter(val => val !== null);
    if (forecastValues.length < 2) return 'unknown';
    
    const firstValue = forecastValues[0];
    const lastValue = forecastValues[forecastValues.length - 1];
    
    return lastValue > firstValue ? 'an increase' : 'a decrease';
  };

  return (
    <div className="dashboard-card">
      <div className="flex items-center justify-between card-header px-4 py-3">
        <h3 className="text-white text-lg font-semibold flex items-center">
          <span className="text-purple-400 mr-2">📈</span>
          Time Series Forecasting
        </h3>
        
        <div className="flex space-x-1">
          <button 
            onClick={() => handleRangeChange('3m')}
            className={`btn-secondary text-xs py-1 px-2 ${timeRange === '3m' 
              ? 'bg-purple-600/30 text-purple-400 border border-purple-600/40' 
              : 'bg-gray-800 text-gray-300 border border-gray-700'}`}
          >
            3 months
          </button>
          <button 
            onClick={() => handleRangeChange('6m')}
            className={`btn-secondary text-xs py-1 px-2 ${timeRange === '6m' 
              ? 'bg-purple-600/30 text-purple-400 border border-purple-600/40' 
              : 'bg-gray-800 text-gray-300 border border-gray-700'}`}
          >
            6 months
          </button>
          <button 
            onClick={() => handleRangeChange('1y')}
            className={`btn-secondary text-xs py-1 px-2 ${timeRange === '1y' 
              ? 'bg-purple-600/30 text-purple-400 border border-purple-600/40' 
              : 'bg-gray-800 text-gray-300 border border-gray-700'}`}
          >
            1 year
          </button>
          <button 
            onClick={() => handleRangeChange('all')}
            className={`btn-secondary text-xs py-1 px-2 ${timeRange === 'all' 
              ? 'bg-purple-600/30 text-purple-400 border border-purple-600/40' 
              : 'bg-gray-800 text-gray-300 border border-gray-700'}`}
          >
            All
          </button>
        </div>
      </div>

      <div className="p-4 bg-gray-900">
        <div className="h-64">
          <Line data={chartData} options={chartOptions} />
        </div>
        
        <div className="mt-4 p-3 bg-gray-800 rounded-lg border border-gray-700">
          <div className="flex items-start">
            <span className="text-purple-400 mr-2">ℹ️</span>
            <div className="text-sm text-gray-300">
              <p className="mb-1">
                Forecast using ARIMA (AutoRegressive Integrated Moving Average) with 95% confidence intervals.
              </p>
              <p>
                {selectedLSOA ? 
                  `The model predicts ${forecastTrend()} in burglaries for ${selectedLSOA} over the next 6 months.` :
                  'Select a region on the map for area-specific forecasting.'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TimeSeriesForecasting; 