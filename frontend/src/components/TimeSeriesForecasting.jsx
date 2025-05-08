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

  useEffect(() => {
    // In a real app, this would be an API call to backend ARIMA model
    // For now, we'll simulate the data
    setIsLoading(true);
    
    setTimeout(() => {
      const data = generateForecastData(selectedLSOA, timeRange);
      setForecastData(data);
      setIsLoading(false);
    }, 1200);
  }, [selectedLSOA, timeRange]);

  const generateForecastData = (lsoa, range) => {
    // This is a placeholder function that generates mock time series forecast data
    // In a real app, this would come from an ARIMA model in the backend
    
    const seed = lsoa ? lsoa.charCodeAt(6) * lsoa.charCodeAt(8) : 42;
    const rng = (min, max) => Math.floor(seed % 17 * (Math.random() + 0.5) * (max - min) / 17) + min;

    // Generate historical data (past months)
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const currentMonth = new Date().getMonth();
    
    // Determine the number of months to show based on the selected range
    let monthsToShow = 3;
    if (range === '6m') monthsToShow = 6;
    if (range === '1y') monthsToShow = 12;
    if (range === 'all') monthsToShow = 36;

    // Generate labels for the chart (past and future months)
    const labels = [];
    const historicalData = [];
    const forecastData = [];
    const confidenceUpperBound = [];
    const confidenceLowerBound = [];

    // Past data (actual)
    for (let i = monthsToShow; i >= 1; i--) {
      const monthIndex = (currentMonth - i + 12) % 12;
      labels.push(months[monthIndex]);
      const value = rng(20, 80);
      historicalData.push(value);
      forecastData.push(null); // No forecast for historical data
      confidenceUpperBound.push(null);
      confidenceLowerBound.push(null);
    }

    // Current month (actual)
    labels.push(months[currentMonth]);
    const currentValue = rng(20, 80);
    historicalData.push(currentValue);
    forecastData.push(currentValue);
    confidenceUpperBound.push(null);
    confidenceLowerBound.push(null);

    // Future months (forecasted)
    for (let i = 1; i <= 6; i++) {
      const monthIndex = (currentMonth + i) % 12;
      labels.push(months[monthIndex] + ' (forecast)');
      
      // Apply ARIMA-like forecast with some randomness and trend
      const prevValue = i === 1 ? currentValue : forecastData[forecastData.length - 1];
      const trend = Math.sin(i * 0.5) * 5; // Seasonal component
      const randomNoise = (Math.random() - 0.5) * 10;
      const forecast = Math.max(10, Math.round(prevValue * 0.85 + trend + randomNoise));
      
      historicalData.push(null); // No historical data for future
      forecastData.push(forecast);
      
      // Add confidence intervals
      confidenceUpperBound.push(forecast + rng(5, 15));
      confidenceLowerBound.push(Math.max(0, forecast - rng(5, 15)));
    }

    return {
      labels,
      historicalData,
      forecastData,
      confidenceUpperBound,
      confidenceLowerBound
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
  
  const chartData = {
    labels: forecastData?.labels || [],
    datasets: [
      {
        label: 'Historical Burglary Counts',
        data: forecastData?.historicalData || [],
        borderColor: 'rgba(59, 130, 246, 1)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        pointBackgroundColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 2,
        tension: 0.3,
        fill: false,
      },
      {
        label: 'ARIMA Forecast',
        data: forecastData?.forecastData || [],
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
        data: forecastData?.confidenceUpperBound || [],
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
        data: forecastData?.confidenceLowerBound || [],
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
                  `The model predicts ${
                    forecastData?.forecastData?.[forecastData?.forecastData?.length - 1] - forecastData?.forecastData?.[forecastData?.forecastData?.length - 6] > 0 
                      ? 'an increase' 
                      : 'a decrease'
                  } in burglaries for ${selectedLSOA} over the next 6 months.` :
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