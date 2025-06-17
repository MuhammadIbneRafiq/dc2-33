import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import MapComponent from '@/components/map/MapComponent';
import RealBoundaryMap from '@/components/RealBoundaryMap';
// import { api } from '@/api/api'; // Remove backend API
import { hardcodedApi } from '@/data/hardcodedData'; // Use hardcoded data instead
import { Button } from "@/components/ui/button";
import LoadingScreen from '@/components/ui/loading-screen';
import { 
  Shield, 
  AlertTriangle,
  ArrowDown,
  TrendingUp,
  FileBarChart,
  X
} from "lucide-react";
import Header from '@/components/Header';
import Sidebar from '@/components/Sidebar';
import PoliceAllocation from '@/components/PoliceAllocation';
import DataAnalytics from '@/components/DataAnalytics';
import PoliceChat from '@/components/PoliceChat';
import TermsDialog from '@/components/TermsDialog';
import { motion } from 'framer-motion';

// Dashboard page component
const Dashboard = () => {
  const [selectedLSOA, setSelectedLSOA] = useState<string | null>(null);
  const [showPoliceAllocation, setShowPoliceAllocation] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingMessage, setLoadingMessage] = useState('Initializing application...');
  const [activeView, setActiveView] = useState('dashboard');
  const [showChatNotification, setShowChatNotification] = useState(true);
  const [showTermsDialog, setShowTermsDialog] = useState(false);
  
  // Prediction state
  const [predictionModel, setPredictionModel] = useState('sdgcn');
  const [showPredictions, setShowPredictions] = useState(false);
  
  // Date range from header slider
  const [dateRange, setDateRange] = useState<number[]>([30]);
  
  // Add state for number of points to display
  const [numPoints, setNumPoints] = useState(90); // Default to 90 points
  
  // Add map level state for LSOA/Borough switching
  const [mapLevel, setMapLevel] = useState<'lsoa' | 'borough'>('lsoa');
  
  // Add prediction factors state
  const [selectedFactors, setSelectedFactors] = useState<string[]>([
    'High population density',
    'Poor street lighting',
    'Proximity to transport',
    'Low income area'
  ]);

  // Add time series data state for real-time updates
  const [realTimeSeriesData, setTimeSeriesData] = useState<any>(null);

  // Available prediction factors
  const availableFactors = [
    'High population density',
    'Poor street lighting', 
    'Proximity to transport',
    'Low income area',
    'High unemployment',
    'Poor housing quality',
    'Limited CCTV coverage',
    'High student population',
    'Tourist area',
    'Commercial district',
    'Night economy',
    'Social housing',
    'Poor community cohesion',
    'Drug-related activity',
    'Gang presence'
  ];
  
  // Handle date range changes from header
  const handleDateRangeChange = (newRange: number[]) => {
    setDateRange(newRange);
    // Reset predictions when date range changes
    setShowPredictions(false);
    
    // Trigger crime data fetch for new date range
    const endDate = new Date().toISOString().split('T')[0];
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - newRange[0]);
    
    // Emit event to MapComponent to fetch new data
    window.dispatchEvent(new CustomEvent('dateRangeChanged', { 
      detail: { 
        startDate: startDate.toISOString().split('T')[0],
        endDate: endDate,
        days: newRange[0]
      } 
    }));
    
    console.log(`📅 Date range changed: ${newRange[0]} days (${startDate.toISOString().split('T')[0]} to ${endDate})`);
  };
  
  // Handle initial loading
  useEffect(() => {
    const loadingMessages = [
      'Loading crime data...',
      'Initializing SARIMA models...',
      'Preparing geospatial data...',
      'Finalizing dashboard...'
    ];
    
    let messageIndex = 0;
    const messageInterval = setInterval(() => {
      if (messageIndex < loadingMessages.length) {
        setLoadingMessage(loadingMessages[messageIndex]);
        messageIndex++;
      } else {
        clearInterval(messageInterval);
        setTimeout(() => {
          setIsLoading(false);
          
          // Check if terms already accepted to prevent duplicate popups
          const termsAccepted = localStorage.getItem('termsAccepted');
          if (!termsAccepted) {
            setShowTermsDialog(true);
          } else {
            setShowTermsDialog(false);
          }
          
        }, 1000);
      }
    }, 800);
    
    return () => clearInterval(messageInterval);
  }, []);
  
  // Hide chat notification after 8 seconds
  useEffect(() => {
    if (!isLoading && showChatNotification) {
      const timer = setTimeout(() => {
        setShowChatNotification(false);
      }, 8000);
      return () => clearTimeout(timer);
    }
  }, [isLoading, showChatNotification]);
  
  // Fetch police allocation data - NOW USING HARDCODED DATA
  const { 
    data: policeAllocationData,
    isLoading: isLoadingPoliceData 
  } = useQuery({
    queryKey: ['policeAllocation'],
    queryFn: () => hardcodedApi.police.optimize(), // Use hardcoded API
    enabled: showPoliceAllocation,
    retry: 1,
    retryDelay: 1000
  });
  
  // Fetch LSOA data for selected LSOA - NOW USING HARDCODED DATA
  const {
    data: lsoaData,
    isLoading: isLoadingLsoaData
  } = useQuery({
    queryKey: ['lsoaData', selectedLSOA],
    queryFn: () => selectedLSOA ? hardcodedApi.lsoa.getWellbeingData(selectedLSOA) : null, // Use hardcoded API
    enabled: !!selectedLSOA,
    retry: 1,
    retryDelay: 1000
  });
  
  // Fetch SARIMA forecast data - NOW USING HARDCODED DATA
  const {
    data: forecastData,
    isLoading: isLoadingForecast
  } = useQuery({
    queryKey: ['forecast', selectedLSOA],
    queryFn: () => selectedLSOA ? hardcodedApi.burglary.getForecast({ lsoa_code: selectedLSOA }) : null, // Use hardcoded API
    enabled: !!selectedLSOA,
    retry: 1,
    retryDelay: 1000
  });
  
  // Fetch Burglary Time Series Data - NOW USING HARDCODED DATA
  const { 
    data: timeSeriesData, 
    isLoading: isLoadingTimeSeries,
    error: errorTimeSeries,
  } = useQuery({
    queryKey: ['burglaryTimeSeries', selectedLSOA, dateRange, numPoints], // Include numPoints if it affects API call
    queryFn: async () => {
      // Determine the number of days for historical data based on numPoints or dateRange[0]
      // The API might expect 'days' or a similar parameter.
      // For now, let's assume numPoints dictates the length of historical data requested.
      // Adjust the 'days' parameter based on your API's expectation.
      const daysToFetch = dateRange[0] || numPoints; // Example logic, adjust as needed
      return hardcodedApi.burglary.getTimeSeries({ // Use hardcoded API
        lsoa_code: selectedLSOA || undefined, // Pass LSOA code if selected
        days: daysToFetch // Or another relevant parameter like 'limit' or 'count'
      });
    },
    enabled: true, // Fetch whenever key parameters change
    retry: 1,
  });
  
  // Handle LSOA selection
  const handleLSOASelect = (lsoa: string) => {
    setSelectedLSOA(lsoa);
  };
  
  // Toggle police allocation
  const handleTogglePoliceAllocation = () => {
    setShowPoliceAllocation(!showPoliceAllocation);
  };
  
  // Handle terms agreement
  const handleTermsAccept = () => {
    setShowTermsDialog(false);
  };
  
  // Toggle prediction factor
  const toggleFactor = (factor: string) => {
    setSelectedFactors(prev => 
      prev.includes(factor) 
        ? prev.filter(f => f !== factor)
        : [...prev, factor]
    );
    
    // Regenerate predictions when factors change
    if (showPredictions) {
      console.log(`🔄 Updating predictions with factor: ${factor}`);
      handleGeneratePrediction();
    }
  };

  // Enhanced prediction generation with factors
  const handleGeneratePrediction = () => {
    setShowPredictions(true);
    console.log(`🎯 Generating predictions with ${selectedFactors.length} factors:`, selectedFactors);
    
    // Trigger prediction update event
    window.dispatchEvent(new CustomEvent('predictionUpdate', { 
      detail: { 
        model: predictionModel,
        factors: selectedFactors,
        dateRange: dateRange
      } 
    }));
  };

  // Listen for LSOA selection to update factors
  useEffect(() => {
    const handleLSOASelection = (event: CustomEvent) => {
      const { lsoaCode, socioData } = event.detail;
      console.log(`📍 LSOA ${lsoaCode} selected with socio data:`, socioData);
      
      // Auto-select relevant factors based on socio-economic data
      const autoFactors = [];
      if (socioData.imd_decile <= 3) autoFactors.push('Low income area');
      if (socioData.crime_rank <= 5000) autoFactors.push('High crime area');
      if (socioData.employment_rank <= 5000) autoFactors.push('High unemployment');
      if (socioData.housing_rank <= 5000) autoFactors.push('Poor housing quality');
      
      setSelectedFactors(prev => [...new Set([...prev, ...autoFactors])]);
    };

    const handleCrimeDataUpdate = (event: CustomEvent) => {
      const { total_crimes, time_series, detailed_crimes, date_range } = event.detail;
      console.log('📊 Crime data updated in Dashboard:', { total_crimes, time_series: time_series?.length, date_range });
      
      // Update time series data for the temporal graph
      setTimeSeriesData({
        time_series: time_series || [],
        total_crimes,
        detailed_crimes: detailed_crimes || [],
        date_range
      });
      
      // Update loading state
      setIsLoading(false);
    };

    const handleCrimeDataLoading = (event: CustomEvent) => {
      const { loading, progress, total, error } = event.detail;
      
      if (error) {
        console.warn('⚠️ Error loading crime data');
        setIsLoading(false);
      } else if (loading) {
        setIsLoading(true);
        console.log(`⏳ Loading crime data: ${progress}/${total}`);
      } else {
        setIsLoading(false);
        console.log('✅ Crime data loading complete');
      }
    };
    
    window.addEventListener('lsoaSelected', handleLSOASelection as EventListener);
    window.addEventListener('crimeDataUpdated', handleCrimeDataUpdate as EventListener);
    window.addEventListener('crimeDataLoading', handleCrimeDataLoading as EventListener);
    
    return () => {
      window.removeEventListener('lsoaSelected', handleLSOASelection as EventListener);
      window.removeEventListener('crimeDataUpdated', handleCrimeDataUpdate as EventListener);
      window.removeEventListener('crimeDataLoading', handleCrimeDataLoading as EventListener);
    };
  }, []);
  
  // Handle prediction model change
  const handleModelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setPredictionModel(e.target.value);
    setShowPredictions(false); // Reset predictions when model changes
  };
  
  // Generate random risk factors for the selected LSOA
  const lsoaRiskFactors = selectedLSOA
    ? [
        'High population density',
        'Low community cohesion',
        'Poor street lighting',
        'Proximity to major transport routes',
        'High unemployment rate',
        'High proportion of rental housing',
        'Recent increase in burglary incidents',
        'Limited natural surveillance',
        'Lack of CCTV coverage',
        'High turnover of residents',
      ].filter(() => Math.random() > 0.5) // Randomly select factors
    : undefined;
  
  // Enhanced time series panel with real API data
  const renderTimeSeriesPanel = () => {
    // Use real API data when available, otherwise use hardcoded data
    const useRealData = dateRange && dateRange.length > 0;
    
    let historicalPoints = [];
    let forecastPoints = [];
    
    if (useRealData && (realTimeSeriesData?.time_series || timeSeriesData?.time_series)) {
      // Use real API data (prioritize real-time data from crime API)
      const dataSource = realTimeSeriesData?.time_series || timeSeriesData?.time_series;
      historicalPoints = dataSource.map((p: any) => ({
        date: new Date(p.date),
        value: p.burglary_count,
      }));
      
      console.log(`📊 Using real data: ${historicalPoints.length} points for ${dateRange[0]} days`);
    } else {
      // Generate realistic mock data based on date range
      const days = dateRange?.[0] || numPoints;
      const months = Math.ceil(days / 30);
      
      historicalPoints = Array.from({ length: months }, (_, i) => {
        const date = new Date();
        date.setMonth(date.getMonth() - (months - 1 - i));
        
        // Seasonal patterns: more crime in winter and summer
        const month = date.getMonth();
        const isWinter = month >= 10 || month <= 2;
        const isSummer = month >= 5 && month <= 8;
        
        let baseValue = 35;
        if (isWinter) baseValue = 55; // Higher in winter
        else if (isSummer) baseValue = 45; // Moderate in summer
        
        // Add factor influence
        const factorMultiplier = 1 + (selectedFactors.length * 0.05);
        const value = Math.round(baseValue * factorMultiplier + (Math.random() - 0.5) * 15);
        
        return {
          date,
          value: Math.max(10, value)
        };
      });
      
      console.log(`🎲 Using mock data: ${historicalPoints.length} points with ${selectedFactors.length} factors`);
    }

    // Generate forecast based on selected factors
    if (showPredictions && forecastData?.forecast && forecastData?.dates) {
      forecastPoints = forecastData.forecast.map((val: number, index: number) => ({
        date: new Date(forecastData.dates[index]),
        value: val,
      }));
    } else if (showPredictions) {
      // Generate mock forecast influenced by selected factors
      const forecastLength = 30; // 30 days ahead
      const lastValue = historicalPoints[historicalPoints.length - 1]?.value || 35;
      
      forecastPoints = Array.from({ length: forecastLength }, (_, i) => {
        const date = new Date();
        date.setDate(date.getDate() + i + 1);
        
        // Factor influence on predictions
        const factorInfluence = selectedFactors.length * 0.1;
        const trend = 1 + factorInfluence + (Math.random() - 0.5) * 0.2;
        const value = Math.round(lastValue * trend + (Math.random() - 0.5) * 10);
        
        return {
          date,
          value: Math.max(5, value)
        };
      });
      
      console.log(`🔮 Generated forecast: ${forecastPoints.length} points influenced by ${selectedFactors.length} factors`);
    }
    
    const allPoints = [...historicalPoints, ...forecastPoints];
    
    if (allPoints.length === 0) {
      return (
        <div className="mt-6 bg-gray-800/70 rounded-xl border border-gray-700/50 p-4 shadow-lg text-center text-white">
          📊 No data available for selected date range
        </div>
      );
    }

    // Chart dimensions and scales
    const width = 450;
    const height = 280;
    const margin = 50;
    
    // Safe scaling functions with NaN protection
    const xScale = (i: number) => {
      if (!allPoints.length || isNaN(i)) return margin;
      return margin + (i / Math.max(allPoints.length - 1, 1)) * (width - 2 * margin);
    };
    
    const yScale = (value: number) => {
      if (!allPoints.length || isNaN(value)) return height - margin;
      const values = allPoints.map(p => p.value).filter(v => !isNaN(v));
      if (values.length === 0) return height - margin;
      
      const minY = Math.min(...values) * 0.9;
      const maxY = Math.max(...values) * 1.1;
      const range = maxY - minY;
      
      if (range === 0) return height - margin - (height - 2 * margin) / 2;
      return height - margin - ((value - minY) / range) * (height - 2 * margin);
    };

    // Safe axis ticks with NaN protection
    const validValues = allPoints.map(p => p.value).filter(v => !isNaN(v));
    const yTicks = validValues.length > 0 ? [
      Math.min(...validValues),
      (Math.min(...validValues) + Math.max(...validValues)) / 2,
      Math.max(...validValues)
    ] : [0, 25, 50];
    
    const xTicks = allPoints.length > 1 ? Array.from({ length: Math.min(6, allPoints.length) }, (_, i) => 
      Math.floor(i * (allPoints.length - 1) / Math.max(Math.min(5, allPoints.length - 1), 1))
    ) : [0];

    return (
      <div className="mt-6 bg-gray-800/70 rounded-xl border border-gray-700/50 p-4 shadow-lg">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-lg font-bold text-white">📈 Crime Forecasting</h3>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400">Range:</span>
            {[30, 90, 180, 365].map(n => (
              <button
                key={n}
                className={`px-2 py-1 rounded text-xs font-semibold border ${
                  (dateRange?.[0] || numPoints) === n 
                    ? 'bg-indigo-700 text-white border-indigo-500' 
                    : 'bg-gray-800 text-gray-300 border-gray-600 hover:bg-gray-700'
                } mx-1 transition-colors`}
                onClick={() => {
                  setDateRange([n]);
                  setNumPoints(n);
                  console.log(`📅 Date range changed to ${n} days`);
                }}
              >
                {n}d
              </button>
            ))}
          </div>
        </div>
        
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-400">Model:</div>
            <div className="flex flex-row gap-2">
              {['sarima', 'lstm', 'sdgcn'].map(model => (
                <button
                  key={model}
                  className={`px-4 py-1 rounded-lg text-xs font-semibold transition-colors ${
                    predictionModel === model 
                      ? 'bg-indigo-700 text-white' 
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                  onClick={() => setPredictionModel(model)}
                >
                  {model.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
          <button
            className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white px-4 py-1 rounded text-sm font-medium transition-all"
            onClick={handleGeneratePrediction}
          >
            {showPredictions ? '🔄 Update' : '🔮 Generate'} Forecast
          </button>
        </div>

        {/* Factors influence indicator */}
        {selectedFactors.length > 0 && (
          <div className="mb-3 p-2 bg-gray-900/50 rounded border border-gray-600">
            <div className="text-xs text-gray-400 mb-1">Active Factors ({selectedFactors.length}):</div>
            <div className="flex flex-wrap gap-1">
              {selectedFactors.slice(0, 3).map(factor => (
                <span key={factor} className="text-xs bg-blue-600/20 text-blue-300 px-2 py-1 rounded">
                  {factor}
                </span>
              ))}
              {selectedFactors.length > 3 && (
                <span className="text-xs text-gray-500">+{selectedFactors.length - 3} more</span>
              )}
            </div>
          </div>
        )}
        
        <div className="h-[280px] bg-gray-900/50 rounded-lg border border-gray-700/50 flex items-center justify-center">
          <svg width={width} height={height}>
            {/* Y axis */}
            <line x1={margin} y1={margin} x2={margin} y2={height - margin} stroke="#cbd5e1" strokeWidth={1.5} />
            {/* Y axis ticks and labels */}
            {yTicks.map((y, i) => (
              <g key={i}>
                <line x1={margin - 5} x2={margin} y1={yScale(y)} y2={yScale(y)} stroke="#cbd5e1" strokeWidth={1} />
                <text x={margin - 8} y={yScale(y) + 4} textAnchor="end" fontSize="11" fill="#cbd5e1">
                  {Math.round(y)}
                </text>
              </g>
            ))}
            
            {/* X axis */}
            <line x1={margin} y1={height - margin} x2={width - margin} y2={height - margin} stroke="#cbd5e1" strokeWidth={1.5} />
            {/* X axis ticks and labels */}
            {xTicks.map((idx, i) => (
              <g key={i}>
                <line x1={xScale(idx)} x2={xScale(idx)} y1={height - margin} y2={height - margin + 5} stroke="#cbd5e1" strokeWidth={1} />
                <text x={xScale(idx)} y={height - margin + 18} textAnchor="middle" fontSize="11" fill="#cbd5e1">
                  {allPoints[idx]?.date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                </text>
              </g>
            ))}
            
            {/* Y axis label */}
            <text x={margin - 35} y={height / 2} textAnchor="middle" fontSize="12" fill="#cbd5e1" 
                  transform={`rotate(-90,${margin - 35},${height / 2})`}>
              Burglary Count
            </text>
            
            {/* X axis label */}
            <text x={width / 2} y={height - margin + 35} textAnchor="middle" fontSize="12" fill="#cbd5e1">
              Date
            </text>
            
            {/* Historical line */}
            <polyline
              fill="none"
              stroke="#3b82f6"
              strokeWidth="2.5"
              points={historicalPoints.map((p, i) => `${xScale(i)},${yScale(p.value)}`).join(' ')}
            />
            
            {/* Predicted line */}
            {forecastPoints.length > 0 && (
              <polyline
                fill="none"
                stroke="#22c55e"
                strokeWidth="2.5"
                strokeDasharray="5,5"
                points={forecastPoints.map((p, i) => `${xScale(i + historicalPoints.length)},${yScale(p.value)}`).join(' ')}
              />
            )}
            
            {/* Data points */}
            {allPoints.map((p, i) => (
              <circle
                key={i}
                cx={xScale(i)}
                cy={yScale(p.value)}
                r={3}
                fill={i < historicalPoints.length ? '#3b82f6' : '#22c55e'}
                stroke="#fff"
                strokeWidth={1}
              />
            ))}
            
            {/* Transition line */}
            {forecastPoints.length > 0 && (
              <line
                x1={xScale(historicalPoints.length - 1) + 1}
                y1={margin}
                x2={xScale(historicalPoints.length - 1) + 1}
                y2={height - margin}
                stroke="#fbbf24"
                strokeDasharray="4 2"
                strokeWidth={2}
              />
            )}
          </svg>
        </div>
        
        <div className="flex justify-between mt-3 text-xs">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-blue-500 rounded-full mr-1"></div>
            <span className="text-gray-400">Historical ({historicalPoints.length} points)</span>
          </div>
          {forecastPoints.length > 0 && (
            <div className="flex items-center">
              <div className="w-3 h-3 bg-green-500 rounded-full mr-1"></div>
              <span className="text-gray-400">Predicted ({predictionModel.toUpperCase()})</span>
            </div>
          )}
          <div className="flex items-center">
            <div className="w-3 h-3 bg-yellow-500 rounded-full mr-1"></div>
            <span className="text-gray-400">Data Source: {useRealData ? 'Police API' : 'Mock'}</span>
          </div>
        </div>
      </div>
    );
  };
  
  // Add the time series panel to the Dashboard content
  const renderContent = () => {
    switch (activeView) {
      case 'dashboard':
        return (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 p-6">
            {/* Main content area */}
            <div className="lg:col-span-2 space-y-6">
              {/* Map Component with level controls */}
              <div className="h-[500px] rounded-xl shadow-2xl overflow-hidden border border-gray-700/50 relative">
                {/* Map Level Toggle Controls */}
                <div className="absolute top-4 left-4 z-[1000]">
                  <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-3 border border-gray-600 shadow-lg">
                    <div className="text-xs font-semibold text-gray-300 mb-2">🗺️ View Level</div>
                    <div className="flex space-x-1">
                      <button
                        onClick={() => {
                          setMapLevel('lsoa');
                          console.log('🎯 Switched to LSOA view');
                        }}
                        className={`px-3 py-2 text-xs font-medium rounded transition-all duration-200 ${
                          mapLevel === 'lsoa' 
                            ? 'bg-blue-600 text-white shadow-md transform scale-105' 
                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600 hover:text-white'
                        }`}
                      >
                        📍 LSOA
                      </button>
                      <button
                        onClick={() => {
                          setMapLevel('borough');
                          console.log('🏛️ Switched to Borough view');
                        }}
                        className={`px-3 py-2 text-xs font-medium rounded transition-all duration-200 ${
                          mapLevel === 'borough' 
                            ? 'bg-blue-600 text-white shadow-md transform scale-105' 
                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600 hover:text-white'
                        }`}
                      >
                        🏘️ Borough
                      </button>
                    </div>
                    
                    {/* Current level indicator */}
                    <div className="mt-2 text-xs text-center">
                      <span className="text-gray-400">Showing: </span>
                      <span className={`font-semibold ${mapLevel === 'lsoa' ? 'text-blue-300' : 'text-green-300'}`}>
                        {mapLevel === 'lsoa' ? 'Local Areas' : 'Boroughs'}
                      </span>
                    </div>
                  </div>
                </div>

                <MapComponent 
                  onLSOASelect={handleLSOASelect} 
                  onBoroughSelect={(borough) => {
                    console.log('🏛️ Borough selected:', borough);
                    // You can add borough-specific logic here
                    setSelectedLSOA(null); // Clear LSOA selection when borough is selected
                  }}
                  showPoliceAllocation={showPoliceAllocation}
                  selectedLSOA={selectedLSOA}
                  selectedBorough={null} // Add borough state if needed
                  showPredictions={showPredictions}
                  predictionModel={predictionModel}
                  dateRange={dateRange}
                  mapLevel={mapLevel} // Pass map level to component
                />
              </div>
              
              {/* Forecasting Models Section - Now below the map */}
              <div className="bg-gray-800 rounded-xl shadow-2xl border border-gray-700/50 p-6">
                <DataAnalytics 
                  selectedLsoaCode={selectedLSOA} 
                  lsoaWellbeingData={lsoaData} 
                  isLoadingLsoaData={isLoadingLsoaData} 
                />
              </div>
            </div>

            {/* Sidebar/Details Area */}
            <div className="lg:col-span-1 space-y-6">
              {/* Time Series Forecasting Panel - Now in sidebar */}
              {renderTimeSeriesPanel()}
              
              {/* Prediction Factors Panel */}
              {selectedLSOA && (
                <div className="bg-gray-800 rounded-xl shadow-2xl border border-gray-700/50 p-4">
                  <h3 className="text-lg font-semibold text-white mb-3 flex items-center">
                    🎯 Prediction Factors
                    <span className="ml-2 text-sm text-gray-400">({selectedFactors.length} selected)</span>
                  </h3>
                  
                  <div className="space-y-2 max-h-60 overflow-y-auto">
                    {availableFactors.map((factor) => (
                      <label 
                        key={factor}
                        className="flex items-center space-x-2 p-2 rounded hover:bg-gray-700/50 cursor-pointer transition-colors"
                      >
                        <input
                          type="checkbox"
                          checked={selectedFactors.includes(factor)}
                          onChange={() => toggleFactor(factor)}
                          className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                        />
                        <span className={`text-sm ${
                          selectedFactors.includes(factor) ? 'text-white' : 'text-gray-400'
                        }`}>
                          {factor}
                        </span>
                      </label>
                    ))}
                  </div>
                  
                  <div className="mt-4 pt-3 border-t border-gray-600">
                    <button
                      onClick={handleGeneratePrediction}
                      className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-medium py-2 px-4 rounded transition-all"
                    >
                      🔮 Update Predictions
                    </button>
                  </div>
                </div>
              )}
              
              <PoliceAllocation 
                onToggle={handleTogglePoliceAllocation}
                showPoliceAllocation={showPoliceAllocation}
              />
              
              {/* MASSIVE POLICE DEPLOYMENT INDICATOR */}
              {showPoliceAllocation && (
                <div className="bg-red-600 rounded-xl p-4 border-4 border-red-400 animate-pulse">
                  <h3 className="text-xl font-bold text-white mb-3 flex items-center">
                    🚨 EMERGENCY DEPLOYMENT ACTIVE 🚨
                  </h3>
                  <div className="space-y-2 text-white">
                    <div className="flex justify-between text-lg font-bold">
                      <span>Total Units Deployed:</span>
                      <span className="text-yellow-300">10,000+</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Armed Response Units:</span>
                      <span className="text-red-300">2,000</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Riot Control Units:</span>
                      <span className="text-orange-300">1,800</span>
                    </div>
                    <div className="flex justify-between">
                      <span>K9 Units:</span>
                      <span className="text-blue-300">1,600</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Patrol Units:</span>
                      <span className="text-green-300">4,600</span>
                    </div>
                    <div className="mt-3 text-center text-yellow-300 font-bold text-lg">
                      🚁 MAXIMUM SECURITY PROTOCOL 🚁
                    </div>
                  </div>
                </div>
              )}
              
              {/* Quick Stats Card */}
              <div className="bg-gray-800 rounded-xl p-4 border border-gray-700/50">
                <h3 className="text-lg font-semibold text-white mb-3">Quick Stats</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-300">Active Police Units</span>
                    <span className="text-green-400 font-semibold">
                      {showPoliceAllocation ? '10,142' : '142'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Coverage</span>
                    <span className="text-blue-400 font-semibold">
                      {showPoliceAllocation ? '98%' : '68%'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Risk Level</span>
                    <span className={`font-semibold ${showPoliceAllocation ? 'text-green-400' : 'text-yellow-400'}`}>
                      {showPoliceAllocation ? 'Low' : 'Medium'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Response Time</span>
                    <span className="text-purple-400 font-semibold">
                      {showPoliceAllocation ? '2.1 min' : '12.5 min'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
      
      case 'map':
        return (
          <div className="p-6">
            {/* Full-screen Map View */}
            <div className="h-[calc(100vh-120px)] rounded-xl shadow-2xl overflow-hidden border border-gray-700/50 relative">
              <MapComponent 
                onLSOASelect={handleLSOASelect} 
                showPoliceAllocation={showPoliceAllocation}
                selectedLSOA={selectedLSOA}
                showPredictions={showPredictions}
                predictionModel={predictionModel}
                dateRange={dateRange}
              />
              
              {/* Map controls overlay */}
              <div className="absolute top-4 right-4 z-[1000] space-y-2">
                <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-3 border border-gray-700/50">
                  <h3 className="text-white font-semibold mb-2">Map Controls</h3>
                  <button
                    onClick={handleGeneratePrediction}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded text-sm transition-colors"
                  >
                    {showPredictions ? 'Hide Predictions' : 'Show Predictions'}
                  </button>
                  <select
                    value={predictionModel}
                    onChange={handleModelChange}
                    className="w-full mt-2 bg-gray-700 text-white border border-gray-600 rounded px-2 py-1 text-sm"
                  >
                    <option value="sdgcn">SD-GCN Model</option>
                    <option value="sarima">SARIMA Model</option>
                    <option value="lstm">LSTM Model</option>
                  </select>
                </div>
              </div>
              
              {/* LSOA Info Panel */}
              {selectedLSOA && (
                <div className="absolute bottom-4 left-4 z-[1000] bg-gray-800/90 backdrop-blur-sm rounded-lg p-4 border border-gray-700/50 max-w-sm">
                  <h3 className="text-white font-semibold mb-2">Selected LSOA: {selectedLSOA}</h3>
                  {lsoaData && (
                    <div className="space-y-1 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-300">Borough:</span>
                        <span className="text-white">{lsoaData.borough}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-300">Population:</span>
                        <span className="text-white">{lsoaData.population.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-300">Crime Rate:</span>
                        <span className="text-white">{lsoaData.crime_rate.toFixed(1)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-300">Safety Score:</span>
                        <span className="text-white">{lsoaData.safety_score.toFixed(0)}/100</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        );
      
      case 'analytics':
        return (
          <div className="p-6 space-y-6">
            <h1 className="text-2xl font-bold text-white mb-6">Data Analytics</h1>
            
            {/* Analytics Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Enhanced Data Analytics Component */}
              <div className="lg:col-span-2">
                <DataAnalytics 
                  selectedLsoaCode={selectedLSOA} 
                  lsoaWellbeingData={lsoaData} 
                  isLoadingLsoaData={isLoadingLsoaData} 
                />
              </div>
              
              {/* Time Series Chart */}
              <div className="bg-gray-800 rounded-xl p-6 border border-gray-700/50">
                <h3 className="text-lg font-semibold text-white mb-4">Burglary Time Series</h3>
                {renderTimeSeriesPanel()}
              </div>
              
              {/* LSOA Details */}
              {selectedLSOA && lsoaData && (
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700/50">
                  <h3 className="text-lg font-semibold text-white mb-4">LSOA Details</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-300">LSOA Code:</span>
                      <span className="text-white font-mono">{lsoaData.lsoa_code}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Borough:</span>
                      <span className="text-white">{lsoaData.borough}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Population:</span>
                      <span className="text-white">{lsoaData.population.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Average Income:</span>
                      <span className="text-white">£{lsoaData.average_income.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Unemployment Rate:</span>
                      <span className="text-white">{lsoaData.unemployment_rate.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Deprivation Score:</span>
                      <span className="text-white">{lsoaData.deprivation_score.toFixed(1)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Education Score:</span>
                      <span className="text-white">{lsoaData.education_score.toFixed(0)}/100</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Health Score:</span>
                      <span className="text-white">{lsoaData.health_score.toFixed(0)}/100</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Housing Score:</span>
                      <span className="text-white">{lsoaData.housing_score.toFixed(0)}/100</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Safety Score:</span>
                      <span className="text-white">{lsoaData.safety_score.toFixed(0)}/100</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Community Cohesion:</span>
                      <span className="text-white">{lsoaData.community_cohesion.toFixed(0)}/100</span>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Forecast Data */}
              {selectedLSOA && forecastData && (
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700/50">
                  <h3 className="text-lg font-semibold text-white mb-4">Forecast Summary</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-300">Next 7 Days:</span>
                      <span className="text-white">{forecastData.forecast.slice(0, 7).reduce((a, b) => a + b, 0)} incidents</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Next 30 Days:</span>
                      <span className="text-white">{forecastData.forecast.reduce((a, b) => a + b, 0)} incidents</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Peak Day:</span>
                      <span className="text-white">{forecastData.dates[forecastData.forecast.indexOf(Math.max(...forecastData.forecast))]}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Peak Incidents:</span>
                      <span className="text-white">{Math.max(...forecastData.forecast)}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        );
      
      case 'chat':
        return <PoliceChat />;
      
      default:
        return (
          <div className="p-6">
            <div className="text-center text-gray-400">
              <h2 className="text-xl font-semibold mb-2">Select a view from the sidebar</h2>
              <p>Use the navigation menu to explore different sections of the dashboard.</p>
            </div>
          </div>
        );
    }
  };
  
  return (
    <div className="min-h-screen w-full bg-gray-900 flex">
      {/* Sidebar */}
      <Sidebar 
        activeView={activeView}
        setActiveView={setActiveView}
        showPoliceAllocation={showPoliceAllocation}
        onTogglePoliceAllocation={handleTogglePoliceAllocation}
        selectedLSOA={selectedLSOA}
      />
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col pl-[280px]">
        <Header 
          onDateRangeChange={handleDateRangeChange}
        />
        <main className="flex-1 overflow-auto">
          {renderContent()}
        </main>
        
        {/* Chat Notification */}
        {showChatNotification && (
          <motion.div 
            initial={{ x: 100, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 100, opacity: 0 }}
            className="fixed bottom-24 right-10 bg-blue-800/90 text-white p-4 rounded-lg shadow-xl border border-blue-700 max-w-xs z-50"
          >
            <div className="flex items-start">
              <div className="flex-1">
                <h3 className="font-bold mb-1">Police Chat Available!</h3>
                <p className="text-sm text-blue-100">The chat assistant has been restored. Click the message icon in the bottom right to get help with crime analysis.</p>
              </div>
              <button 
                onClick={() => setShowChatNotification(false)}
                className="text-blue-200 hover:text-white"
              >
                <X size={18} />
              </button>
            </div>
          </motion.div>
        )}
      </div>

      {/* Police Chat Component */}
      <PoliceChat 
        selectedLSOA={selectedLSOA}
        selectedAllocation={policeAllocationData ? { 
          vehiclePatrols: policeAllocationData.clusters ? Math.floor(policeAllocationData.clusters.length / 3) : 0,
          footPatrols: policeAllocationData.clusters ? Math.ceil(policeAllocationData.clusters.length * 2 / 3) : 0,
          avgEffectiveness: 85
        } : null}
      />
      

      
      {/* Terms and Services Dialog - Render this last to ensure it's on top */}
      <div className="relative z-[100000]">
        <TermsDialog
          open={showTermsDialog}
          onClose={() => {}}  // Prevent closing without accepting
          onAccept={handleTermsAccept}
        />
      </div>
    </div>
  );
};

export default Dashboard;
