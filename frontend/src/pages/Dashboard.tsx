import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import MapComponent from '@/components/map/MapComponent';
import { api } from '@/api/api';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import LoadingScreen from '@/components/ui/loading-screen';
import { 
  LayoutDashboard, 
  Map, 
  Shield, 
  BarChart3, 
  AlertTriangle,
  ChevronRight,
  ArrowUp,
  ArrowDown,
  TrendingUp,
  FileBarChart,
  X
} from "lucide-react";
import Header from '@/components/Header';
import Sidebar from '@/components/Sidebar';
import PoliceAllocation from '@/components/PoliceAllocation';
import DataAnalytics from '@/components/DataAnalytics';
import EmmieExplanation from '@/components/EmmieExplanation';
import PoliceChat from '@/components/PoliceChat';
import TermsDialog from '@/components/TermsDialog';
import TutorialVideo from '@/components/TutorialVideo';
import { motion } from 'framer-motion';

// Dashboard page component
const Dashboard = () => {
  const [selectedLSOA, setSelectedLSOA] = useState<string | null>(null);
  const [showPoliceAllocation, setShowPoliceAllocation] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingMessage, setLoadingMessage] = useState('Initializing application...');
  const [activeView, setActiveView] = useState('dashboard');
  const [showChatNotification, setShowChatNotification] = useState(true);
  const [showTermsDialog, setShowTermsDialog] = useState(true);
  const [showTutorial, setShowTutorial] = useState(false);
  
  // Prediction state
  const [predictionModel, setPredictionModel] = useState('sdgcn');
  const [showPredictions, setShowPredictions] = useState(false);
  
  // Date range from header slider
  const [dateRange, setDateRange] = useState<number[]>([30]);
  
  // Add state for number of points to display
  const [numPoints, setNumPoints] = useState(90); // Default to 90 points
  
  // Handle date range changes from header
  const handleDateRangeChange = (newRange: number[]) => {
    setDateRange(newRange);
    // Reset predictions when date range changes
    setShowPredictions(false);
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
          
          // Always show terms dialog - no condition check needed anymore
          // const termsAccepted = localStorage.getItem('termsAccepted');
          // if (!termsAccepted) {
          //   setShowTermsDialog(true);
          // }
          
          // Force terms dialog to show
          setShowTermsDialog(true);
          
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
  
  // Fetch police allocation data
  const { 
    data: policeAllocationData,
    isLoading: isLoadingPoliceData 
  } = useQuery({
    queryKey: ['policeAllocation'],
    queryFn: () => api.police.optimize(),
    enabled: showPoliceAllocation,
    retry: 1,
    retryDelay: 1000
  });
  
  // Fetch LSOA data for selected LSOA
  const {
    data: lsoaData,
    isLoading: isLoadingLsoaData
  } = useQuery({
    queryKey: ['lsoaData', selectedLSOA],
    queryFn: () => selectedLSOA ? api.lsoa.getWellbeingData(selectedLSOA) : null,
    enabled: !!selectedLSOA,
    retry: 1,
    retryDelay: 1000
  });
  
  // Fetch SARIMA forecast data
  const {
    data: forecastData,
    isLoading: isLoadingForecast
  } = useQuery({
    queryKey: ['forecast', selectedLSOA],
    queryFn: () => selectedLSOA ? api.burglary.getForecast({ lsoa_code: selectedLSOA }) : null,
    enabled: !!selectedLSOA,
    retry: 1,
    retryDelay: 1000
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
  
  // Handle watch tutorial
  const handleWatchTutorial = () => {
    setShowTermsDialog(false);
    setShowTutorial(true);
  };
  
  // Handle prediction generation
  const handleGeneratePrediction = () => {
    setShowPredictions(true);
  };
  
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
  
  if (isLoading) {
    return <LoadingScreen message={loadingMessage} />;
  }
  
  // Create a time series visualization component at the bottom of the map
  const renderTimeSeriesPanel = () => {
    const historyDays = Math.min(dateRange[0], numPoints);
    const futureDays = Math.min(dateRange[1], Math.floor(numPoints / 4)); // Limit future points for clarity
    // Generate mock historical data
    const historicalPoints = Array.from({ length: historyDays }, (_, i) => ({
      date: new Date(Date.now() - (historyDays - i) * 24 * 60 * 60 * 1000),
      value: Math.floor(15 + Math.random() * 25)
    }));
    // Generate mock forecast data
    const forecastPoints = Array.from({ length: futureDays }, (_, i) => {
      let baseValue = 20;
      if (predictionModel === 'sarima') {
        baseValue = 20 + 5 * Math.sin(i / 3);
      } else if (predictionModel === 'lstm') {
        baseValue = 22 - i * 0.3;
      } else {
        baseValue = 18 + i * 0.2;
      }
      return {
        date: new Date(Date.now() + i * 24 * 60 * 60 * 1000),
        value: Math.floor(baseValue + Math.random() * 4)
      };
    });
    // Combine for line chart
    const allPoints = [...historicalPoints, ...forecastPoints];
    // SVG line chart dimensions
    const width = 600;
    const height = 220;
    const margin = 40;
    const maxValue = Math.max(...allPoints.map(p => p.value), 1);
    const minValue = Math.min(...allPoints.map(p => p.value), 0);
    const yScale = v => height - margin - ((v - minValue) / (maxValue - minValue + 1e-6)) * (height - 2 * margin);
    const xScale = i => margin + (i / (allPoints.length - 1)) * (width - 2 * margin);
    // Y axis ticks
    const yTicks = Array.from({ length: 5 }, (_, i) => minValue + (i * (maxValue - minValue) / 4));
    // X axis ticks (show at most 6)
    const xTicks = Array.from({ length: Math.min(6, allPoints.length) }, (_, i) => Math.floor(i * (allPoints.length - 1) / (Math.min(5, allPoints.length - 1) || 1)));
    return (
      <div className="mt-6 bg-gray-800/70 rounded-xl border border-gray-700/50 p-4 shadow-lg">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-lg font-bold text-white">Time Series Forecasting</h3>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400">Points:</span>
            {[30, 90, 180, 365].map(n => (
              <button
                key={n}
                className={`px-2 py-1 rounded text-xs font-semibold border ${numPoints === n ? 'bg-indigo-700 text-white border-indigo-500' : 'bg-gray-800 text-gray-300 border-gray-600'} mx-1`}
                onClick={() => setNumPoints(n)}
              >
                {n}
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-400">Model:</div>
            <div className="flex flex-row gap-4">
              <Button 
                variant={predictionModel === 'sarima' ? 'default' : 'outline'} 
                size="sm"
                className={`px-6 py-2 rounded-lg font-semibold ${predictionModel === 'sarima' ? 'bg-indigo-700 text-white' : ''}`}
                onClick={() => setPredictionModel('sarima')}
              >
                SARIMA
              </Button>
              <Button 
                variant={predictionModel === 'lstm' ? 'default' : 'outline'} 
                size="sm"
                className={`px-6 py-2 rounded-lg font-semibold ${predictionModel === 'lstm' ? 'bg-indigo-700 text-white' : ''}`}
                onClick={() => setPredictionModel('lstm')}
              >
                LSTM
              </Button>
              <Button 
                variant={predictionModel === 'sdgcn' ? 'default' : 'outline'} 
                size="sm"
                className={`px-6 py-2 rounded-lg font-semibold ${predictionModel === 'sdgcn' ? 'bg-indigo-700 text-white' : ''}`}
                onClick={() => setPredictionModel('sdgcn')}
              >
                LSTM-GCN
              </Button>
            </div>
          </div>
          <Button 
            variant="default" 
            size="sm"
            onClick={handleGeneratePrediction}
          >
            {showPredictions ? 'Update Forecast' : 'Generate Forecast'}
          </Button>
        </div>
        <div className="h-[260px] bg-gray-900/50 rounded-lg border border-gray-700/50 flex items-center justify-center">
          <svg width={width} height={height}>
            {/* Y axis */}
            <line x1={margin} y1={margin} x2={margin} y2={height - margin} stroke="#cbd5e1" strokeWidth={1.5} />
            {/* Y axis ticks and labels */}
            {yTicks.map((y, i) => (
              <g key={i}>
                <line x1={margin - 5} x2={margin} y1={yScale(y)} y2={yScale(y)} stroke="#cbd5e1" strokeWidth={1} />
                <text x={margin - 8} y={yScale(y) + 4} textAnchor="end" fontSize="11" fill="#cbd5e1">{Math.round(y)}</text>
              </g>
            ))}
            {/* X axis */}
            <line x1={margin} y1={height - margin} x2={width - margin} y2={height - margin} stroke="#cbd5e1" strokeWidth={1.5} />
            {/* X axis ticks and labels */}
            {xTicks.map((idx, i) => (
              <g key={i}>
                <line x1={xScale(idx)} x2={xScale(idx)} y1={height - margin} y2={height - margin + 5} stroke="#cbd5e1" strokeWidth={1} />
                <text x={xScale(idx)} y={height - margin + 18} textAnchor="middle" fontSize="11" fill="#cbd5e1">
                  {allPoints[idx].date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                </text>
              </g>
            ))}
            {/* Y axis label */}
            <text x={margin - 30} y={height / 2} textAnchor="middle" fontSize="13" fill="#cbd5e1" transform={`rotate(-90,${margin - 30},${height / 2})`}>
              Residential Burglaries
            </text>
            {/* X axis label */}
            <text x={width / 2} y={height - margin + 36} textAnchor="middle" fontSize="13" fill="#cbd5e1">
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
            <polyline
              fill="none"
              stroke="#22c55e"
              strokeWidth="2.5"
              points={forecastPoints.map((p, i) => `${xScale(i + historicalPoints.length)},${yScale(p.value)}`).join(' ')}
            />
            {/* Dots for all points */}
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
            {/* Vertical line at transition */}
            <line
              x1={xScale(historicalPoints.length - 1) + 1}
              y1={margin}
              x2={xScale(historicalPoints.length - 1) + 1}
              y2={height - margin}
              stroke="#fbbf24"
              strokeDasharray="4 2"
              strokeWidth={2}
            />
          </svg>
        </div>
        <div className="flex justify-between mt-2">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-blue-500 rounded-full mr-1"></div>
            <span className="text-xs text-gray-400">Historical Data</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-green-500 rounded-full mr-1"></div>
            <span className="text-xs text-gray-400">Predicted ({predictionModel})</span>
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
          <div className="p-6">
            <h1 className="text-2xl font-bold text-white mb-6">
              <span className="mr-2">📊</span>
              London Residential Burglary Overview
            </h1>
            
            {/* Summary Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
              <div className="bg-gray-800/70 rounded-xl border border-gray-700/50 p-4 shadow-lg">
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="text-sm text-gray-400 mb-1">Total Burglaries</h3>
                    <p className="text-2xl font-bold text-white">3,487</p>
                    <p className="text-xs text-gray-400 mt-1">Last 30 days</p>
                  </div>
                  <div className="bg-blue-500/20 p-2 rounded-lg">
                    <FileBarChart className="text-blue-400" size={20} />
                  </div>
                </div>
                <div className="flex items-center mt-3">
                  <ArrowDown className="text-green-400 mr-1" size={14} />
                  <span className="text-green-400 text-sm font-medium">8.5%</span>
                  <span className="text-xs text-gray-400 ml-2">from previous period</span>
                </div>
              </div>
              
              <div className="bg-gray-800/70 rounded-xl border border-gray-700/50 p-4 shadow-lg">
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="text-sm text-gray-400 mb-1">High Risk Areas</h3>
                    <p className="text-2xl font-bold text-white">126</p>
                    <p className="text-xs text-gray-400 mt-1">Out of 633 total areas</p>
                  </div>
                  <div className="bg-red-500/20 p-2 rounded-lg">
                    <AlertTriangle className="text-red-400" size={20} />
                  </div>
                </div>
                <div className="flex items-center mt-3">
                  <ArrowDown className="text-green-400 mr-1" size={14} />
                  <span className="text-green-400 text-sm font-medium">2.3%</span>
                  <span className="text-xs text-gray-400 ml-2">from previous period</span>
                </div>
              </div>
              
              <div className="bg-gray-800/70 rounded-xl border border-gray-700/50 p-4 shadow-lg">
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="text-sm text-gray-400 mb-1">Deployed Officers</h3>
                    <p className="text-2xl font-bold text-white">{showPoliceAllocation ? "147" : "0"}</p>
                    <p className="text-xs text-gray-400 mt-1">Across hotspot areas</p>
                  </div>
                  <div className="bg-purple-500/20 p-2 rounded-lg">
                    <Shield className="text-purple-400" size={20} />
                  </div>
                </div>
                <div className="flex items-center mt-3">
                  <span className={`text-xs ${showPoliceAllocation ? 'text-blue-400' : 'text-gray-400'} ml-2`}>
                    {showPoliceAllocation ? "Resource allocation active" : "Not currently deployed"}
                  </span>
                </div>
              </div>
              
              <div className="bg-gray-800/70 rounded-xl border border-gray-700/50 p-4 shadow-lg">
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="text-sm text-gray-400 mb-1">Projected Reduction</h3>
                    <p className="text-2xl font-bold text-white">{showPoliceAllocation ? "32%" : "0%"}</p>
                    <p className="text-xs text-gray-400 mt-1">Potential with current allocation</p>
                  </div>
                  <div className="bg-green-500/20 p-2 rounded-lg">
                    <TrendingUp className="text-green-400" size={20} />
                  </div>
                </div>
                <div className="flex items-center mt-3">
                  {showPoliceAllocation ? (
                    <span className="text-xs text-green-400">
                      EMMIE-based optimization active
                    </span>
                  ) : (
                    <span className="text-xs text-gray-400">
                      Enable resource allocation to see projections
                    </span>
                  )}
                </div>
              </div>
            </div>
            
            {/* Main Dashboard Content */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 bg-gray-800/70 rounded-xl border border-gray-700/50 p-4 shadow-lg overflow-hidden">
                <h3 className="text-lg font-semibold text-white mb-4">London Burglary Risk Map</h3>
                <div className="h-[50vh]">
                  <MapComponent 
                    onLSOASelect={handleLSOASelect}
                    showPoliceAllocation={showPoliceAllocation}
                    selectedLSOA={selectedLSOA}
                    policeAllocationData={policeAllocationData}
                    showPredictions={showPredictions}
                    predictionModel={predictionModel}
                    dateRange={dateRange}
                  />
                </div>
                
                {/* Add the time series visualization below the map */}
                {renderTimeSeriesPanel()}
                
                <div className="flex items-center space-x-4 mt-4 p-2 bg-gray-900/50 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 rounded-full bg-red-500"></div>
                    <span className="text-sm text-gray-300">High Risk</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                    <span className="text-sm text-gray-300">Medium Risk</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 rounded-full bg-green-500"></div>
                    <span className="text-sm text-gray-300">Low Risk</span>
                  </div>
                  {showPoliceAllocation && (
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                      <span className="text-sm text-gray-300">Police Units</span>
                    </div>
                  )}
                </div>
              </div>
              
              <div className="flex flex-col gap-6">
                <div className="bg-gray-800/70 rounded-xl border border-gray-700/50 p-4 shadow-lg">
                  <h3 className="text-lg font-semibold text-white mb-4">Resource Allocation</h3>
                  <div className="mb-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-400">Allocation Status</span>
                      <div className="flex items-center">
                        <div 
                          className={`h-2 w-2 rounded-full mr-2 ${showPoliceAllocation ? 'bg-green-500' : 'bg-gray-500'}`}
                        ></div>
                        <span className={`text-sm ${showPoliceAllocation ? 'text-green-400' : 'text-gray-400'}`}>
                          {showPoliceAllocation ? 'Active' : 'Inactive'}
                        </span>
                      </div>
                    </div>
                    <Button 
                      variant={showPoliceAllocation ? "default" : "outline"} 
                      className="w-full" 
                      onClick={handleTogglePoliceAllocation}
                    >
                      {showPoliceAllocation ? 'Disable Allocation' : 'Enable Resource Allocation'}
                    </Button>
                  </div>
                  
                  {showPoliceAllocation && (
                    <div className="mt-4">
                      <div className="grid grid-cols-2 gap-2 mb-4">
                        <div className="bg-gray-900/50 p-2 rounded-lg">
                          <p className="text-xs text-gray-400">Allocation Method</p>
                          <p className="text-sm text-white">CPTED Principles</p>
                        </div>
                        <div className="bg-gray-900/50 p-2 rounded-lg">
                          <p className="text-xs text-gray-400">Coverage</p>
                          <p className="text-sm text-white">Police deployment based on environmental risk factors</p>
                        </div>
                      </div>
                      <div className="bg-blue-500/10 text-blue-300 p-3 rounded-md text-xs border border-blue-500/20">
                        <p>
                          Police resources are now allocated using CPTED (Crime Prevention Through Environmental Design) strategies. Officers are deployed to maximize natural surveillance, reinforce territoriality, control access, support positive activity, and maintain safe environments in high-risk areas.
                        </p>
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Crime Prediction Panel */}
                <div className="bg-gray-800/70 rounded-xl border border-gray-700/50 p-4 shadow-lg">
                  <h3 className="text-lg font-semibold text-white mb-4">Crime Prediction</h3>
                  
                  <div className="mb-4">
                    <label className="text-sm text-gray-400 block mb-2">Prediction Model</label>
                    <select 
                      className="w-full bg-gray-700/80 text-white text-sm rounded-md border border-gray-600 px-3 py-2"
                      defaultValue="sdgcn"
                      onChange={handleModelChange}
                    >
                      <option value="lstm-gcn">LSTM-GCN (Spatial-Temporal)</option>
                      <option value="sdgcn">SDGCN (Social-Economic)</option>
                      <option value="sarima">SARIMA (Seasonal)</option>
                    </select>
                    <p className="mt-2 text-xs text-gray-400">
                      {predictionModel === 'lstm-gcn' && 
                        "Using socioeconomic factors: income, housing, education, health"
                      }
                      {predictionModel === 'sdgcn' && 
                        "Using socioeconomic factors: income, employment, education, crime density"
                      }
                      {predictionModel === 'sarima' && 
                        "Using time series data only (no socioeconomic factors)"
                      }
                    </p>
                  </div>
                  
                  <div className="mt-2 p-2 bg-indigo-900/30 border border-indigo-700/30 rounded-md mb-4">
                    <div className="text-xs text-indigo-300">
                      {predictionModel === 'lstm-gcn' && (
                        <span className="font-semibold">LSTM-GCN:</span>
                      )}
                      {predictionModel === 'sdgcn' && (
                        <span className="font-semibold">SDGCN:</span>
                      )}
                      {predictionModel === 'sarima' && (
                        <span className="font-semibold">SARIMA:</span>
                      )}
                      {' '}
                      {predictionModel === 'lstm-gcn' && 
                        "Long Short-Term Memory with Graph Convolutional Networks. Combines temporal patterns with spatial relationships."
                      }
                      {predictionModel === 'sdgcn' && 
                        "Socioeconomic Dynamic Graph Convolutional Network. Integrates socioeconomic indicators with spatio-temporal crime data."
                      }
                      {predictionModel === 'sarima' && 
                        "Seasonal AutoRegressive Integrated Moving Average. Captures seasonal patterns in historical crime data."
                      }
                    </div>
                  </div>
                  
                  <Button 
                    variant="default"
                    className={`w-full ${showPredictions 
                      ? 'bg-gradient-to-r from-green-500 to-teal-500 hover:from-green-600 hover:to-teal-600' 
                      : 'bg-gradient-to-r from-red-500 to-orange-500 hover:from-red-600 hover:to-orange-600'} 
                      text-white`}
                    onClick={handleGeneratePrediction}
                  >
                    {showPredictions ? 'Predictions Active' : 'Generate Prediction'}
                  </Button>
                  
                  {/* Prediction Status */}
                  {showPredictions && (
                    <div className="mt-3 p-2 bg-green-500/10 text-green-300 text-xs rounded-md border border-green-500/20 flex items-center">
                      <div className="h-2 w-2 rounded-full bg-green-500 mr-2 animate-pulse"></div>
                      <p>Prediction model active - {Math.floor(30 + Math.random() * 20)} hotspots identified</p>
                    </div>
                  )}
                </div>
                
                <div className="bg-gray-800/70 rounded-xl border border-gray-700/50 p-4 shadow-lg">
                  <h3 className="text-lg font-semibold text-white mb-4">Selected Area Details</h3>
                  {selectedLSOA ? (
                    <div className="space-y-3">
                      <div>
                        <h3 className="text-sm font-medium text-gray-400">LSOA Code</h3>
                        <p className="text-sm text-white">{selectedLSOA}</p>
                      </div>
                      {lsoaData && (
                        <>
                          <div>
                            <h3 className="text-sm font-medium text-gray-400">Area Name</h3>
                            <p className="text-sm text-white">{lsoaData.name || "Unknown"}</p>
                          </div>
                          <div>
                            <h3 className="text-sm font-medium text-gray-400">Risk Level</h3>
                            <div className="flex items-center gap-2">
                              <div 
                                className={`h-3 w-3 rounded-full ${
                                  lsoaData.risk_level === 'High' ? 'bg-red-500' : 
                                  lsoaData.risk_level === 'Medium' ? 'bg-amber-500' : 'bg-green-500'
                                }`}
                              ></div>
                              <p className="text-sm text-white">{lsoaData.risk_level || "Medium"}</p>
                            </div>
                          </div>
                          {forecastData && (
                            <div>
                              <h3 className="text-sm font-medium text-gray-400">SARIMA Forecast</h3>
                              <p className="text-sm text-gray-300">
                                Expected burglaries next month: 
                                <span className="font-bold text-amber-400 ml-1">
                                  {Math.round(forecastData.forecast[0])}
                                </span>
                              </p>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  ) : (
                    <div className="text-center p-4 text-gray-400 bg-gray-900/50 rounded-lg">
                      <p>Select an area on the map to view details</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        );
        
      case 'map':
        return (
          <div className="p-6 relative">
            <div className="mb-6">
              <h1 className="text-2xl font-bold text-white mb-2">
                <span className="mr-2">📍</span>
                London Residential Burglary Risk Map
              </h1>
              <div className="flex items-center space-x-4 mt-4">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full bg-red-500"></div>
                  <span className="text-sm text-gray-300">High Risk</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                  <span className="text-sm text-gray-300">Medium Risk</span>
                </div>
              </div>
            </div>
            
            <div className="flex items-start">
              <div className="flex-1 max-w-[calc(100%-300px)]">
                <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden shadow-2xl shadow-black/50 h-[70vh]">
                  <MapComponent 
                    onLSOASelect={handleLSOASelect}
                    showPoliceAllocation={showPoliceAllocation}
                    selectedLSOA={selectedLSOA}
                    policeAllocationData={policeAllocationData}
                  />
                </div>
              </div>
              
              <div className="ml-6">
                <PoliceAllocation 
                  visible={true} 
                  onToggle={() => setShowPoliceAllocation(!showPoliceAllocation)}
                />
              </div>
            </div>
          </div>
        );
        
      case 'allocation':
        return (
          <div className="p-6">
            <h1 className="text-2xl font-bold text-white mb-6">
              <span className="mr-2">👮</span>
              Police Resource Allocation
            </h1>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 bg-gray-900 border border-gray-800 rounded-xl overflow-hidden shadow-2xl shadow-black/50">
                <MapComponent 
                  onLSOASelect={handleLSOASelect}
                  showPoliceAllocation={true}
                  selectedLSOA={selectedLSOA}
                  policeAllocationData={policeAllocationData}
                />
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden shadow-2xl shadow-black/50 p-6">
                <h2 className="text-xl font-bold text-white mb-4">Allocation Strategy</h2>
                <p className="text-gray-300 mb-4">
                  Configure and deploy police resources based on predictive crime modeling and geographic profiling.
                </p>
                <div className="space-y-4">
                  <div className="p-4 bg-blue-500/10 text-blue-400 rounded-lg border border-blue-500/20">
                    <h3 className="font-semibold mb-2">CPTED Principles</h3>
                    <p className="text-sm">
                      Our algorithm uses CPTED (Crime Prevention Through Environmental Design) principles to identify crime hotspots and allocate resources efficiently.
                    </p>
                  </div>
                  <div className="p-4 bg-green-500/10 text-green-400 rounded-lg border border-green-500/20">
                    <h3 className="font-semibold mb-2">Optimization</h3>
                    <p className="text-sm">
                      83.3% of high-risk areas covered with current allocation of 100 units and 2 hours deployment.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
        
      case 'analytics':
        return <DataAnalytics />;
        
      case 'emmie':
        return <EmmieExplanation selectedLSOA={selectedLSOA} lsoaFactors={lsoaRiskFactors} />;
        
      default:
        return <div>Select a view</div>;
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
          onOpenTutorial={() => setShowTutorial(true)} 
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
      
      {/* Tutorial Video Dialog */}
      <TutorialVideo
        open={showTutorial}
        onClose={() => setShowTutorial(false)}
      />
      
      {/* Terms and Services Dialog - Render this last to ensure it's on top */}
      <div className="relative z-[100000]">
        <TermsDialog
          open={showTermsDialog}
          onClose={() => {}}  // Prevent closing without accepting
          onAccept={handleTermsAccept}
          onWatchTutorial={handleWatchTutorial}
        />
      </div>
    </div>
  );
};

export default Dashboard;
