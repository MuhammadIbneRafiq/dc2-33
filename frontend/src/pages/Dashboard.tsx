import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import MapComponent from '@/components/map/MapComponent';
import { api } from '@/api/api';
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
  
  // Fetch Burglary Time Series Data
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
      return api.burglary.getTimeSeries({ 
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
    // Use fetched timeSeriesData and forecastData
    const historicalPoints = timeSeriesData?.time_series 
      ? timeSeriesData.time_series.map((p: any) => ({
          date: new Date(p.date), // Assuming p.date is a string like 'YYYY-MM-DD' or timestamp
          value: p.burglary_count,
        }))
      : [];

    // The forecastData useQuery is already defined above, let's use its result
    // It's fetched based on selectedLSOA.
    const forecastApiValues = forecastData?.forecast; // Array of numbers
    const forecastApiDates = forecastData?.dates;   // Array of date strings like 'YYYY-MM'

    let forecastPoints: { date: Date; value: number }[] = [];
    if (showPredictions && forecastApiValues && forecastApiDates) {
      forecastPoints = forecastApiValues.map((val: number, index: number) => ({
        date: new Date(forecastApiDates[index]), // Ensure dates are parsed correctly
        value: val,
      }));
    }
    
    // Combine for line chart
    const allPoints = [...historicalPoints, ...forecastPoints];

    if (isLoadingTimeSeries) {
      return (
        <div className="mt-6 bg-gray-800/70 rounded-xl border border-gray-700/50 p-4 shadow-lg text-center text-white">
          Loading Time Series Data...
        </div>
      );
    }
    if (errorTimeSeries) {
      return (
        <div className="mt-6 bg-red-700/70 rounded-xl border border-red-600/50 p-4 shadow-lg text-center text-white">
          Error loading time series: {(errorTimeSeries as Error).message}
        </div>
      );
    }
    if (allPoints.length === 0) {
      return (
        <div className="mt-6 bg-gray-800/70 rounded-xl border border-gray-700/50 p-4 shadow-lg text-center text-white">
          No time series data available for the current selection.
        </div>
      );
    }
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
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 p-6">
            {/* Main content area */}
            <div className="lg:col-span-2">
              {/* Map Component taking up the majority of the space */}
              <div className="h-[600px] rounded-xl shadow-2xl overflow-hidden border border-gray-700/50 relative">
                <MapComponent 
                  onLSOASelect={handleLSOASelect} 
                  showPoliceAllocation={showPoliceAllocation}
                  selectedLSOA={selectedLSOA}
                  showPredictions={showPredictions}
                  predictionModel={predictionModel}
                  dateRange={dateRange} // Pass dateRange to map
                />
                 {/* Time Series Panel - to be rendered below the map */}
                 <div className="absolute bottom-0 left-0 right-0 z-[1000] p-2 pointer-events-none">
                  <div className="pointer-events-auto max-w-3xl mx-auto">
                    {renderTimeSeriesPanel()} 
                  </div>
                </div>
              </div>
            </div>

            {/* Sidebar/Details Area */}
            <div className="lg:col-span-1 space-y-6">
              <PoliceAllocation 
                onToggle={handleTogglePoliceAllocation}
                showPoliceAllocation={showPoliceAllocation}
              />
              <DataAnalytics 
                selectedLsoaCode={selectedLSOA} 
                lsoaWellbeingData={lsoaData} 
                isLoadingLsoaData={isLoadingLsoaData} 
              />
              <EmmieExplanation />
            </div>
          </div>
        );
      case 'chat':
        return <PoliceChat />;
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
