import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { BarChart3, Map, Users, TrendingUp, ToggleLeft, ToggleRight } from 'lucide-react';
import MapComponent from './map/MapComponent';
import Header from './Header';
import Sidebar from './Sidebar';
import CrimeMap from './CrimeMap';
import PoliceAllocation from './PoliceAllocation';
import DashboardStats from './DashboardStats';
import DataAnalytics from './DataAnalytics';
import PoliceChat from './PoliceChat';

interface MapDashboardProps {
  onLSOASelect?: (lsoa: string) => void;
  selectedLSOA?: string | null;
}

const MapDashboard: React.FC<MapDashboardProps> = ({ onLSOASelect, selectedLSOA }) => {
  const [activeView, setActiveView] = useState<'lsoa' | 'borough'>('lsoa');
  const [showPoliceAllocation, setShowPoliceAllocation] = useState<boolean>(false);
  const [showPredictions, setShowPredictions] = useState<boolean>(false);
  const [isBackendConnected] = useState<boolean>(false); // Always false - no backend
  const [policeData, setPoliceData] = useState<any[] | null>(null);
  const [allocationMetrics, setAllocationMetrics] = useState<any | null>(null);
  const [policeAllocationEnabled, setPoliceAllocationEnabled] = useState(false);
  const [policeUnits, setPoliceUnits] = useState<any[]>([]);
  const [mapLevel, setMapLevel] = useState<'lsoa' | 'borough'>('lsoa');
  const [burglaryData, setBurglaryData] = useState<any[]>([]);
  const [isLoadingBurglaryData, setIsLoadingBurglaryData] = useState(false);
  const [boundariesLoaded, setBoundariesLoaded] = useState(false);
  const [isGeneratingForecast, setIsGeneratingForecast] = useState(false);

  // Step 1: Load boundaries immediately on component mount
  useEffect(() => {
    loadBoundariesOnly();
  }, []);

  // Step 2: Load boundaries first (fast operation)
  const loadBoundariesOnly = async () => {
    try {
      console.log('🗺️ Starting LSOA and borough boundaries loading...');
      // Don't set to true immediately - wait for actual loading to complete
      setBoundariesLoaded(false);
      console.log('⏳ Waiting for boundaries to load from ONS API...');
    } catch (error) {
      console.error('❌ Error initiating boundary loading:', error);
      setBoundariesLoaded(false);
    }
  };

  // Callback when MapComponent finishes loading boundaries 
  const handleBoundariesLoaded = () => {
    console.log('✅ Boundaries loaded successfully!');
    setBoundariesLoaded(true);
  };

  // Step 3: Generate INSTANT random dots
  const handleGenerateForecast = () => {
    setIsGeneratingForecast(true);
    setShowPredictions(true);
    
    console.log('🔮 Generating instant random burglary dots...');
    
    // Generate random dots instantly - no async, no delays
    const randomDots = generateDummyForecastData();
    setBurglaryData([...randomDots]);
    
    console.log(`✅ Generated ${randomDots.length} random burglary dots on map instantly!`);
    
    setIsGeneratingForecast(false);
  };

  // Generate INSTANT random dots - no delays, no complex processing
  const generateDummyForecastData = (): any[] => {
    const dummyData = [];
    const londonCenter = { lat: 51.5074, lng: -0.1278 };
    
    // Generate 4000 burglary points all over London with red alert emojis 🚨
    for (let i = 0; i < 4000; i++) {
      const lat = londonCenter.lat + (Math.random() - 0.5) * 0.3; // ~15km spread
      const lng = londonCenter.lng + (Math.random() - 0.5) * 0.4; // ~20km spread
      
      dummyData.push({
        id: `burglary-alert-${i}`,
        lat,
        lng,
        borough: ['Westminster', 'Camden', 'Hackney', 'Tower Hamlets', 'Southwark', 'Lambeth', 'Islington', 'Newham', 'Greenwich', 'Lewisham'][Math.floor(Math.random() * 10)],
        category: 'burglary',
        risk_level: ['High', 'High', 'High', 'Medium', 'Low'][Math.floor(Math.random() * 5)], // More high risk
        alert_emoji: '🚨',
        alert_level: 'RED ALERT',
        date: new Date().toISOString().slice(0, 10),
        location_type: 'High Risk Area',
        outcome_status: 'Active Alert'
      });
    }
    
    return dummyData;
  };

  // Generate minimal burglary data as fallback
  const generateMinimalBurglaryData = (): any[] => {
    const minimalData = [];
    const londonCenter = { lat: 51.5074, lng: -0.1278 };
    
    // Just 10 points for minimal load
    for (let i = 0; i < 10; i++) {
      const lat = londonCenter.lat + (Math.random() - 0.5) * 0.1;
      const lng = londonCenter.lng + (Math.random() - 0.5) * 0.15;
      
      minimalData.push({
        id: `minimal-${i}`,
        lat,
        lng,
        borough: 'Westminster',
        category: 'burglary',
        risk_level: 'Medium'
      });
    }
    
    return minimalData;
  };

  // Step 4: INSTANT police allocation with random placement
  const handlePoliceAllocation = () => {
    setPoliceAllocationEnabled(true);
    console.log('👮 Placing 4000 police units instantly across London...');
    
    // Generate 4000 instant police units all over London with red alert emojis 🚨
    const randomPoliceUnits = [];
    const londonCenter = { lat: 51.5074, lng: -0.1278 };
    const numUnits = 4000; // Fixed number
    
    for (let i = 0; i < numUnits; i++) {
      const lat = londonCenter.lat + (Math.random() - 0.5) * 0.3;
      const lng = londonCenter.lng + (Math.random() - 0.5) * 0.4;
      
      randomPoliceUnits.push({
        id: `police-unit-${i}`,
        lat,
        lng,
        type: Math.random() > 0.5 ? 'vehicle' : 'officer',
        assignedArea: ['Westminster', 'Camden', 'Hackney', 'Tower Hamlets', 'Southwark', 'Lambeth', 'Islington', 'Newham', 'Greenwich', 'Lewisham'][Math.floor(Math.random() * 10)],
        status: 'active_patrol',
        alert_emoji: '🚨',
        alert_level: 'RED ALERT',
        unit_type: Math.random() > 0.7 ? 'Armed Response' : Math.random() > 0.5 ? 'Patrol Unit' : 'Foot Patrol',
        response_time: Math.round(Math.random() * 10) + 2 + ' mins'
      });
    }
    
    setPoliceUnits([...randomPoliceUnits]);
    setShowPoliceAllocation(true);
    console.log(`🚨 Deployed ${randomPoliceUnits.length} police units across London with RED ALERT status!`);
  };

  // Generate some high-risk areas if no burglary data exists
  const generateHighRiskAreas = () => {
    const knownHighRiskAreas = [
      { lat: 51.5074, lng: -0.1278, borough: 'Westminster' }, // Central London
      { lat: 51.5290, lng: -0.1255, borough: 'Camden' }, // Camden
      { lat: 51.5203, lng: -0.0293, borough: 'Tower Hamlets' }, // Tower Hamlets
      { lat: 51.5450, lng: -0.0553, borough: 'Hackney' }, // Hackney
      { lat: 51.5032, lng: -0.0851, borough: 'Southwark' }, // Southwark
    ];
    
    return knownHighRiskAreas;
  };

  const handleTogglePoliceAllocation = () => {
    setShowPoliceAllocation(!showPoliceAllocation);
  };

  const handlePoliceDataLoaded = (data: any[]) => {
    setPoliceData(data);
  };

  const handleSelectLSOA = (lsoa: string) => {
    onLSOASelect && onLSOASelect(lsoa);
  };

  const handleMetricsUpdate = (metrics: any) => {
    setAllocationMetrics(metrics);
  };

  const handleBoroughSelect = (borough: string) => {
    console.log('Borough selected:', borough);
    // You can add borough-specific logic here
  };

  // Handle LSOA click - no external API calls, just local processing
  const handleLSOAClick = async (lsoaCode: string) => {
    try {
      console.log('LSOA clicked:', lsoaCode);
      onLSOASelect && onLSOASelect(lsoaCode);
      
      // Simple local risk calculation based on existing data
      const localCrimes = burglaryData.filter(crime => 
        Math.abs(crime.lat - 51.5074) < 0.01 && Math.abs(crime.lng + 0.1278) < 0.01
      );
      
      const riskLevel = localCrimes.length > 5 ? 'High' : localCrimes.length > 2 ? 'Medium' : 'Low';
      
      console.log('Local LSOA analysis:', {
        code: lsoaCode,
        risk_level: riskLevel,
        local_crime_count: localCrimes.length,
        data_source: 'Local calculation'
      });
      
    } catch (error) {
      console.error('Error processing LSOA click:', error);
    }
  };

  // Get risk level from crime count
  const getRiskLevel = (crimeCount: number) => {
    if (crimeCount > 10) return 'High';
    if (crimeCount > 5) return 'Medium';
    return 'Low';
  };

  return (
    <div className="h-full flex flex-col bg-gray-900">
      {/* Enhanced Header with Level Controls */}
      <div className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h2 className="text-xl font-bold text-white">London Crime Map</h2>
            
            {/* Boundaries Status Indicator */}
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${boundariesLoaded ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm text-gray-300">
                {boundariesLoaded ? 'Boundaries Loaded' : 'Loading Boundaries...'}
              </span>
            </div>
          </div>

          {/* View Level Toggle */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center bg-gray-700 rounded-lg p-1">
              <button
                onClick={() => setMapLevel('lsoa')}
                className={`px-3 py-1 text-sm rounded transition-colors ${
                  mapLevel === 'lsoa' 
                    ? 'bg-blue-600 text-white' 
                    : 'text-gray-300 hover:text-white'
                }`}
              >
                LSOA View
              </button>
              <button
                onClick={() => setMapLevel('borough')}
                className={`px-3 py-1 text-sm rounded transition-colors ${
                  mapLevel === 'borough' 
                    ? 'bg-blue-600 text-white' 
                    : 'text-gray-300 hover:text-white'
                }`}
              >
                Borough View
              </button>
        </div>

            {/* Quick Action Buttons */}
            <button
              onClick={handleGenerateForecast}
              disabled={isGeneratingForecast}
              className="bg-green-600 hover:bg-green-700 disabled:bg-green-800 disabled:opacity-50 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
            >
              <TrendingUp className="w-4 h-4" />
              <span>{isGeneratingForecast ? 'Generating...' : 'Generate Forecast'}</span>
            </button>

            <button
              onClick={handlePoliceAllocation}
              disabled={policeAllocationEnabled}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:opacity-50 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
            >
              <Users className="w-4 h-4" />
              <span>{policeAllocationEnabled ? 'Allocated' : 'Allocate Police'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Enhanced Map View */}
      <div className="flex-1 relative">
        <motion.div 
          className="h-full"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <MapComponent
            onLSOASelect={handleSelectLSOA}
            onBoroughSelect={handleBoroughSelect}
            selectedLSOA={selectedLSOA}
            showPoliceAllocation={showPoliceAllocation}
            showPredictions={showPredictions}
            mapLevel={activeView}
            burglaryData={burglaryData}
            policeUnits={policeUnits}
            isLoadingBurglaryData={isLoadingBurglaryData}
            onBoundariesLoaded={handleBoundariesLoaded}
          />
        </motion.div>

        {/* Level Information Panel */}
        <div className="absolute top-4 left-4 bg-gray-800 bg-opacity-90 rounded-lg p-4 text-white max-w-xs">
          <h3 className="font-bold text-lg mb-2">
            {activeView === 'lsoa' ? 'LSOA View' : 'Borough View'}
          </h3>
          <p className="text-sm text-gray-300 mb-2">
            {activeView === 'lsoa' 
              ? 'Detailed area-level analysis showing individual Lower Super Output Areas with precise burglary risk assessment.'
              : 'Borough-level overview showing aggregated burglary statistics across London\'s administrative boroughs.'
            }
          </p>
          <div className="text-xs text-gray-400">
            {activeView === 'lsoa' 
              ? 'Click on areas to view detailed LSOA statistics'
              : 'Click on boroughs to view aggregated borough data'
            }
          </div>
        </div>

        {/* Real Data Indicator */}
        {isBackendConnected && (
          <div className="absolute bottom-4 right-4 bg-green-600 bg-opacity-90 rounded-lg p-3 text-white">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-300 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium">Live London Data</span>
            </div>
            <p className="text-xs mt-1 opacity-90">
              Real LSOA boundaries & burglary statistics
            </p>
          </div>
        )}

        {/* Map Controls */}
        <div className="absolute top-4 right-4 z-10 flex flex-col space-y-2">
          <div className="flex space-x-2">
            <button
              onClick={() => setMapLevel(mapLevel === 'lsoa' ? 'borough' : 'lsoa')}
              className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                mapLevel === 'lsoa' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {mapLevel === 'lsoa' ? 'LSOA View' : 'Borough View'}
            </button>
            
            <button
              onClick={handlePoliceAllocation}
              className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                policeAllocationEnabled 
                  ? 'bg-green-600 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {policeAllocationEnabled ? 'Police Units Active' : 'Deploy Police'}
            </button>
          </div>
          
          {/* Legend */}
          <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-3 text-xs">
            <div className="text-white font-semibold mb-2">Risk Levels</div>
            <div className="space-y-1">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-red-500 rounded"></div>
                <span className="text-gray-300">High Risk</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-yellow-500 rounded"></div>
                <span className="text-gray-300">Medium Risk</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-500 rounded"></div>
                <span className="text-gray-300">Low Risk</span>
              </div>
              {policeAllocationEnabled && (
                <div className="flex items-center space-x-2 border-t border-gray-600 pt-1 mt-2">
                  <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                  <span className="text-gray-300">Police Units</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Sidebar */}
      <Sidebar 
        activeView={activeView} 
        setActiveView={(view: string) => setActiveView(view as 'lsoa' | 'borough')} 
        showPoliceAllocation={showPoliceAllocation}
        onTogglePoliceAllocation={handleTogglePoliceAllocation}
        selectedLSOA={selectedLSOA}
      />

      {/* Main Content - Remove unused view conditions */}
      <div className="flex-1 ml-[280px]">
        <Header />
        
        <div className="container mx-auto p-6">
          {/* Only show map content since activeView is 'lsoa' | 'borough' */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="lg:col-span-1">
              <CrimeMap 
                showPoliceAllocation={showPoliceAllocation}
                policeData={policeData}
                onSelectLSOA={handleSelectLSOA}
              />
            </div>
            <div className="lg:col-span-1">
              <PoliceAllocation
                showPoliceAllocation={showPoliceAllocation}
                onTogglePoliceAllocation={handleTogglePoliceAllocation}
                onPoliceDataLoaded={handlePoliceDataLoaded}
                onMetricsUpdate={handleMetricsUpdate}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Police Chat Widget */}
      <PoliceChat 
        selectedLSOA={selectedLSOA}
        selectedAllocation={allocationMetrics}
      />
    </div>
  );
};

export default MapDashboard;
