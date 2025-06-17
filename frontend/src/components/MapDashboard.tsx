import React, { useState } from 'react';
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

  // Note: No backend connection - using external APIs (UK Police API, ONS) and frontend-only data

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

  // Handle police allocation - using real crime data to determine optimal placement
  const handlePoliceAllocation = async () => {
    try {
      setPoliceAllocationEnabled(true);
      console.log('Calculating police allocation based on real crime data...');
      
      // Import the real API functions
      const { api } = await import('../api/api');
      
      // Get real burglary data for the last 3 months
      const endDate = new Date();
      const startDate = new Date();
      startDate.setMonth(startDate.getMonth() - 3);
      
      const months = api.utils.generateMonthsArray(
        startDate.toISOString().slice(0, 7),
        endDate.toISOString().slice(0, 7)
      );
      
      const realBurglaryData = await api.police.getLondonBurglaryData(months);
      
      // Calculate police unit placement based on real crime hotspots
      const crimeHotspots: { [key: string]: { lat: number; lng: number; count: number } } = {};
      
      realBurglaryData.forEach(crime => {
        const key = `${Math.round(crime.lat * 1000)}_${Math.round(crime.lng * 1000)}`;
        if (!crimeHotspots[key]) {
          crimeHotspots[key] = { lat: crime.lat, lng: crime.lng, count: 0 };
        }
        crimeHotspots[key].count++;
      });
      
      // Sort by crime count and place police units at top hotspots
      const sortedHotspots = Object.values(crimeHotspots)
        .sort((a, b) => b.count - a.count)
        .slice(0, 100); // Top 100 hotspots
      
      const policeUnitsFromRealData = sortedHotspots.map((hotspot, i) => ({
        id: i,
        lat: hotspot.lat + (Math.random() - 0.5) * 0.001, // Small offset for visibility
        lng: hotspot.lng + (Math.random() - 0.5) * 0.001,
        type: hotspot.count > 5 ? 'vehicle' : 'officer',
        cluster: Math.floor(i / 5),
        crimeCount: hotspot.count
      }));
      
      setPoliceUnits(policeUnitsFromRealData);
      console.log(`Police allocation applied based on ${realBurglaryData.length} real crimes:`, policeUnitsFromRealData.length, 'units');
    } catch (error) {
      console.error('Error applying police allocation:', error);
    }
  };

  // Handle LSOA click - fetch real data from external APIs
  const handleLSOAClick = async (lsoaCode: string) => {
    try {
      console.log('LSOA clicked:', lsoaCode);
      onLSOASelect && onLSOASelect(lsoaCode);
      
      // Get the center coordinates for this LSOA (simplified - in real app you'd have LSOA boundary data)
      const { api } = await import('../api/api');
      
      // Find the nearest borough for API calls
      const londonCenter = { lat: 51.5074, lng: -0.1278 };
      const nearestBorough = api.utils.LONDON_BOROUGHS[0]; // Simplified - use Westminster as default
      
      // Get real burglary data for the area
      const currentMonth = new Date().toISOString().slice(0, 7);
      const realBurglaryData = await api.police.getBurglaryData(
        nearestBorough.coords[0],
        nearestBorough.coords[1],
        currentMonth
      );
      
      // Calculate risk level based on real crime density
      const burglaryCount = Array.isArray(realBurglaryData) ? realBurglaryData.length : 0;
      const riskLevel = burglaryCount > 10 ? 'High' : burglaryCount > 5 ? 'Medium' : 'Low';
      
      console.log('Real LSOA data:', {
        code: lsoaCode,
        risk_level: riskLevel,
        burglary_count: burglaryCount,
        data_source: 'UK Police API'
      });
      
    } catch (error) {
      console.error('Error fetching real LSOA data:', error);
    }
  };

  // Get risk level from real crime count
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
            
            {/* Backend Status Indicator */}
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${isBackendConnected ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-xs text-gray-400">
                {isBackendConnected ? 'Real Data Connected' : 'Using Mock Data'}
              </span>
            </div>
          </div>

          {/* Map Level Toggle */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 bg-gray-700 rounded-lg p-1">
              <button
                onClick={() => setActiveView('lsoa')}
                className={`px-3 py-1 text-sm rounded transition-all ${
                  activeView === 'lsoa' 
                    ? 'bg-blue-600 text-white shadow-lg' 
                    : 'text-gray-300 hover:text-white hover:bg-gray-600'
                }`}
              >
                <div className="flex items-center space-x-1">
                  <Map size={14} />
                  <span>LSOA Level</span>
                </div>
              </button>
              <button
                onClick={() => setActiveView('borough')}
                className={`px-3 py-1 text-sm rounded transition-all ${
                  activeView === 'borough' 
                    ? 'bg-blue-600 text-white shadow-lg' 
                    : 'text-gray-300 hover:text-white hover:bg-gray-600'
                }`}
              >
                <div className="flex items-center space-x-1">
                  <BarChart3 size={14} />
                  <span>Borough Level</span>
                </div>
              </button>
            </div>
          </div>
        </div>

        {/* Secondary Controls */}
        <div className="mt-3 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-400">
              Viewing: <span className="text-white font-medium">
                {activeView === 'lsoa' ? 'Lower Super Output Areas' : 'London Boroughs'}
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            {/* Police Allocation Toggle */}
            <button
              onClick={() => setShowPoliceAllocation(!showPoliceAllocation)}
              className={`flex items-center space-x-2 px-3 py-1 rounded text-sm transition-all ${
                showPoliceAllocation 
                  ? 'bg-green-600 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <Users size={14} />
              <span>Police Units</span>
              {showPoliceAllocation ? <ToggleRight size={16} /> : <ToggleLeft size={16} />}
            </button>

            {/* Predictions Toggle */}
            <button
              onClick={() => setShowPredictions(!showPredictions)}
              className={`flex items-center space-x-2 px-3 py-1 rounded text-sm transition-all ${
                showPredictions 
                  ? 'bg-purple-600 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <TrendingUp size={14} />
              <span>Predictions</span>
              {showPredictions ? <ToggleRight size={16} /> : <ToggleLeft size={16} />}
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
        setActiveView={setActiveView} 
        showPoliceAllocation={showPoliceAllocation}
        onTogglePoliceAllocation={handleTogglePoliceAllocation}
        selectedLSOA={selectedLSOA}
      />

      {/* Main Content */}
      <div className="flex-1 ml-[280px]">
        <Header />
        
        <div className="container mx-auto p-6">
          {activeView === 'dashboard' && (
            <>
              <DashboardStats />
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
            </>
          )}
          
          {activeView === 'map' && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
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
          )}
          
          {activeView === 'allocation' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="lg:col-span-1">
                <PoliceAllocation
                  showPoliceAllocation={showPoliceAllocation}
                  onTogglePoliceAllocation={handleTogglePoliceAllocation}
                  onPoliceDataLoaded={handlePoliceDataLoaded}
                  onMetricsUpdate={handleMetricsUpdate}
                />
              </div>
              <div className="lg:col-span-1">
                <CrimeMap 
                  showPoliceAllocation={showPoliceAllocation}
                  policeData={policeData}
                  onSelectLSOA={handleSelectLSOA}
                />
              </div>
            </div>
          )}
          
          {activeView === 'analytics' && <DataAnalytics />}
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
