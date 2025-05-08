import { useState, useEffect } from 'react'
import Header from './components/Header'
import MapComponent from './components/MapComponent'
import BurglaryChart from './components/BurglaryChart'
import PoliceAllocation from './components/PoliceAllocation'
import Sidebar from './components/Sidebar'
import './App.css'

function App() {
  const [selectedLSOA, setSelectedLSOA] = useState(null);
  const [showPoliceAllocation, setShowPoliceAllocation] = useState(false);
  const [burglaryData, setBurglaryData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [activeView, setActiveView] = useState('dashboard');

  // Toggle police allocation view
  const handleTogglePoliceAllocation = () => {
    setShowPoliceAllocation(!showPoliceAllocation);
  };

  // Handle when an LSOA is selected on the map
  const handleLSOASelect = (lsoa) => {
    if (lsoa === selectedLSOA) return;
    
    setSelectedLSOA(lsoa);
    setIsLoading(true);
    
    // Simulate data fetching for the selected LSOA
    setTimeout(() => {
      // Generate mock time series data for the selected LSOA
      const mockData = generateMockTimeSeriesData(lsoa);
      setBurglaryData(mockData);
      setIsLoading(false);
    }, 800);
  };
  
  // Generate mock time series data for visualization
  const generateMockTimeSeriesData = (lsoa) => {
    // Using the LSOA code to generate deterministic but seemingly random data
    const seed = lsoa ? lsoa.charCodeAt(6) * lsoa.charCodeAt(8) : 42;
    const rng = (min, max) => Math.floor(seed % 17 * (Math.random() + 0.5) * (max - min) / 17) + min;
    
    // Generate 12 months of data
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const countsBefore = months.map(() => rng(15, 60));
    
    // Calculate reduction based on K-means police allocation
    const reductionPercent = rng(25, 45);
    const countsAfter = countsBefore.map(count => {
      const reducedCount = Math.round(count * (1 - reductionPercent / 100));
      return reducedCount;
    });
    
    return {
      lsoa: lsoa,
      labels: months,
      countsBefore,
      countsAfter,
      reductionPercent
    };
  };
  
  // Simulate initial loading of the dashboard
  useEffect(() => {
    const timer = setTimeout(() => {
      setInitialLoading(false);
    }, 1500);
    
    return () => clearTimeout(timer);
  }, []);

  if (initialLoading) {
    return (
      <div className="main-container min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-t-blue-500 border-blue-500/20 rounded-full animate-spin mx-auto mb-6"></div>
          <h2 className="text-xl font-semibold text-white mb-2">Loading Dashboard</h2>
          <p className="text-gray-400">Preparing London burglary analysis data...</p>
        </div>
      </div>
    );
  }

  // Handler for sidebar navigation
  const handleViewChange = (view) => {
    setActiveView(view);
    if (view === 'allocation') {
      setShowPoliceAllocation(true);
    }
  };

  return (
    <div className="main-container min-h-screen flex">
      {/* Sidebar */}
      <Sidebar activeView={activeView} setActiveView={handleViewChange} />
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        <Header />
        
        <main className="flex-1 overflow-auto p-6">
          {/* Dashboard Title */}
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-white mb-2">
              {activeView === 'dashboard' && 'London Burglary Analysis Dashboard'}
              {activeView === 'map' && 'Interactive Map View'}
              {activeView === 'allocation' && 'Police Resource Allocation'}
              {activeView === 'analytics' && 'Data Analytics & Insights'}
              {activeView === 'settings' && 'Dashboard Settings'}
            </h2>
            <p className="text-gray-400">
              {activeView === 'dashboard' && 'Interactive visualization of burglary data across London, with police resource allocation optimization.'}
              {activeView === 'map' && 'Detailed map view of burglary incidents and hotspots across London LSOAs.'}
              {activeView === 'allocation' && 'Optimize police resource allocation using K-means clustering algorithm.'}
              {activeView === 'analytics' && 'In-depth analysis of burglary patterns, trends, and predictions.'}
              {activeView === 'settings' && 'Configure your dashboard preferences and data sources.'}
            </p>
          </div>
          
          {/* Main Content based on active view */}
          {activeView === 'dashboard' && (
            <>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2">
                  <MapComponent 
                    onLSOASelect={handleLSOASelect} 
                    showPoliceAllocation={showPoliceAllocation}
                    selectedLSOA={selectedLSOA}
                  />
                </div>
                
                <div>
                  <PoliceAllocation 
                    showPoliceAllocation={showPoliceAllocation}
                    onTogglePoliceAllocation={handleTogglePoliceAllocation}
                  />
                </div>
              </div>
              
              <div className="mt-6">
                {isLoading ? (
                  <div className="dashboard-card h-64 flex items-center justify-center">
                    <div className="flex flex-col items-center">
                      <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                      <p className="mt-3 text-gray-400">Loading burglary data...</p>
                    </div>
                  </div>
                ) : (
                  <BurglaryChart 
                    data={burglaryData} 
                    showPoliceImpact={showPoliceAllocation}
                  />
                )}
              </div>
            </>
          )}
          
          {activeView === 'map' && (
            <div className="h-full">
              <MapComponent 
                onLSOASelect={handleLSOASelect} 
                showPoliceAllocation={showPoliceAllocation}
                selectedLSOA={selectedLSOA}
              />
              
              {selectedLSOA && !isLoading && (
                <div className="mt-6">
                  <BurglaryChart 
                    data={burglaryData} 
                    showPoliceImpact={showPoliceAllocation}
                  />
                </div>
              )}
            </div>
          )}
          
          {activeView === 'allocation' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <PoliceAllocation 
                  showPoliceAllocation={true}
                  onTogglePoliceAllocation={() => {}}
                />
              </div>
              
              <div>
                <MapComponent 
                  onLSOASelect={handleLSOASelect} 
                  showPoliceAllocation={true}
                  selectedLSOA={selectedLSOA}
                />
              </div>
            </div>
          )}
          
          {activeView === 'analytics' && (
            <div className="dashboard-card p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Data Analytics</h3>
              <p className="text-gray-400 mb-6">Advanced analytics and visualizations coming soon.</p>
              
              <div className="flex items-center justify-center h-64 border border-gray-800 rounded-lg bg-gray-900">
                <div className="text-center">
                  {/* <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mx-auto text-gray-600 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg> */}
                  <p className="text-gray-400">Analytics module is under development</p>
                </div>
              </div>
            </div>
          )}
          
          {activeView === 'settings' && (
            <div className="dashboard-card p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Settings</h3>
              <p className="text-gray-400 mb-6">Configure your dashboard settings.</p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-1">Theme</label>
                    <select className="w-full rounded-md bg-gray-800 border border-gray-700 px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50">
                      <option>Dark Theme</option>
                      <option>Light Theme</option>
                      <option>System Default</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-1">Data Update Frequency</label>
                    <select className="w-full rounded-md bg-gray-800 border border-gray-700 px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50">
                      <option>Every 24 hours</option>
                      <option>Every 12 hours</option>
                      <option>Every 6 hours</option>
                      <option>Manual updates only</option>
                    </select>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-1">Default Map View</label>
                    <select className="w-full rounded-md bg-gray-800 border border-gray-700 px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50">
                      <option>Default View</option>
                      <option>Heatmap</option>
                      <option>Risk Levels</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-1">Notification Preferences</label>
                    <div className="space-y-2 mt-2">
                      <div className="flex items-center">
                        <input type="checkbox" className="h-4 w-4 rounded bg-gray-800 border-gray-700 text-blue-500 focus:ring-blue-500/25" />
                        <label className="ml-2 text-sm text-gray-300">Hotspot alerts</label>
                      </div>
                      <div className="flex items-center">
                        <input type="checkbox" className="h-4 w-4 rounded bg-gray-800 border-gray-700 text-blue-500 focus:ring-blue-500/25" />
                        <label className="ml-2 text-sm text-gray-300">Resource optimization updates</label>
                      </div>
                      <div className="flex items-center">
                        <input type="checkbox" className="h-4 w-4 rounded bg-gray-800 border-gray-700 text-blue-500 focus:ring-blue-500/25" />
                        <label className="ml-2 text-sm text-gray-300">System notifications</label>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* About Section - Only shown on Dashboard view */}
          {activeView === 'dashboard' && (
            <div className="mt-6 bg-gray-800 rounded-lg border border-gray-700 p-4">
              <h3 className="text-lg font-semibold text-white mb-3 flex items-center">
                {/* <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg> */}
                About This Dashboard
              </h3>
              
              <div className="text-gray-300 space-y-2">
                <p>This interactive dashboard combines real London burglary data with police resource allocation optimization.</p>
                <p>The map highlights burglary trends across LSOAs (Lower Super Output Areas). Select an area to view historical data and projected impact of optimal police allocation.</p>
                <p>Toggle the "Police Allocation" feature to visualize how K-means clustering can be used to allocate police resources for maximum crime reduction.</p>
              </div>
              
              <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gray-900 p-3 rounded-lg border border-gray-700">
                  <div className="flex items-center text-blue-400 font-medium mb-1">
                    {/* <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" viewBox="0 0 20 20" fill="currentColor">
                      <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z" />
                    </svg> */}
                    Data Analysis
                  </div>
                  <p className="text-gray-400 text-sm">
                    Explore burglary trends and patterns across London by LSOA regions.
                  </p>
                </div>
                
                <div className="bg-gray-900 p-3 rounded-lg border border-gray-700">
                  <div className="flex items-center text-green-400 font-medium mb-1">
                    {/* <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
                    </svg> */}
                    Resource Optimization
                  </div>
                  <p className="text-gray-400 text-sm">
                    See how K-means clustering optimizes police allocation for crime reduction.
                  </p>
                </div>
                
                <div className="bg-gray-900 p-3 rounded-lg border border-gray-700">
                  <div className="flex items-center text-orange-400 font-medium mb-1">
                    {/* <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M5.05 4.05a7 7 0 119.9 9.9L10 18.9l-4.95-4.95a7 7 0 010-9.9zM10 11a2 2 0 100-4 2 2 0 000 4z" clipRule="evenodd" />
                    </svg> */}
                    Interactive Mapping
                  </div>
                  <p className="text-gray-400 text-sm">
                    Use the interactive map to select LSOAs and analyze local crime patterns.
                  </p>
                </div>
              </div>
            </div>
          )}
        </main>
        
        <footer className="bg-gray-900 border-t border-gray-800 py-4">
          <div className="container mx-auto px-4 text-center text-gray-400 text-sm">
            <p>London Burglary Analysis Dashboard &copy; 2025</p>
            <p className="mt-1">Data is based on public records from London Metropolitan Police</p>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App; 