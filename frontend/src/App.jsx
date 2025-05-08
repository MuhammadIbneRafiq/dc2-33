import { useState } from 'react'
import Header from './components/Header'
import MapComponent from './components/MapComponent'
import BurglaryChart from './components/BurglaryChart'
import PoliceAllocation from './components/PoliceAllocation'
import './App.css'

function App() {
  const [selectedLSOA, setSelectedLSOA] = useState(null);
  const [showPoliceAllocation, setShowPoliceAllocation] = useState(false);
  const [burglaryData, setBurglaryData] = useState(null);

  // Toggle police allocation view
  const handleTogglePoliceAllocation = () => {
    setShowPoliceAllocation(!showPoliceAllocation);
  };

  // Handle when an LSOA is selected on the map
  const handleLSOASelect = (lsoa) => {
    setSelectedLSOA(lsoa);
    // In a real app, we would fetch data for this LSOA
    // For now, we'll simulate this with mock data
    const mockBurglaryData = generateMockBurglaryData(lsoa);
    setBurglaryData(mockBurglaryData);
  };

  // Generate mock burglary data for demonstration
  const generateMockBurglaryData = (lsoa) => {
    const months = [
      "2022-03", "2022-04", "2022-05", "2022-06", "2022-07", "2022-08",
      "2022-09", "2022-10", "2022-11", "2022-12", "2023-01", "2023-02",
      "2023-03", "2023-04", "2023-05", "2023-06", "2023-07", "2023-08"
    ];
    
    // Generate random counts between 10-50 for each month
    const countsBefore = months.map(() => Math.floor(Math.random() * 40) + 10);
    
    // For allocated police officers, reduce counts by 30-50%
    const countsAfter = countsBefore.map(count => {
      const reductionFactor = 0.5 + (Math.random() * 0.2);
      return Math.floor(count * reductionFactor);
    });
    
    return {
      lsoa: lsoa,
      lsoaName: `${lsoa.slice(0, 3)} ${lsoa.slice(3)}`,
      months: months,
      countsBefore: countsBefore,
      countsAfter: countsAfter,
      totalBefore: countsBefore.reduce((sum, count) => sum + count, 0),
      totalAfter: countsAfter.reduce((sum, count) => sum + count, 0)
    };
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        <div className="bg-white rounded-lg shadow-lg overflow-hidden mb-8">
          <div className="p-4 bg-dark text-white">
            <h2 className="text-xl font-bold">London Burglary Data Dashboard</h2>
            <p className="text-sm">Select an area on the map to view burglary trends and police allocation impact</p>
          </div>

          <div className="p-4">
            <div className="flex items-center justify-end mb-4">
              <span className="mr-2 text-sm font-medium">Show Police Allocation</span>
              <label className="toggle-switch">
                <input 
                  type="checkbox" 
                  checked={showPoliceAllocation}
                  onChange={handleTogglePoliceAllocation}
                />
                <span className="toggle-slider"></span>
              </label>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-gray-50 p-4 rounded-lg shadow">
                <h3 className="text-lg font-semibold mb-2">Interactive Map</h3>
                <MapComponent 
                  onLSOASelect={handleLSOASelect} 
                  showPoliceAllocation={showPoliceAllocation}
                  selectedLSOA={selectedLSOA}
                />
                
                {showPoliceAllocation && (
                  <div className="mt-4">
                    <PoliceAllocation />
                  </div>
                )}
              </div>
              
              <div className="bg-gray-50 p-4 rounded-lg shadow">
                <h3 className="text-lg font-semibold mb-2">
                  {selectedLSOA ? `Burglary Trends for ${selectedLSOA}` : 'Select an area to view burglary trends'}
                </h3>
                
                {burglaryData ? (
                  <BurglaryChart 
                    data={burglaryData} 
                    showPoliceImpact={showPoliceAllocation} 
                  />
                ) : (
                  <div className="flex items-center justify-center h-64 bg-gray-100 rounded">
                    <p className="text-gray-500">Click on an area in the map to view data</p>
                  </div>
                )}
                
                {burglaryData && showPoliceAllocation && (
                  <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                    <h4 className="font-medium text-green-800">Police Allocation Impact</h4>
                    <p className="text-sm text-green-700 mt-1">
                      With optimal police allocation, burglaries could be reduced from
                      <span className="font-bold"> {burglaryData.totalBefore} </span>
                      to
                      <span className="font-bold"> {burglaryData.totalAfter} </span>
                      ({Math.round(((burglaryData.totalBefore - burglaryData.totalAfter) / burglaryData.totalBefore) * 100)}% reduction)
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow-lg overflow-hidden mb-8">
          <div className="p-4">
            <h2 className="text-lg font-semibold mb-2">About This Dashboard</h2>
            <p className="text-sm text-gray-600">
              This interactive dashboard visualizes burglary data across London based on Lower Super Output Areas (LSOAs).
              The map highlights burglary hotspots, and when an area is selected, you can view historical trends.
              Toggle the "Show Police Allocation" switch to see how optimal police resource allocation, 
              determined using K-means clustering, could potentially reduce burglary rates in different areas.
            </p>
          </div>
        </div>
      </main>
      
      <footer className="bg-dark text-white p-4 text-center">
        <p>&copy; 2025 London Crime Analysis Dashboard</p>
      </footer>
    </div>
  )
}

export default App 