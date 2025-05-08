import React, { useState, useEffect } from 'react';

// Mock data for police allocation
const allocationData = {
  totalOfficers: 45,
  deploymentMethod: 'K-means Clustering',
  coveragePercentage: 78,
  estimatedReduction: 42,
  optimizedAreas: [
    { lsoa: 'E01000001', officerCount: 3, reductionPercentage: 38 },
    { lsoa: 'E01032740', officerCount: 5, reductionPercentage: 45 },
    { lsoa: 'E01000005', officerCount: 4, reductionPercentage: 39 },
    { lsoa: 'E01032739', officerCount: 2, reductionPercentage: 32 },
  ],
};

const PoliceAllocation = ({ showPoliceAllocation, onTogglePoliceAllocation }) => {
  const [selectedTab, setSelectedTab] = useState('optimization');
  const [isLoading, setIsLoading] = useState(true);
  
  // Simulate loading data
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 800);
    
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="dashboard-card">
      <div className="flex items-center justify-between card-header px-4 py-3">
        <h3 className="text-white text-lg font-semibold flex items-center">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
            <path d="M13 6a3 3 0 11-6 0 3 3 0 016 0zM18 8a2 2 0 11-4 0 2 2 0 014 0zM14 15a4 4 0 00-8 0v3h8v-3zM6 8a2 2 0 11-4 0 2 2 0 014 0zM16 18v-3a5.972 5.972 0 00-.75-2.906A3.005 3.005 0 0119 15v3h-3zM4.75 12.094A5.973 5.973 0 004 15v3H1v-3a3 3 0 013.75-2.906z" />
          </svg>
          Resource Allocator
        </h3>
        
        <div className="flex items-center space-x-2">
          <div className="flex items-center">
            <label className="toggle-switch mr-2">
              <input 
                type="checkbox" 
                checked={showPoliceAllocation}
                onChange={onTogglePoliceAllocation}
              />
              <span className="toggle-slider"></span>
            </label>
            <span className="text-sm text-gray-300">
              {showPoliceAllocation ? 'Enabled' : 'Disabled'}
            </span>
          </div>
        </div>
      </div>
      
      {isLoading ? (
        <div className="p-10 flex justify-center items-center">
          <div className="w-8 h-8 border-2 border-t-blue-500 border-blue-500/20 rounded-full animate-spin"></div>
        </div>
      ) : (
        <div className="p-4 bg-gray-900">
          <div className="mb-4">
            <div className="flex border-b border-gray-700">
              <button
                className={`py-2 px-4 text-sm font-medium ${selectedTab === 'optimization' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-300'}`}
                onClick={() => setSelectedTab('optimization')}
              >
                Optimization
              </button>
              <button
                className={`py-2 px-4 text-sm font-medium ${selectedTab === 'deployment' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-300'}`}
                onClick={() => setSelectedTab('deployment')}
              >
                Deployment
              </button>
              <button
                className={`py-2 px-4 text-sm font-medium ${selectedTab === 'analytics' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-300'}`}
                onClick={() => setSelectedTab('analytics')}
              >
                Analytics
              </button>
            </div>
          </div>
          
          {selectedTab === 'optimization' && (
            <>
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm text-gray-300">Officer Deployment</label>
                  <span className="text-blue-400 font-medium text-sm">{allocationData.totalOfficers} officers</span>
                </div>
                <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-blue-500 to-green-500" 
                    style={{ width: `${allocationData.coveragePercentage}%` }}
                  ></div>
                </div>
                <div className="flex justify-between mt-1 text-xs text-gray-400">
                  <span>0</span>
                  <span>25</span>
                  <span>50</span>
                  <span>75</span>
                  <span>100</span>
                </div>
              </div>
              
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm text-gray-300">Deployment Hours</label>
                  <div className="flex items-center">
                    <span className="text-blue-400 font-medium text-sm">8 hours</span>
                    <div className="flex items-center ml-2">
                      <button className="p-1 bg-gray-800 rounded-l border border-gray-700 text-gray-400 hover:text-white hover:bg-gray-700">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M5 10a1 1 0 011-1h8a1 1 0 110 2H6a1 1 0 01-1-1z" clipRule="evenodd" />
                        </svg>
                      </button>
                      <button className="p-1 bg-gray-800 rounded-r border-t border-r border-b border-gray-700 text-gray-400 hover:text-white hover:bg-gray-700">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110-2h3V6a1 1 0 011-1z" clipRule="evenodd" />
                        </svg>
                      </button>
                    </div>
                  </div>
                </div>
                <input
                  type="range"
                  min="4"
                  max="12"
                  step="1"
                  defaultValue="8"
                  className="range-slider w-full"
                />
                <div className="flex justify-between mt-1 text-xs text-gray-400">
                  <span>4h</span>
                  <span>6h</span>
                  <span>8h</span>
                  <span>10h</span>
                  <span>12h</span>
                </div>
              </div>
              
              <div className="flex flex-wrap -mx-2 mb-4">
                <div className="w-1/2 px-2 mb-4">
                  <div className="p-3 rounded-lg bg-gray-800 border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">Method</div>
                    <div className="flex items-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                        <path d="M2 10a8 8 0 018-8v8h8a8 8 0 11-16 0z" />
                        <path d="M12 2.252A8.014 8.014 0 0117.748 8H12V2.252z" />
                      </svg>
                      <span className="text-sm font-medium text-white">{allocationData.deploymentMethod}</span>
                    </div>
                  </div>
                </div>
                <div className="w-1/2 px-2 mb-4">
                  <div className="p-3 rounded-lg bg-gray-800 border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">Coverage</div>
                    <div className="flex items-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      <span className="text-sm font-medium text-white">{allocationData.coveragePercentage}% of hot spots</span>
                    </div>
                  </div>
                </div>
                <div className="w-1/2 px-2 mb-4">
                  <div className="p-3 rounded-lg bg-gray-800 border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">Predicted Reduction</div>
                    <div className="flex items-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M12 13a1 1 0 100 2h5a1 1 0 001-1V9a1 1 0 10-2 0v2.586l-4.293-4.293a1 1 0 00-1.414 0L8 9.586 3.707 5.293a1 1 0 00-1.414 1.414l5 5a1 1 0 001.414 0L11 9.414 14.586 13H12z" clipRule="evenodd" />
                      </svg>
                      <span className="text-sm font-medium text-white">{allocationData.estimatedReduction}% burglaries</span>
                    </div>
                  </div>
                </div>
                <div className="w-1/2 px-2 mb-4">
                  <div className="p-3 rounded-lg bg-gray-800 border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">Optimization Score</div>
                    <div className="flex items-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                        <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                      </svg>
                      <span className="text-sm font-medium text-white">85 / 100</span>
                    </div>
                  </div>
                </div>
              </div>
              
              <button 
                className="btn btn-primary w-full py-3"
                onClick={onTogglePoliceAllocation}
              >
                {showPoliceAllocation ? 'Update Optimization' : 'Run Optimization'}
              </button>
            </>
          )}
          
          {selectedTab === 'deployment' && (
            <>
              <div className="overflow-x-auto">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>LSOA</th>
                      <th>Officers</th>
                      <th>Est. Reduction</th>
                      <th>Priority</th>
                    </tr>
                  </thead>
                  <tbody>
                    {allocationData.optimizedAreas.map((area, index) => (
                      <tr key={index}>
                        <td className="font-mono text-gray-300">{area.lsoa}</td>
                        <td className="text-center">{area.officerCount}</td>
                        <td className="text-green-400 font-medium">{area.reductionPercentage}%</td>
                        <td>
                          <div className="px-2 py-1 rounded text-xs font-medium inline-block
                            ${area.reductionPercentage > 40 ? 'bg-red-900/40 text-red-400' : 
                              area.reductionPercentage > 35 ? 'bg-orange-900/40 text-orange-400' : 
                              'bg-blue-900/40 text-blue-400'}"
                          >
                            {area.reductionPercentage > 40 ? 'High' : area.reductionPercentage > 35 ? 'Medium' : 'Normal'}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              
              <div className="mt-4 p-3 rounded-lg bg-blue-900/20 border border-blue-900/30">
                <div className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-blue-400 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                  </svg>
                  <div className="text-sm text-gray-300">
                    Allocation is based on K-means clustering of historical burglary data, with
                    weighting for recency and severity. Toggle allocation to view effect on map.
                  </div>
                </div>
              </div>
            </>
          )}
          
          {selectedTab === 'analytics' && (
            <>
              <div className="p-4 rounded-lg bg-gray-800 border border-gray-700 mb-4">
                <h4 className="font-medium text-white mb-3">Optimization Performance</h4>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-1 text-sm">
                      <span className="text-gray-400">Resource efficiency</span>
                      <span className="text-green-400 font-medium">92%</span>
                    </div>
                    <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div className="h-full bg-green-500" style={{ width: '92%' }}></div>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-1 text-sm">
                      <span className="text-gray-400">Coverage of hot spots</span>
                      <span className="text-blue-400 font-medium">78%</span>
                    </div>
                    <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div className="h-full bg-blue-500" style={{ width: '78%' }}></div>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-1 text-sm">
                      <span className="text-gray-400">Predicted crime reduction</span>
                      <span className="text-yellow-400 font-medium">42%</span>
                    </div>
                    <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div className="h-full bg-yellow-500" style={{ width: '42%' }}></div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="p-3 rounded-lg bg-gray-800 border border-gray-700">
                  <div className="text-xs text-gray-400 mb-1">Estimated Cost</div>
                  <div className="text-sm font-medium text-white">£12,480 / day</div>
                </div>
                <div className="p-3 rounded-lg bg-gray-800 border border-gray-700">
                  <div className="text-xs text-gray-400 mb-1">ROI Score</div>
                  <div className="text-sm font-medium text-green-400">3.5x</div>
                </div>
              </div>
              
              <div className="p-3 rounded-lg bg-gray-800 border border-gray-700 mb-4">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-white">Deployment Timeline</h4>
                  <div className="text-xs text-gray-400">Next 7 days</div>
                </div>
                
                <div className="grid grid-cols-7 gap-1 mb-1">
                  {['M', 'T', 'W', 'T', 'F', 'S', 'S'].map((day, i) => (
                    <div key={i} className="text-center text-xs text-gray-400">
                      {day}
                    </div>
                  ))}
                </div>
                
                <div className="grid grid-cols-7 gap-1">
                  {[45, 42, 48, 45, 50, 32, 28].map((count, i) => (
                    <div 
                      key={i} 
                      className="h-8 rounded-sm relative overflow-hidden bg-gray-700"
                    >
                      <div 
                        className={`absolute bottom-0 left-0 right-0 ${
                          count > 45 ? 'bg-red-500' :
                          count > 40 ? 'bg-orange-500' :
                          count > 35 ? 'bg-yellow-500' :
                          'bg-green-500'
                        }`}
                        style={{ height: `${(count / 50) * 100}%` }}
                      ></div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default PoliceAllocation; 