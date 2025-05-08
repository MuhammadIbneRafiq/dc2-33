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
          <span className="text-blue-400 mr-2">👮</span>
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
                        -
                      </button>
                      <button className="p-1 bg-gray-800 rounded-r border-t border-r border-b border-gray-700 text-gray-400 hover:text-white hover:bg-gray-700">
                        +
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
                      <span className="text-blue-400 mr-1">📊</span>
                      <span className="text-sm font-medium text-white">{allocationData.deploymentMethod}</span>
                    </div>
                  </div>
                </div>
                <div className="w-1/2 px-2 mb-4">
                  <div className="p-3 rounded-lg bg-gray-800 border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">Coverage</div>
                    <div className="flex items-center">
                      <span className="text-green-400 mr-1">✓</span>
                      <span className="text-sm font-medium text-white">{allocationData.coveragePercentage}% of hot spots</span>
                    </div>
                  </div>
                </div>
                <div className="w-1/2 px-2 mb-4">
                  <div className="p-3 rounded-lg bg-gray-800 border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">Predicted Reduction</div>
                    <div className="flex items-center">
                      <span className="text-green-400 mr-1">↓</span>
                      <span className="text-sm font-medium text-white">{allocationData.estimatedReduction}% burglaries</span>
                    </div>
                  </div>
                </div>
                <div className="w-1/2 px-2 mb-4">
                  <div className="p-3 rounded-lg bg-gray-800 border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">Optimization Score</div>
                    <div className="flex items-center">
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
                  <span className="text-blue-400 mr-2">ℹ️</span>
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
                <h4 className="text-sm font-medium text-white mb-3">Performance Metrics</h4>
                
                <div className="space-y-3">
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <label className="text-xs text-gray-400">Response Time</label>
                      <span className="text-xs text-green-400">7.2 min</span>
                    </div>
                    <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div className="h-full bg-green-500" style={{ width: '75%' }}></div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <label className="text-xs text-gray-400">Coverage Efficiency</label>
                      <span className="text-xs text-blue-400">82%</span>
                    </div>
                    <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div className="h-full bg-blue-500" style={{ width: '82%' }}></div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <label className="text-xs text-gray-400">Resource Utilization</label>
                      <span className="text-xs text-yellow-400">68%</span>
                    </div>
                    <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div className="h-full bg-yellow-500" style={{ width: '68%' }}></div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="p-4 rounded-lg bg-gray-800 border border-gray-700">
                <h4 className="text-sm font-medium text-white mb-3">Impact Assessment</h4>
                
                <div className="flex items-center mb-4">
                  <div className="text-3xl font-bold text-blue-400">-42%</div>
                  <div className="ml-3">
                    <div className="text-sm text-white">Estimated Burglary Reduction</div>
                    <div className="text-xs text-gray-400">Based on historical response data</div>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-2 rounded bg-gray-900 border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">Cost Efficiency</div>
                    <div className="text-sm text-white">£2,450 / incident</div>
                  </div>
                  
                  <div className="p-2 rounded bg-gray-900 border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">Officer Safety</div>
                    <div className="text-sm text-green-400">94% rating</div>
                  </div>
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