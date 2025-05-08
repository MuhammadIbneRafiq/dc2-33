import React, { useState, useEffect } from 'react';
import { getPoliceAllocation } from '../api/backendService';

const PoliceAllocation = ({ showPoliceAllocation, onTogglePoliceAllocation, onPoliceDataLoaded }) => {
  const [selectedTab, setSelectedTab] = useState('optimization');
  const [isLoading, setIsLoading] = useState(true);
  const [policeData, setPoliceData] = useState(null);
  const [unitCount, setUnitCount] = useState(100);
  const [deploymentHours, setDeploymentHours] = useState(8);
  const [error, setError] = useState(null);
  
  // Load police allocation data
  const loadPoliceData = async (units = 100) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const data = await getPoliceAllocation(units);
      setPoliceData(data);
      
      // Pass the data to parent component for map display
      if (onPoliceDataLoaded) {
        onPoliceDataLoaded(data);
      }
      
    } catch (error) {
      console.error('Error loading police data:', error);
      setError('Failed to load resource allocation data. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Load initial data
  useEffect(() => {
    loadPoliceData(unitCount);
  }, []);
  
  // Handle unit count change
  const handleUnitCountChange = (newCount) => {
    setUnitCount(newCount);
  };
  
  // Handle update button click
  const handleUpdate = () => {
    loadPoliceData(unitCount);
    if (!showPoliceAllocation) {
      onTogglePoliceAllocation();
    }
  };
  
  // Calculate summary metrics from police data
  const calculateMetrics = () => {
    if (!policeData || !policeData.length) return null;
    
    const totalUnits = policeData.length;
    const totalBurglaries = policeData.reduce((sum, unit) => sum + unit.estimated_burglaries, 0);
    const avgEffectiveness = policeData.reduce((sum, unit) => sum + unit.effectiveness_score, 0) / totalUnits;
    const vehiclePatrols = policeData.filter(unit => unit.patrol_type === 'Vehicle').length;
    const footPatrols = policeData.filter(unit => unit.patrol_type === 'Foot').length;
    
    return {
      totalUnits,
      totalBurglaries,
      avgEffectiveness,
      vehiclePatrols,
      footPatrols
    };
  };
  
  const metrics = calculateMetrics();
  
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
                disabled={isLoading || !policeData}
              />
              <span className="toggle-slider"></span>
            </label>
            <span className="text-sm text-gray-300">
              {showPoliceAllocation ? 'Enabled' : 'Disabled'}
            </span>
          </div>
        </div>
      </div>
      
      {error && (
        <div className="p-4 bg-red-900/30 border border-red-800 m-4 rounded-md">
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}
      
      {isLoading ? (
        <div className="p-10 flex flex-col justify-center items-center">
          <div className="w-10 h-10 border-3 border-t-blue-500 border-blue-500/20 rounded-full animate-spin mb-3"></div>
          <p className="text-gray-400 text-sm">Optimizing resource allocation...</p>
          <p className="text-gray-500 text-xs mt-1">This may take a moment for large datasets</p>
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
          
          {selectedTab === 'optimization' && metrics && (
            <>
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm text-gray-300">Police Units</label>
                  <span className="text-blue-400 font-medium text-sm">{metrics.totalUnits} units</span>
                </div>
                <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-blue-500 to-green-500" 
                    style={{ width: `${(metrics.totalUnits / 200) * 100}%` }}
                  ></div>
                </div>
                <div className="flex justify-between mt-1 text-xs text-gray-400">
                  <span>0</span>
                  <span>50</span>
                  <span>100</span>
                  <span>150</span>
                  <span>200</span>
                </div>
              </div>
              
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm text-gray-300">Deployment Hours</label>
                  <div className="flex items-center">
                    <span className="text-blue-400 font-medium text-sm">{deploymentHours} hours</span>
                    <div className="flex items-center ml-2">
                      <button 
                        className="p-1 bg-gray-800 rounded-l border border-gray-700 text-gray-400 hover:text-white hover:bg-gray-700"
                        onClick={() => setDeploymentHours(Math.max(4, deploymentHours - 1))}
                      >
                        -
                      </button>
                      <button 
                        className="p-1 bg-gray-800 rounded-r border-t border-r border-b border-gray-700 text-gray-400 hover:text-white hover:bg-gray-700"
                        onClick={() => setDeploymentHours(Math.min(12, deploymentHours + 1))}
                      >
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
                  value={deploymentHours}
                  onChange={(e) => setDeploymentHours(parseInt(e.target.value))}
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
                      <span className="text-sm font-medium text-white">K-means Clustering</span>
                    </div>
                  </div>
                </div>
                <div className="w-1/2 px-2 mb-4">
                  <div className="p-3 rounded-lg bg-gray-800 border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">Coverage</div>
                    <div className="flex items-center">
                      <span className="text-green-400 mr-1">✓</span>
                      <span className="text-sm font-medium text-white">{(metrics.totalUnits / 120 * 100).toFixed(1)}% of hot spots</span>
                    </div>
                  </div>
                </div>
                <div className="w-1/2 px-2 mb-4">
                  <div className="p-3 rounded-lg bg-gray-800 border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">Patrol Types</div>
                    <div className="flex items-center">
                      <span className="text-blue-400 mr-1">🚓</span>
                      <span className="text-sm font-medium text-white">{metrics.vehiclePatrols} Vehicle / {metrics.footPatrols} Foot</span>
                    </div>
                  </div>
                </div>
                <div className="w-1/2 px-2 mb-4">
                  <div className="p-3 rounded-lg bg-gray-800 border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">Avg. Effectiveness</div>
                    <div className="flex items-center">
                      <span className="text-green-400 mr-1">⭐</span>
                      <span className="text-sm font-medium text-white">{metrics.avgEffectiveness.toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
                <div className="w-full px-2 mb-4">
                  <div className="p-3 rounded-lg bg-gray-800 border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">Number of Units</div>
                    <div className="flex items-center justify-between">
                      <input
                        type="number"
                        min="10"
                        max="200"
                        step="10"
                        value={unitCount}
                        onChange={(e) => handleUnitCountChange(parseInt(e.target.value))}
                        className="bg-gray-700 border border-gray-600 text-white px-2 py-1 rounded w-20 text-sm"
                      />
                      <div className="flex">
                        <button 
                          className="px-2 py-1 bg-gray-700 text-xs text-white rounded-l border border-gray-600 hover:bg-gray-600"
                          onClick={() => handleUnitCountChange(Math.max(10, unitCount - 10))}
                        >
                          -10
                        </button>
                        <button 
                          className="px-2 py-1 bg-gray-700 text-xs text-white rounded-r border-t border-r border-b border-gray-600 hover:bg-gray-600"
                          onClick={() => handleUnitCountChange(Math.min(200, unitCount + 10))}
                        >
                          +10
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              <button 
                className="btn btn-primary w-full py-3"
                onClick={handleUpdate}
                disabled={isLoading}
              >
                {isLoading ? 'Optimizing...' : 'Update Optimization'}
              </button>
            </>
          )}
          
          {selectedTab === 'deployment' && policeData && (
            <>
              <div className="overflow-x-auto max-h-80 scrollbar-thin">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Unit ID</th>
                      <th>Est. Burglaries</th>
                      <th>Effectiveness</th>
                      <th>Patrol Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {policeData.slice(0, 20).map((unit) => (
                      <tr key={unit.unit_id}>
                        <td className="font-mono text-gray-300">{unit.unit_id}</td>
                        <td className="text-center">{unit.estimated_burglaries.toLocaleString()}</td>
                        <td className="text-green-400 font-medium">{unit.effectiveness_score}%</td>
                        <td>
                          <div className={`px-2 py-1 rounded text-xs font-medium inline-block
                            ${unit.patrol_type === 'Vehicle' ? 'bg-blue-900/40 text-blue-400' : 'bg-green-900/40 text-green-400'}`}
                          >
                            {unit.patrol_type}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              
              {policeData.length > 20 && (
                <div className="text-center text-gray-400 text-xs mt-2">
                  Showing 20 of {policeData.length} units. View all on the map.
                </div>
              )}
              
              <div className="mt-4 p-3 rounded-lg bg-blue-900/20 border border-blue-900/30">
                <div className="flex items-start">
                  <span className="text-blue-400 mr-2">ℹ️</span>
                  <div className="text-sm text-gray-300">
                    Allocation is based on K-means clustering of historical burglary data, with
                    weighting for recency and severity. Toggle allocation to view units on the map.
                  </div>
                </div>
              </div>
            </>
          )}
          
          {selectedTab === 'analytics' && metrics && (
            <>
              <div className="p-4 rounded-lg bg-gray-800 border border-gray-700 mb-4">
                <h4 className="text-sm font-medium text-white mb-3">Resource Distribution</h4>
                
                <div className="space-y-3">
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <label className="text-xs text-gray-400">Vehicle Patrols</label>
                      <span className="text-xs text-blue-400">{metrics.vehiclePatrols}</span>
                    </div>
                    <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div className="h-full bg-blue-500" style={{ width: `${(metrics.vehiclePatrols / metrics.totalUnits) * 100}%` }}></div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <label className="text-xs text-gray-400">Foot Patrols</label>
                      <span className="text-xs text-green-400">{metrics.footPatrols}</span>
                    </div>
                    <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div className="h-full bg-green-500" style={{ width: `${(metrics.footPatrols / metrics.totalUnits) * 100}%` }}></div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <label className="text-xs text-gray-400">Effectiveness Score</label>
                      <span className="text-xs text-yellow-400">{metrics.avgEffectiveness.toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div className="h-full bg-yellow-500" style={{ width: `${metrics.avgEffectiveness}%` }}></div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="p-4 rounded-lg bg-gray-800 border border-gray-700">
                <h4 className="text-sm font-medium text-white mb-3">Estimated Impact</h4>
                
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 rounded bg-gray-700/50 border border-gray-700">
                    <div className="text-2xl font-bold text-blue-400">{metrics.totalBurglaries.toLocaleString()}</div>
                    <div className="text-xs text-gray-400">Burglaries Addressed</div>
                  </div>
                  
                  <div className="p-3 rounded bg-gray-700/50 border border-gray-700">
                    <div className="text-2xl font-bold text-green-400">{Math.round(metrics.totalBurglaries * (metrics.avgEffectiveness / 100) * 0.01).toLocaleString()}</div>
                    <div className="text-xs text-gray-400">Potential Reduction</div>
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