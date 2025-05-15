import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
// import { getPoliceAllocation } from '../api/backendService';

interface PoliceAllocationProps {
  visible?: boolean;
  showPoliceAllocation?: boolean;
  onToggle?: () => void;
  onTogglePoliceAllocation?: () => void;
  onPoliceDataLoaded?: (data: any[]) => void;
  onMetricsUpdate?: (metrics: any) => void;
}

const PoliceAllocation: React.FC<PoliceAllocationProps> = ({ 
  visible,
  showPoliceAllocation,
  onToggle,
  onTogglePoliceAllocation,
  onPoliceDataLoaded,
  onMetricsUpdate
}) => {
  const [policeUnits, setPoliceUnits] = useState(100);
  const [deploymentHours, setDeploymentHours] = useState(2);
  const [deploymentTime, setDeploymentTime] = useState('08:00');
  const [isLoading, setIsLoading] = useState(true);
  const [policeData, setPoliceData] = useState<any[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [enabled, setEnabled] = useState(false);
  
  // Handle toggle changes
  const handleToggle = () => {
    if (onToggle) {
      onToggle();
      setEnabled(!enabled);
    }
  };
  
  // Use showPoliceAllocation if provided, otherwise fallback to visible
  const isVisible = typeof showPoliceAllocation === 'boolean' ? showPoliceAllocation : visible;
  
  if (!isVisible) return null;
  
  // Replace getPoliceAllocation with a local mock function
  const mockPoliceAllocation = async (units = 100, deploymentHours = 2, deploymentTime = '08:00') => {
    // Return an array of mock police units
    return Array.from({ length: units }, (_, i) => ({
      unit_id: `PU-${String(i + 1).padStart(3, '0')}`,
      estimated_burglaries: Math.floor(Math.random() * 30) + 5,
      effectiveness_score: Math.floor(Math.random() * 30) + 60,
      patrol_type: Math.random() > 0.6 ? 'Vehicle' : 'Foot',
      deployment_hours: deploymentHours,
      deployment_time: deploymentTime,
      latitude: 51.5 + (Math.random() * 0.1) - 0.05,
      longitude: -0.12 + (Math.random() * 0.2) - 0.1,
    }));
  };
  
  // In loadPoliceData, replace getPoliceAllocation with mockPoliceAllocation
  const loadPoliceData = async (units = 100) => {
    try {
      setIsLoading(true);
      setError(null);
      // const data = await getPoliceAllocation(units, deploymentHours, deploymentTime);
      const data = await mockPoliceAllocation(units, deploymentHours, deploymentTime);
      setPoliceData(data);
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
    loadPoliceData(policeUnits);
  }, []);
  
  // Reload data when deployment parameters change
  useEffect(() => {
    if (!isLoading) {
      loadPoliceData(policeUnits);
    }
  }, [deploymentHours, deploymentTime]);
  
  // Calculate summary metrics from police data
  const calculateMetrics = () => {
    if (!policeData || !policeData.length) return null;
    
    const totalUnits = policeData.length;
    const totalBurglaries = policeData.reduce((sum, unit) => sum + unit.estimated_burglaries, 0);
    const avgEffectiveness = policeData.reduce((sum, unit) => sum + unit.effectiveness_score, 0) / totalUnits;
    const vehiclePatrols = policeData.filter(unit => unit.patrol_type === 'Vehicle').length;
    const footPatrols = policeData.filter(unit => unit.patrol_type === 'Foot').length;
    
    const metrics = {
      totalUnits,
      totalBurglaries,
      avgEffectiveness,
      vehiclePatrols,
      footPatrols
    };
    
    if (onMetricsUpdate) {
      onMetricsUpdate(metrics);
    }
    
    return metrics;
  };
  
  const metrics = calculateMetrics();
  
  return (
    <motion.div 
      className="bg-gray-900/90 backdrop-blur-sm border border-gray-800/50 rounded-xl shadow-2xl shadow-black/30 p-4 ml-6 mt-4 w-72 text-gray-300"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex flex-col">
        <motion.div className="flex justify-between items-center">
          <div className="flex items-center mb-4">
            <span className="text-blue-400 mr-2 text-xl">🛡️</span>
            <span className="text-lg font-semibold text-white">Resource Allocator</span>
          </div>
          
          <div className="flex items-center">
            <label className="toggle-switch">
              <input 
                type="checkbox" 
                checked={enabled} 
                onChange={handleToggle}
              />
              <span className="toggle-slider"></span>
            </label>
            <span className="ml-2 text-xs text-gray-400">{enabled ? 'Enabled' : 'Disabled'}</span>
          </div>
        </motion.div>
        
        <div>
          <div className="border-b border-gray-700/50 pb-2 mb-4">
            <h3 className="text-sm font-medium text-white">Resource Configuration</h3>
          </div>
          
          <div>
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-medium">Police Units</label>
              <span className="text-blue-400 font-semibold">{policeUnits} units</span>
            </div>
            <div className="relative pt-1">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-gray-500">0</span>
                <span className="text-xs text-gray-500">50</span>
                <span className="text-xs text-gray-500">100</span>
                <span className="text-xs text-gray-500">150</span>
                <span className="text-xs text-gray-500">200</span>
              </div>
              <input
                type="range"
                className="resource-slider"
                min="0"
                max="200"
                step="1"
                value={policeUnits}
                onChange={(e) => setPoliceUnits(parseInt(e.target.value))}
              />
            </div>
          </div>
          
          <div className="mt-6">
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-medium">Deployment Hours</label>
              <div className="flex items-center">
                <select 
                  className="bg-gray-800 text-white text-sm rounded-md border border-gray-700 px-2 py-1"
                  value={deploymentHours}
                  onChange={(e) => setDeploymentHours(parseInt(e.target.value))}
                >
                  <option value="2">2 hours</option>
                  <option value="4">4 hours</option>
                  <option value="6">6 hours</option>
                  <option value="8">8 hours</option>
                </select>
              </div>
            </div>
            
            <div className="mt-3">
              <div className="flex justify-between items-center mb-2">
                <label className="text-sm font-medium">Start Time</label>
                <div className="flex items-center">
                  <select 
                    className="bg-gray-800 text-white text-sm rounded-md border border-gray-700 px-2 py-1"
                    value={deploymentTime}
                    onChange={(e) => setDeploymentTime(e.target.value)}
                  >
                    {deploymentHours === 2 && (
                      <>
                        <option value="08:00">08:00 - 10:00</option>
                        <option value="10:00">10:00 - 12:00</option>
                        <option value="12:00">12:00 - 14:00</option>
                        <option value="14:00">14:00 - 16:00</option>
                        <option value="16:00">16:00 - 18:00</option>
                        <option value="18:00">18:00 - 20:00</option>
                        <option value="20:00">20:00 - 22:00</option>
                        <option value="22:00">22:00 - 00:00</option>
                        <option value="00:00">00:00 - 02:00</option>
                        <option value="02:00">02:00 - 04:00</option>
                        <option value="04:00">04:00 - 06:00</option>
                        <option value="06:00">06:00 - 08:00</option>
                      </>
                    )}
                    
                    {deploymentHours === 4 && (
                      <>
                        <option value="08:00">08:00 - 12:00</option>
                        <option value="12:00">12:00 - 16:00</option>
                        <option value="16:00">16:00 - 20:00</option>
                        <option value="20:00">20:00 - 00:00</option>
                        <option value="00:00">00:00 - 04:00</option>
                        <option value="04:00">04:00 - 08:00</option>
                      </>
                    )}
                    
                    {deploymentHours === 6 && (
                      <>
                        <option value="06:00">06:00 - 12:00</option>
                        <option value="12:00">12:00 - 18:00</option>
                        <option value="18:00">18:00 - 00:00</option>
                        <option value="00:00">00:00 - 06:00</option>
                      </>
                    )}
                    
                    {deploymentHours === 8 && (
                      <>
                        <option value="08:00">08:00 - 16:00</option>
                        <option value="16:00">16:00 - 00:00</option>
                        <option value="00:00">00:00 - 08:00</option>
                      </>
                    )}
                  </select>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-3 mt-4">
          <div className="bg-gray-800/70 p-3 rounded-lg border border-gray-700/50">
            <h3 className="text-xs text-gray-400 mb-1">Method</h3>
            <div className="flex items-center">
              <span className="text-xs mr-2">🏘️</span>
              <span className="text-sm font-medium">CPTED-Based</span>
            </div>
          </div>
          <div className="bg-gray-800/70 p-3 rounded-lg border border-gray-700/50">
            <h3 className="text-xs text-gray-400 mb-1">Coverage</h3>
            <div className="flex items-center">
              <span className="text-green-400 mr-1">✓</span>
              <span className="text-sm font-medium">91.2% of hot spots</span>
            </div>
          </div>
          <div className="bg-gray-800/70 p-3 rounded-lg border border-gray-700/50">
            <h3 className="text-xs text-gray-400 mb-1">Patrol Types</h3>
            <div className="flex items-center">
              <span className="text-xs mr-2">🚓</span>
              <span className="text-sm font-medium">42 Vehicle / 58 Foot</span>
            </div>
          </div>
          <div className="bg-gray-800/70 p-3 rounded-lg border border-gray-700/50">
            <h3 className="text-xs text-gray-400 mb-1">Avg. Effectiveness</h3>
            <div className="flex items-center">
              <span className="text-xs mr-2">⭐</span>
              <span className="text-sm font-medium">87.5%</span>
            </div>
          </div>
          <div className="bg-gray-800/70 p-3 rounded-lg border border-gray-700/50 col-span-2">
            <h3 className="text-xs text-gray-400 mb-1">Deployment Time</h3>
            <div className="flex items-center">
              <span className="text-xs mr-2">🕒</span>
              <span className="text-sm font-medium">
                {deploymentHours === 2 && `${deploymentTime} - ${parseInt(deploymentTime.split(':')[0]) + 2}:00`}
                {deploymentHours === 4 && `${deploymentTime} - ${parseInt(deploymentTime.split(':')[0]) + 4}:00`}
                {deploymentHours === 6 && `${deploymentTime} - ${parseInt(deploymentTime.split(':')[0]) + 6}:00`}
                {deploymentHours === 8 && `${deploymentTime} - ${parseInt(deploymentTime.split(':')[0]) + 8}:00`}
              </span>
            </div>
          </div>
        </div>
        
        <div className="mt-4 p-3 bg-gray-800/70 rounded-lg border border-gray-700/50">
          <h3 className="text-sm font-medium text-white mb-2">CPTED Deployment Strategies</h3>
          
          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between">
              <span>Natural Surveillance</span>
              <div className="w-16 bg-gray-700 rounded-full h-1.5">
                <div className="bg-blue-500 h-1.5 rounded-full" style={{ width: '80%' }}></div>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span>Access Control</span>
              <div className="w-16 bg-gray-700 rounded-full h-1.5">
                <div className="bg-blue-500 h-1.5 rounded-full" style={{ width: '95%' }}></div>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span>Territorial Reinforcement</span>
              <div className="w-16 bg-gray-700 rounded-full h-1.5">
                <div className="bg-blue-500 h-1.5 rounded-full" style={{ width: '75%' }}></div>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span>Activity Support</span>
              <div className="w-16 bg-gray-700 rounded-full h-1.5">
                <div className="bg-blue-500 h-1.5 rounded-full" style={{ width: '60%' }}></div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="mt-4">
          <button
            className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-500 text-white font-medium rounded-lg shadow"
            onClick={() => loadPoliceData(policeUnits)}
          >
            Apply Allocation
          </button>
          
          <div className="mt-2 text-xs text-gray-500 text-center">
            CPTED strategy optimizes police presence based on environmental design principles
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default PoliceAllocation;
