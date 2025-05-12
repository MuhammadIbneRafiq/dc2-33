import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { getPoliceAllocation } from '../api/backendService';

interface PoliceAllocationProps {
  visible: boolean;
  onToggle?: () => void;
}

const PoliceAllocation: React.FC<PoliceAllocationProps> = ({ 
  visible,
  onToggle
}) => {
  const [policeUnits, setPoliceUnits] = useState(100);
  const [deploymentHours, setDeploymentHours] = useState(8);
  const [activeTab, setActiveTab] = useState('optimization');
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
  
  if (!visible) return null;
  
  // Load police allocation data
  const loadPoliceData = async (units = 100) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const data = await getPoliceAllocation(units);
      setPoliceData(data);
      
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
        
        <div className="mb-6">
          <div className="flex space-x-1 mb-4 border-b border-gray-800/50 pb-1">
            <button 
              className={`py-2 px-3 text-sm font-medium rounded-t-lg ${
                activeTab === 'optimization' 
                  ? 'bg-blue-900/20 text-blue-400 border-b-2 border-blue-500' 
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
              }`}
              onClick={() => setActiveTab('optimization')}
            >
              Optimization
            </button>
            <button 
              className={`py-2 px-3 text-sm font-medium rounded-t-lg ${
                activeTab === 'deployment' 
                  ? 'bg-blue-900/20 text-blue-400 border-b-2 border-blue-500' 
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
              }`}
              onClick={() => setActiveTab('deployment')}
            >
              Deployment
            </button>
            <button 
              className={`py-2 px-3 text-sm font-medium rounded-t-lg ${
                activeTab === 'analytics' 
                  ? 'bg-blue-900/20 text-blue-400 border-b-2 border-blue-500' 
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
              }`}
              onClick={() => setActiveTab('analytics')}
            >
              Analytics
            </button>
          </div>
          
          <div className="mt-4">
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
                <button 
                  className="w-6 h-6 bg-gray-800 rounded-md flex items-center justify-center text-gray-400 hover:bg-gray-700"
                  onClick={() => setDeploymentHours(Math.max(4, deploymentHours - 1))}
                >
                  -
                </button>
                <span className="text-blue-400 font-semibold mx-2">{deploymentHours} hours</span>
                <button 
                  className="w-6 h-6 bg-gray-800 rounded-md flex items-center justify-center text-gray-400 hover:bg-gray-700"
                  onClick={() => setDeploymentHours(Math.min(12, deploymentHours + 1))}
                >
                  +
                </button>
              </div>
            </div>
            <div className="relative pt-1">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-gray-500">4h</span>
                <span className="text-xs text-gray-500">6h</span>
                <span className="text-xs text-gray-500">8h</span>
                <span className="text-xs text-gray-500">10h</span>
                <span className="text-xs text-gray-500">12h</span>
              </div>
              <input
                type="range"
                className="resource-slider"
                min="4"
                max="12"
                step="1"
                value={deploymentHours}
                onChange={(e) => setDeploymentHours(parseInt(e.target.value))}
              />
              <div 
                className="absolute h-4 w-4 bg-blue-500 rounded-full -mt-1 border-2 border-white shadow" 
                style={{ 
                  left: `${((deploymentHours - 4) / 8) * 100}%`, 
                  top: '50%' 
                }}
              />
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-3 mt-4">
          <div className="bg-gray-800/70 p-3 rounded-lg border border-gray-700/50">
            <h3 className="text-xs text-gray-400 mb-1">Method</h3>
            <div className="flex items-center">
              <span className="text-xs mr-2">🧮</span>
              <span className="text-sm font-medium">K-means Clustering</span>
            </div>
          </div>
          <div className="bg-gray-800/70 p-3 rounded-lg border border-gray-700/50">
            <h3 className="text-xs text-gray-400 mb-1">Coverage</h3>
            <div className="flex items-center">
              <span className="text-green-400 mr-1">✓</span>
              <span className="text-sm font-medium">83.3% of hot spots</span>
            </div>
          </div>
          <div className="bg-gray-800/70 p-3 rounded-lg border border-gray-700/50">
            <h3 className="text-xs text-gray-400 mb-1">Patrol Types</h3>
            <div className="flex items-center">
              <span className="text-xs mr-2">🚓</span>
              <span className="text-sm font-medium">39 Vehicle / 61 Foot</span>
            </div>
          </div>
          <div className="bg-gray-800/70 p-3 rounded-lg border border-gray-700/50">
            <h3 className="text-xs text-gray-400 mb-1">Avg. Effectiveness</h3>
            <div className="flex items-center">
              <span className="text-xs mr-2">⭐</span>
              <span className="text-sm font-medium">80.1%</span>
            </div>
          </div>
        </div>
        
        <div className="mt-6">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Number of Units</label>
            <select className="bg-gray-800 border border-gray-700 rounded-md text-sm p-1">
              <option>100</option>
              <option>150</option>
              <option>200</option>
            </select>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default PoliceAllocation;
