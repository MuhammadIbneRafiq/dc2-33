import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

const SidebarPoliceControl = ({ 
  showPoliceAllocation, 
  onTogglePoliceAllocation,
  isCollapsed
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [officerCount, setOfficerCount] = useState(2);
  const [deploymentTime, setDeploymentTime] = useState(25);
  const [returnVisits, setReturnVisits] = useState(3);
  
  const handleOptimize = () => {
    setIsLoading(true);
    // Simulate optimization process
    setTimeout(() => {
      setIsLoading(false);
      onTogglePoliceAllocation(true);
    }, 1500);
  };
  
  const textVariants = {
    expanded: { opacity: 1, display: 'block' },
    collapsed: { opacity: 0, display: 'none' }
  };

  return (
    <div className="px-4 py-3">
      <motion.div className="mb-2" variants={textVariants}>
        <div className="text-sm font-medium text-white mb-1 flex items-center">
          <span className="text-blue-400 mr-2">👮</span>
          Resource Allocator
        </div>
        <p className="text-xs text-gray-400 mb-3">
          Configure and deploy police resources to optimize crime reduction
        </p>
      </motion.div>
      
      {/* Toggle Button */}
      <div className="flex items-center justify-between mb-4">
        <motion.div variants={textVariants}>
          <span className="text-xs text-gray-300">Police Allocation</span>
        </motion.div>
        <div className="flex items-center">
          <label className="toggle-switch mr-2">
            <input 
              type="checkbox" 
              checked={showPoliceAllocation}
              onChange={() => onTogglePoliceAllocation(!showPoliceAllocation)}
            />
            <span className="toggle-slider"></span>
          </label>
          <motion.span 
            className="text-xs text-gray-300"
            variants={textVariants}
          >
            {showPoliceAllocation ? 'Enabled' : 'Disabled'}
          </motion.span>
        </div>
      </div>
      
      {/* Available Resources Section */}
      <motion.div variants={textVariants}>
        <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">
          Available Resources
        </h4>
        
        <div className="space-y-3">
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs text-gray-300">Units Patrolling:</label>
              <div className="flex items-center">
                <button 
                  onClick={() => setOfficerCount(Math.max(1, officerCount - 1))}
                  className="w-6 h-6 bg-gray-800 rounded-l border border-gray-700 text-gray-400 hover:text-white"
                >
                  -
                </button>
                <div className="w-8 h-6 bg-gray-800 border-t border-b border-gray-700 flex items-center justify-center text-xs text-white">
                  {officerCount}
                </div>
                <button 
                  onClick={() => setOfficerCount(Math.min(5, officerCount + 1))}
                  className="w-6 h-6 bg-gray-800 rounded-r border-t border-r border-b border-gray-700 text-gray-400 hover:text-white"
                >
                  +
                </button>
                <select className="ml-2 w-24 text-xs bg-gray-800 border border-gray-700 rounded-sm p-1 text-white">
                  <option>vehicles</option>
                  <option>officers</option>
                  <option>teams</option>
                </select>
              </div>
            </div>
          </div>
          
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs text-gray-300">Devoting:</label>
              <div className="flex items-center">
                <select className="w-16 text-xs bg-gray-800 border border-gray-700 rounded-sm p-1 text-white">
                  <option>25%</option>
                  <option>50%</option>
                  <option>75%</option>
                  <option>100%</option>
                </select>
                <span className="ml-1 text-xs text-gray-400">of their time</span>
              </div>
            </div>
          </div>
          
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs text-gray-300">Returning to locations:</label>
              <div className="flex items-center">
                <button 
                  onClick={() => setReturnVisits(Math.max(1, returnVisits - 1))}
                  className="w-6 h-6 bg-gray-800 rounded-l border border-gray-700 text-gray-400 hover:text-white"
                >
                  -
                </button>
                <div className="w-8 h-6 bg-gray-800 border-t border-b border-gray-700 flex items-center justify-center text-xs text-white">
                  {returnVisits}
                </div>
                <button 
                  onClick={() => setReturnVisits(Math.min(5, returnVisits + 1))}
                  className="w-6 h-6 bg-gray-800 rounded-r border-t border-r border-b border-gray-700 text-gray-400 hover:text-white"
                >
                  +
                </button>
                <span className="ml-1 text-xs text-gray-400">times during period</span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
      
      {/* Risk Gradient */}
      <motion.div className="mt-4" variants={textVariants}>
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs text-gray-300">Deploy force to:</span>
        </div>
        <div className="h-6 bg-gradient-to-r from-blue-500 via-green-500 to-red-500 rounded-md flex items-center justify-between px-2 mb-1">
          <span className="text-xs text-white font-medium">High risk</span>
          <span className="text-xs text-white font-medium">Low risk</span>
        </div>
      </motion.div>
      
      {/* Publish Missions Button */}
      <div className="mt-4">
        <button 
          className={`w-full p-2 ${isCollapsed ? 'rounded-full' : 'rounded-md'} transition-colors duration-200 ${
            isLoading 
              ? 'bg-gray-700 text-gray-500 cursor-not-allowed' 
              : 'bg-blue-500 hover:bg-blue-600 text-white'
          }`}
          onClick={handleOptimize}
          disabled={isLoading}
        >
          {isLoading ? (
            <div className="flex items-center justify-center">
              <div className="w-4 h-4 border-2 border-gray-400 border-t-gray-600 rounded-full animate-spin mr-2"></div>
              <motion.span variants={textVariants}>Optimizing...</motion.span>
            </div>
          ) : (
            <div className="flex items-center justify-center">
              <span className="mr-1">📡</span>
              <motion.span variants={textVariants}>Publish Missions</motion.span>
            </div>
          )}
        </button>
      </div>
      
      {/* Deploy Map */}
      {showPoliceAllocation && (
        <motion.div 
          className="mt-4 p-3 rounded-md bg-blue-900/20 border border-blue-800/30 text-blue-400 text-xs"
          variants={textVariants}
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          transition={{ duration: 0.3 }}
        >
          <div className="flex items-start">
            <span className="text-blue-400 mr-2">ℹ️</span>
            <p>
              Police resources have been allocated. The map now shows officer distribution based on K-means clustering of burglary hotspots.
            </p>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default SidebarPoliceControl; 