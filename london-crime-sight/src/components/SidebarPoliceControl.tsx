
import React from 'react';
import { motion } from 'framer-motion';

interface SidebarPoliceControlProps {
  showPoliceAllocation: boolean;
  onTogglePoliceAllocation: () => void;
  isCollapsed: boolean;
}

const SidebarPoliceControl: React.FC<SidebarPoliceControlProps> = ({
  showPoliceAllocation,
  onTogglePoliceAllocation,
  isCollapsed
}) => {
  const logoTextVariants = {
    expanded: { opacity: 1, display: 'block' },
    collapsed: { opacity: 0, display: 'none' }
  };

  return (
    <div className="px-4 py-3">
      <div className="flex items-center justify-between mb-2">
        <motion.div variants={logoTextVariants}>
          <h3 className="text-sm font-medium text-white">Police Units</h3>
        </motion.div>

        <div className="flex items-center">
          <label className="toggle-switch">
            <input 
              type="checkbox" 
              checked={showPoliceAllocation}
              onChange={onTogglePoliceAllocation}
            />
            <span className="toggle-slider"></span>
          </label>
        </div>
      </div>

      <motion.div 
        className="bg-gray-800/60 px-4 py-3 rounded-lg border border-gray-700/30"
        variants={isCollapsed ? { expanded: {}, collapsed: { display: 'none' } } : {}}
      >
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center">
            <span className="text-blue-400 mr-2 text-xl">👮</span>
            <span className="text-sm text-gray-300 font-medium">Active Units</span>
          </div>
          <span className="text-blue-400 font-medium">100</span>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <span className="text-green-400 mr-2 text-xl">🚓</span>
            <span className="text-sm text-gray-300 font-medium">Coverage</span>
          </div>
          <span className="text-green-400 font-medium">68%</span>
        </div>
      </motion.div>
    </div>
  );
};

export default SidebarPoliceControl;
