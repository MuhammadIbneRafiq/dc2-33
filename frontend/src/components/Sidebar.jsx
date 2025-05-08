import React, { useState } from 'react';
import { motion } from 'framer-motion';
import SidebarPoliceControl from './SidebarPoliceControl';
import EmmieScorecard from './EmmieScorecard';

const Sidebar = ({ 
  activeView, 
  setActiveView, 
  showPoliceAllocation, 
  onTogglePoliceAllocation, 
  selectedLSOA 
}) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  
  const navItems = [
    { id: 'dashboard', name: 'Dashboard', icon: '📊' },
    { id: 'map', name: 'Map View', icon: '🗺️' },
    { id: 'allocation', name: 'Police Allocation', icon: '👮' },
    { id: 'analytics', name: 'Data Analytics', icon: '📈' },
  ];

  const sidebarVariants = {
    expanded: { width: '280px' },
    collapsed: { width: '80px' }
  };

  const logoTextVariants = {
    expanded: { opacity: 1, display: 'block' },
    collapsed: { opacity: 0, display: 'none' }
  };

  return (
    <motion.div 
      className="bg-gray-900 h-screen border-r border-gray-800 flex flex-col overflow-y-auto fixed left-0 top-0 z-10"
      initial="expanded"
      animate={isCollapsed ? "collapsed" : "expanded"}
      variants={sidebarVariants}
      transition={{ duration: 0.3, ease: "easeInOut" }}
    >
      {/* Logo and Title */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center">
          <motion.div 
            className="text-xl font-bold flex items-center"
            whileHover={{ scale: 1.05 }}
            transition={{ duration: 0.2 }}
          >

            <motion.span 
              className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400"
              variants={logoTextVariants}
            >
              CRIME FORECASTING
            </motion.span>
          </motion.div>
        </div>
      </div>

      {/* Last Updated */}
      <motion.div 
        className="flex items-center space-x-2 px-3 py-3 mx-3 mt-3 bg-gray-800 rounded-md"
        whileHover={{ backgroundColor: "#1e293b" }}
        transition={{ duration: 0.2 }}
        variants={isCollapsed ? { expanded: {}, collapsed: { display: 'none' } } : {}}
      >
        <span className="text-gray-400 text-sm">Last updated:</span> 
        <span className="font-mono text-blue-400 text-sm font-semibold">Feb 2025</span>
      </motion.div>

      <div className="p-3 bg-red-900/20 border-b border-red-900/20 mx-3 mt-3 rounded-md"
        variants={isCollapsed ? { expanded: {}, collapsed: { display: 'none' } } : {}}
      >
        <p className="text-xs text-red-400">This data is for demonstration purposes only.</p>
      </div>

      {/* Filters - Only visible when expanded */}
      <motion.div 
        className="px-4 py-3"
        variants={logoTextVariants}
      >
        <h3 className="text-sm font-medium text-white mb-1">Filters</h3>
        <p className="text-xs text-gray-400 mb-4">Filter dashboard by division, crime type, or time period.</p>
      </motion.div>

      {/* Navigation Menu */}
      <nav className="pt-4">
        <div className="px-4 mb-3">
          <h3 className="text-xs font-medium text-gray-400 uppercase tracking-wider flex items-center">
            <span className="mr-2">🧭</span>
            Navigation
          </h3>
        </div>
        <div className="space-y-2 px-2">
          {navItems.map((item) => (
            <motion.div
              key={item.id}
              whileHover={{ x: 8 }}
              transition={{
                type: "spring", 
                stiffness: 400,
                damping: 20
              }}
            >
              <button
                className={`
                  flex items-center w-full px-4 py-3
                  rounded-xl text-sm font-medium
                  transition-all duration-300 ease-out
                  backdrop-blur-sm
                  ${activeView === item.id
                    ? 'bg-gradient-to-r from-blue-600/40 to-indigo-500/20 text-blue-300 shadow-lg shadow-blue-900/30 border-l-2 border-blue-400 hover:from-blue-500/50 hover:to-indigo-400/30'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/90 hover:shadow-lg hover:shadow-gray-900/30 hover:border-l-2 hover:border-gray-600'
                  }
                `}
                onClick={() => setActiveView(item.id)}
              >
                <span className="text-xl mr-4 opacity-90">{item.icon}</span>
                <motion.span
                  variants={logoTextVariants}
                  className="font-medium tracking-wide"
                >
                  {item.name}
                </motion.span>
              </button>
            </motion.div>
          ))}
        </div>
      </nav>
      
      {/* Police Control Section */}
      <div className="border-t border-gray-800 mt-4 pt-4">
        <SidebarPoliceControl 
          showPoliceAllocation={showPoliceAllocation}
          onTogglePoliceAllocation={onTogglePoliceAllocation}
          isCollapsed={isCollapsed}
        />
      </div>
      
      {/* EMMIE Scorecard */}
      <motion.div 
        className="mt-4 mx-3 overflow-hidden"
        variants={isCollapsed ? { expanded: {}, collapsed: { display: 'none' } } : {}}
      >
        <EmmieScorecard selectedLSOA={selectedLSOA} />
      </motion.div>

      {/* Collapse/Expand toggle */}
      <div className="p-4 mt-auto border-t border-gray-800">
        <motion.button 
          className="flex items-center justify-center w-full p-2 rounded-md bg-gray-800 text-gray-400 hover:bg-gray-700 transition-colors"
          onClick={() => setIsCollapsed(!isCollapsed)}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {isCollapsed ? '→' : '←'}
        </motion.button>
      </div>
    </motion.div>
  );
};

export default Sidebar; 