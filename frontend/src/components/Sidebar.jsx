import React, { useState } from 'react';
import { motion } from 'framer-motion';

const Sidebar = ({ activeView, setActiveView }) => {
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
            <span className="text-blue-500 mr-2">CF</span>
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

      <motion.div 
        className="px-4 mb-4"
        variants={logoTextVariants}
      >
        <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">Police Division</h4>
        <select className="w-full rounded-sm bg-gray-800 border border-gray-700 px-2 py-1 text-sm text-gray-300 focus:outline-none focus:ring-1 focus:ring-blue-500/50">
          <option>All</option>
          <option>Central</option>
          <option>North</option>
          <option>South</option>
        </select>
      </motion.div>

      {/* Navigation */}
      <nav className="flex-1 pt-2">
        <div className="px-4 mb-2">
          <h3 className="text-xs font-medium text-gray-400 uppercase tracking-wider">Navigation</h3>
        </div>
        <ul>
          {navItems.map((item) => (
            <motion.li 
              key={item.id} 
              className="mb-1 px-2"
              whileHover={{ x: 5 }}
              transition={{ duration: 0.2 }}
            >
              <button
                className={`flex items-center w-full px-2 py-2 rounded-md text-sm font-medium transition-colors duration-150 
                  ${activeView === item.id 
                    ? 'bg-blue-700/20 text-blue-400 border-l-2 border-blue-500' 
                    : 'text-gray-400 hover:text-gray-100 hover:bg-gray-800/50'}`}
                onClick={() => setActiveView(item.id)}
              >
                <span className="mr-2">{item.icon}</span>
                <motion.span variants={logoTextVariants}>{item.name}</motion.span>
              </button>
            </motion.li>
          ))}
        </ul>
      </nav>

      {/* Collapse/Expand toggle */}
      <div className="p-4 border-t border-gray-800">
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