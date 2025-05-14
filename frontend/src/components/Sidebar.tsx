import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { BarChart, Map, Shield, TrendingUp, ChevronLeft, ChevronRight, Info } from 'lucide-react';
import SidebarPoliceControl from './SidebarPoliceControl';
import EmmieScorecard from './EmmieScorecard';

interface SidebarProps {
  activeView: string;
  setActiveView: (view: string) => void;
  showPoliceAllocation: boolean;
  onTogglePoliceAllocation: () => void;
  selectedLSOA: string | null;
}

const Sidebar: React.FC<SidebarProps> = ({ 
  activeView, 
  setActiveView, 
  showPoliceAllocation, 
  onTogglePoliceAllocation, 
  selectedLSOA 
}) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  
  const navItems = [
    { id: 'dashboard', name: 'Dashboard', icon: <BarChart size={22} /> },
    { id: 'map', name: 'Map View', icon: <Map size={22} /> },
    { id: 'allocation', name: 'Police Allocation', icon: <Shield size={22} /> },
    { id: 'analytics', name: 'Data Analytics', icon: <TrendingUp size={22} /> },
    { id: 'emmie', name: 'EMMIE Framework', icon: <Info size={22} /> },
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
      className="bg-gradient-to-b from-gray-900 to-gray-950 h-screen border-r border-gray-800/50 flex flex-col overflow-y-auto fixed left-0 top-0 z-10 shadow-xl shadow-black/30"
      initial="expanded"
      animate={isCollapsed ? "collapsed" : "expanded"}
      variants={sidebarVariants}
      transition={{ duration: 0.3, ease: "easeInOut" }}
    >
      {/* Logo and Title */}
      <div className="p-4 border-b border-indigo-900/30 bg-gradient-to-r from-gray-900 to-indigo-900/10">
        <div className="flex items-center">
          <motion.div 
            className="text-xl font-bold flex items-center w-full justify-center"
            whileHover={{ scale: 1.05 }}
            transition={{ duration: 0.2 }}
          >
            <motion.span 
              className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 tracking-wide drop-shadow-lg"
              variants={logoTextVariants}
            >
              CRIME FORECASTING
            </motion.span>
          </motion.div>
        </div>
      </div>

      {/* Last Updated */}
      <motion.div 
        className="flex items-center justify-between px-4 py-3 mx-3 mt-3 bg-gray-800/50 rounded-lg border border-gray-700/30 backdrop-blur-sm shadow-lg shadow-black/10 hover:shadow-indigo-900/20 transition-all duration-300"
        whileHover={{ scale: 1.02, backgroundColor: "rgba(30, 41, 59, 0.7)" }}
        transition={{ duration: 0.2 }}
        variants={isCollapsed ? { expanded: {}, collapsed: { display: 'none' } } : {}}
      >
        <span className="text-gray-400 text-sm">Last updated:</span> 
        <span className="font-mono text-blue-400 text-sm font-semibold">May 2025</span>
      </motion.div>

      <motion.div 
        className="p-3 bg-red-900/20 border border-red-900/20 mx-3 mt-3 rounded-lg shadow-lg shadow-red-900/5 text-center"
        whileHover={{ scale: 1.02, backgroundColor: "rgba(127, 29, 29, 0.25)" }}
        transition={{ duration: 0.2 }}
        variants={isCollapsed ? { expanded: {}, collapsed: { display: 'none' } } : {}}
      >
        <p className="text-xs text-red-400">This is confidential data, click here to read terms and services</p>
      </motion.div>

      {/* Filters - Only visible when expanded */}
      <motion.div 
        className="px-4 py-3"
        variants={logoTextVariants}
      >
        <h3 className="text-sm font-medium text-white mb-1 text-center">Filters</h3>
        <p className="text-xs text-gray-400 mb-4 text-center">Filter dashboard by division or time period.</p>
      </motion.div>

      {/* Navigation Menu */}
      <nav className="pt-2">
        <div className="px-4 mb-2 flex justify-center">
          <h3 className="text-xs font-medium text-gray-400 uppercase tracking-wider flex items-center">
            <span className="mr-2">🧭</span>
            Navigation
          </h3>
        </div>
        <div className="space-y-2 px-3">
          {navItems.map((item) => (
            <motion.div
              key={item.id}
              whileHover={{ scale: 1.03 }}
              transition={{
                type: "spring", 
                stiffness: 400,
                damping: 20
              }}
            >
              <button
                className={`
                  flex items-center w-full px-4 py-3
                  rounded-lg text-sm font-medium
                  transition-all duration-300 ease-out
                  backdrop-blur-sm
                  ${activeView === item.id
                    ? 'bg-gradient-to-r from-indigo-600/40 to-blue-500/20 text-blue-300 shadow-lg shadow-blue-900/30 border-l-2 border-blue-400 hover:from-indigo-500/50 hover:to-blue-400/30'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/90 hover:shadow-lg hover:shadow-gray-900/30 hover:border-l hover:border-gray-600'
                  }
                `}
                onClick={() => setActiveView(item.id)}
              >
                <span className="text-xl mr-3 opacity-90">{item.icon}</span>
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
      <div className="border-t border-gray-800/50 mt-4 pt-4 bg-gradient-to-b from-transparent to-gray-800/20">
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
      <div className="p-4 mt-auto border-t border-gray-800/70 bg-gradient-to-t from-gray-950 to-transparent">
        <motion.button 
          className="flex items-center justify-center w-full p-2 rounded-lg bg-gray-800 text-gray-400 hover:bg-indigo-800/50 hover:text-white transition-all duration-300 shadow-lg shadow-black/20 border border-gray-700/30 backdrop-blur-sm"
          onClick={() => setIsCollapsed(!isCollapsed)}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {isCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
        </motion.button>
      </div>
    </motion.div>
  );
};

export default Sidebar;
