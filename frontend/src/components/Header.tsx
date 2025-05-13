// Only modifying the relevant portions
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Slider } from "./ui/slider";
import { Video } from "lucide-react";

interface HeaderProps {
  onOpenTutorial?: () => void;
}

const Header: React.FC<HeaderProps> = ({ onOpenTutorial }) => {
  const [isSearchFocused, setIsSearchFocused] = useState(false);
  const [dateRange, setDateRange] = useState([30]); // Default to 30 days
  const [showRangeSlider, setShowRangeSlider] = useState(false);

  const handleRangeChange = (value: number[]) => {
    setDateRange(value);
  };

  return (
    <header className="w-full">
      {/* Main navbar */}
      <div className="bg-gradient-to-r from-gray-900 via-indigo-950/30 to-gray-900 border-b border-indigo-900/20 shadow-md shadow-black/10">
        <div className="container mx-auto px-6 py-3">
          <div className="flex items-center justify-between">
            <motion.div 
              className="flex items-center"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <motion.div 
                className="h-8 w-8 rounded-lg bg-gradient-to-tr from-blue-600 to-indigo-600 mr-3 flex items-center justify-center text-white text-sm font-bold shadow-md shadow-indigo-900/20"
                whileHover={{ scale: 1.05, rotate: 5 }}
                whileTap={{ scale: 0.95 }}
              >
                CF
              </motion.div>
              <h1 className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-indigo-300 to-purple-400">
                London Residential Burglary Forecasting
              </h1>
            </motion.div>
            
            <motion.div 
              className="hidden md:flex items-center space-x-1"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <NavItem label="Dashboard" active />
              <NavItem label="Reports" />
              <NavItem label="Settings" />
            </motion.div>
            
            <motion.div 
              className="flex items-center space-x-4"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              {onOpenTutorial && (
                <motion.button 
                  className="px-3 py-1.5 rounded-lg text-sm bg-blue-800/30 text-blue-300 border border-blue-700/40 flex items-center"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={onOpenTutorial}
                >
                  <Video className="w-4 h-4 mr-1" />
                  Tutorial
                </motion.button>
              )}
              
              <motion.button 
                className="w-8 h-8 flex items-center justify-center rounded-full bg-gray-800/80 hover:bg-indigo-800/50 text-gray-300 transition-colors duration-200 border border-gray-700/50 shadow-md shadow-black/10"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <span className="text-xs">🔔</span>
              </motion.button>
              
              <motion.div 
                className="flex items-center"
                whileHover={{ scale: 1.05 }}
              >
                <div className="w-8 h-8 rounded-full bg-gradient-to-r from-indigo-500 to-blue-600 flex items-center justify-center text-white text-xs font-medium mr-2 shadow-md shadow-indigo-600/20">
                  PD
                </div>
                <span className="text-sm text-gray-300 hidden md:inline-block">Admin</span>
              </motion.div>
            </motion.div>
          </div>
        </div>
      </div>

      {/* Command bar */}
      <div className="bg-gray-900/90 backdrop-blur-sm py-2 border-b border-gray-800/50 shadow-sm">
        <div className="container mx-auto px-6 flex flex-wrap items-center justify-between">
          <div className="flex space-x-2">
            <TimeButton label="Last 30 days" active={dateRange[0] === 30} onClick={() => setDateRange([30])} />
            <TimeButton label="3 months" active={dateRange[0] === 90} onClick={() => setDateRange([90])} />
            <TimeButton label="6 months" active={dateRange[0] === 180} onClick={() => setDateRange([180])} />
            <TimeButton label="1 year" active={dateRange[0] === 365} onClick={() => setDateRange([365])} />
            <motion.button
              className={`btn-secondary text-xs py-1.5 px-3 bg-gray-800/80 text-gray-300 border ${
                showRangeSlider ? "border-indigo-600/50" : "border-gray-700/50"
              } rounded-lg ml-2 hidden md:flex items-center shadow-sm ${
                showRangeSlider ? "bg-indigo-900/20" : ""
              }`}
              whileHover={{ scale: 1.05, backgroundColor: "rgba(79, 70, 229, 0.2)" }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setShowRangeSlider(!showRangeSlider)}
            >
              <span className="mr-1">📅</span> Custom Range
            </motion.button>
          </div>
          
          <div className="flex items-center space-x-3 mt-0">
            <motion.div 
              className="relative"
              animate={{ 
                width: isSearchFocused ? '220px' : '160px',
                boxShadow: isSearchFocused ? '0 4px 12px rgba(79, 70, 229, 0.2)' : '0 2px 5px rgba(0, 0, 0, 0.2)'
              }}
              transition={{ duration: 0.3 }}
            >
              <input 
                type="text" 
                placeholder="Search areas or metrics..." 
                className="text-xs rounded-lg bg-gray-800/90 border border-gray-700/50 px-3 py-2 text-gray-300 focus:outline-none focus:ring-1 focus:ring-indigo-500/60 focus:border-indigo-500/50 w-full pr-8"
                onFocus={() => setIsSearchFocused(true)}
                onBlur={() => setIsSearchFocused(false)}
              />
              <motion.span 
                className="text-gray-400 absolute right-3 top-2 text-xs"
                animate={{ 
                  color: isSearchFocused ? '#818cf8' : '#9ca3af'
                }}
              >
                🔍
              </motion.span>
            </motion.div>
            
            <motion.button 
              className="w-8 h-8 flex items-center justify-center rounded-lg bg-gray-800/80 text-gray-300 hover:bg-indigo-800/30 transition-colors duration-200 border border-gray-700/50 shadow-sm"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span className="text-xs">⚙️</span>
            </motion.button>
          </div>
        </div>
        
        {/* Custom Range Slider */}
        {showRangeSlider && (
          <motion.div 
            className="container mx-auto px-6 pt-4"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="bg-gray-800/90 rounded-lg p-4 border border-gray-700/50 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-white">Custom Date Range</h3>
                <div className="bg-indigo-900/40 border border-indigo-700/30 px-3 py-1 rounded-md">
                  <span className="text-indigo-300 text-sm font-medium">{dateRange[0]} days</span>
                </div>
              </div>
              
              <Slider
                defaultValue={dateRange}
                max={365}
                min={1}
                step={1}
                value={dateRange}
                onValueChange={handleRangeChange}
                className="my-4"
              />
              
              <div className="flex justify-between text-xs text-gray-400">
                <span>1 day</span>
                <span>30 days</span>
                <span>90 days</span>
                <span>180 days</span>
                <span>365 days</span>
              </div>
              
              <div className="flex justify-end mt-4 space-x-2">
                <button 
                  className="px-3 py-1 text-xs text-gray-300 bg-gray-700/50 hover:bg-gray-700 rounded-md"
                  onClick={() => setShowRangeSlider(false)}
                >
                  Cancel
                </button>
                <button 
                  className="px-3 py-1 text-xs text-white bg-indigo-600 hover:bg-indigo-500 rounded-md"
                  onClick={() => setShowRangeSlider(false)}
                >
                  Apply
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </header>
  );
};

// Reusable button components
const NavItem = ({ label, active }: { label: string; active?: boolean }) => (
  <motion.button 
    className={`px-3 py-1.5 rounded-lg text-sm transition-colors duration-200 ${
      active 
        ? 'bg-indigo-800/30 text-indigo-300 border border-indigo-700/40' 
        : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/70'
    }`}
    whileHover={{ scale: 1.05 }}
    whileTap={{ scale: 0.95 }}
  >
    {label}
  </motion.button>
);

const TimeButton = ({ 
  label, 
  active, 
  onClick 
}: { 
  label: string; 
  active?: boolean;
  onClick?: () => void;
}) => (
  <motion.button 
    className={`
      btn-secondary text-xs py-1.5 px-3 rounded-lg flex items-center shadow-sm
      ${active 
        ? 'bg-gradient-to-r from-indigo-800/60 to-blue-800/40 text-indigo-300 border border-indigo-700/50' 
        : 'bg-gray-800/80 text-gray-300 border border-gray-700/50 hover:bg-gray-700/50'
      }
    `}
    whileHover={{ scale: 1.05 }}
    whileTap={{ scale: 0.95 }}
    onClick={onClick}
  >
    {label}
  </motion.button>
);

export default Header;
