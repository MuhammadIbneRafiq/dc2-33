import React from 'react';

const Header = () => {
  return (
    <header className="navbar">
      <div className="container mx-auto px-4 py-2">
        <div className="flex justify-between items-center">
          <div className="flex items-center">
            <div className="text-xl font-bold flex items-center">
              <span className="text-blue-500 mr-2">CF</span>
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">CRIME FORECASTING</span>
            </div>
          </div>
          
          <div className="flex space-x-4 items-center">
            <div className="flex items-center space-x-2 px-3 py-1.5 bg-gray-800 rounded-md">
              <span className="text-gray-400 text-sm">Last updated:</span> 
              <span className="font-mono text-blue-400 text-sm font-semibold">Feb 2025</span>
            </div>
            
            <div className="flex items-center space-x-3">
              <button className="btn btn-secondary text-xs py-1 px-2 flex items-center">
                Share
              </button>
              
              <div className="p-1.5 rounded-full bg-gray-800 hover:bg-gray-700 cursor-pointer transition-colors">
                <span className="text-gray-400 text-xs">?</span>
              </div>
              
              <div className="p-1.5 rounded-full bg-gray-800 hover:bg-gray-700 cursor-pointer transition-colors">
                <span className="text-gray-400 text-xs">⚙️</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Command bar */}
      <div className="bg-gray-900 py-1 border-y border-gray-800">
        <div className="container mx-auto px-4 flex flex-wrap items-center justify-between">
          <div className="flex space-x-1">
            <button className="btn-secondary text-xs py-1 px-2 bg-blue-600/30 text-blue-400 border border-blue-600/40">
              Last 30 days
            </button>
            <button className="btn-secondary text-xs py-1 px-2 bg-gray-800 text-gray-300 border border-gray-700">
              3 months
            </button>
            <button className="btn-secondary text-xs py-1 px-2 bg-gray-800 text-gray-300 border border-gray-700">
              6 months
            </button>
            <button className="btn-secondary text-xs py-1 px-2 bg-gray-800 text-gray-300 border border-gray-700">
              1 year
            </button>
          </div>
          
          <div className="flex items-center space-x-3 mt-0">
            <div className="relative">
              <input type="text" placeholder="Search..." className="text-xs rounded-md bg-gray-800 border border-gray-700 px-2 py-1 text-gray-300 focus:outline-none focus:ring-1 focus:ring-blue-500/50 w-36" />
              <span className="text-gray-400 absolute right-2 top-1.5 text-xs">🔍</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header; 