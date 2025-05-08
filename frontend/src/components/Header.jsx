import React from 'react';

const Header = () => {
  return (
    <header className="navbar">
      <div className="container mx-auto px-4 py-2">
        <div className="flex justify-between items-center">
          <div className="flex items-center">
            <div className="text-xl font-bold flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6 text-blue-500 mr-2">
                <path fillRule="evenodd" d="M12 1.5a.75.75 0 0 1 .75.75V4.5a.75.75 0 0 1-1.5 0V2.25A.75.75 0 0 1 12 1.5ZM5.636 4.136a.75.75 0 0 1 1.06 0l1.592 1.591a.75.75 0 0 1-1.061 1.06l-1.591-1.59a.75.75 0 0 1 0-1.061Zm12.728 0a.75.75 0 0 1 0 1.06l-1.591 1.592a.75.75 0 0 1-1.06-1.061l1.59-1.591a.75.75 0 0 1 1.061 0Zm-6.816 4.496a.75.75 0 0 1 .82.311l5.228 7.917a.75.75 0 0 1-.153.976l-2.5 1.888a.75.75 0 0 1-1.06-.122l-4.5-5.25a.75.75 0 0 1 .152-1.096l1.5-1.123a.75.75 0 0 1 .51-.102Z" clipRule="evenodd" />
              </svg>
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
                <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 mr-1" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M15 8a3 3 0 10-2.977-2.63l-4.94 2.47a3 3 0 100 4.319l4.94 2.47a3 3 0 10.895-1.789l-4.94-2.47a3.027 3.027 0 000-.74l4.94-2.47C13.456 7.68 14.19 8 15 8z" />
                </svg>
                Share
              </button>
              
              <div className="p-1.5 rounded-full bg-gray-800 hover:bg-gray-700 cursor-pointer transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-gray-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                </svg>
              </div>
              
              <div className="p-1.5 rounded-full bg-gray-800 hover:bg-gray-700 cursor-pointer transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-gray-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
                </svg>
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
              <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3 text-gray-400 absolute right-2 top-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header; 