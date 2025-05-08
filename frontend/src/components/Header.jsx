import React from 'react';

const Header = () => {
  return (
    <header className="bg-dark text-white shadow-md">
      <div className="container mx-auto px-4 py-3">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center mb-4 md:mb-0">
            <div className="text-2xl font-bold mr-4">
              <span className="text-blue-400">London</span> Crime Analytics
            </div>
            <div className="text-sm bg-blue-500 px-2 py-1 rounded-md">
              Burglary Analysis
            </div>
          </div>
          
          <div className="flex space-x-4">
            <div className="flex items-center space-x-2">
              <span className="text-sm">Data updated: Feb 2025</span>
              <div className="h-2 w-2 bg-green-400 rounded-full"></div>
            </div>
            
            <div className="hidden md:flex items-center space-x-3">
              <div className="flex items-center text-sm">
                <span className="mr-2">Crime types:</span>
                <span className="px-2 py-1 bg-secondary text-white rounded-md text-xs">Burglary</span>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap justify-between items-center">
          <div className="flex space-x-4 mb-2 md:mb-0">
            <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm">
              Last 30 days
            </button>
            <button className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm">
              3 months
            </button>
            <button className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm">
              6 months
            </button>
            <button className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm">
              1 year
            </button>
          </div>
          
          <div className="flex space-x-2 items-center">
            <div className="text-sm">Crime count: <span className="font-bold">5,431</span></div>
            <div className="h-4 w-px bg-gray-600"></div>
            <div className="text-sm">LSOA count: <span className="font-bold">983</span></div>
          </div>
        </div>
      </div>

      {/* Images section */}
      <div className="hidden md:block bg-gray-900 py-3">
        <div className="container mx-auto flex justify-between items-center px-4">
          <div className="flex items-center">
            <div className="h-10 w-10 flex items-center justify-center bg-white rounded-full mr-3">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 text-blue-600">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75 11.25 15 15 9.75m-3-7.036A11.959 11.959 0 0 1 3.598 6 11.99 11.99 0 0 0 3 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285Z" />
              </svg>
            </div>
            <div>
              <p className="text-xs text-gray-400">This data visualization is for demonstration purposes</p>
              <p className="text-xs text-gray-400">Based on London Metropolitan Police data</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="hidden lg:flex space-x-4">
              {/* Police Dashboard icon */}
              <div className="flex items-center p-1 bg-gray-800 rounded">
                <svg className="w-8 h-8 text-blue-400" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2" />
                  <circle cx="8" cy="8" r="2" fill="currentColor" />
                  <circle cx="16" cy="8" r="2" fill="currentColor" />
                  <circle cx="8" cy="16" r="2" fill="currentColor" />
                  <circle cx="16" cy="16" r="2" fill="currentColor" />
                </svg>
                
                {/* Police Network icon */}
                <svg className="w-8 h-8 ml-2 text-green-400" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="12" cy="12" r="3" fill="currentColor" />
                  <circle cx="19" cy="12" r="2" fill="currentColor" />
                  <circle cx="5" cy="12" r="2" fill="currentColor" />
                  <circle cx="12" cy="19" r="2" fill="currentColor" />
                  <circle cx="12" cy="5" r="2" fill="currentColor" />
                  <line x1="12" y1="9" x2="12" y2="7" stroke="currentColor" strokeWidth="1" />
                  <line x1="12" y1="17" x2="12" y2="15" stroke="currentColor" strokeWidth="1" />
                  <line x1="9" y1="12" x2="7" y2="12" stroke="currentColor" strokeWidth="1" />
                  <line x1="17" y1="12" x2="15" y2="12" stroke="currentColor" strokeWidth="1" />
                </svg>
              </div>
              <div>
                <div className="text-xs text-right text-gray-400">Social Network Analysis</div>
                <div className="text-xs text-gray-400">Last updated: 12 Feb 2025</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header; 