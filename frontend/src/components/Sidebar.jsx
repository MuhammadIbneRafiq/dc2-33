import React from 'react';

const Sidebar = ({ activeView, setActiveView }) => {
  const navItems = [
    { id: 'dashboard', name: 'Dashboard' },
    { id: 'map', name: 'Map View' },
    { id: 'allocation', name: 'Police Allocation' },
    { id: 'analytics', name: 'Data Analytics' },
  ];

  // Crime types
  const crimeTypes = [
    { id: 'all', name: 'All Crimes' },
    { id: 'burglary', name: 'Burglary' },
    { id: 'theft', name: 'MV Theft' },
    { id: 'assault', name: 'Agg. Assault' },
    { id: 'robbery', name: 'Robbery' },
    { id: 'larceny', name: 'Larceny' },
  ];

  return (
    <div className="bg-gray-900 w-64 h-full border-r border-gray-800 flex flex-col overflow-y-auto">
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center">
          <div className="w-8 h-8 rounded-md bg-gray-700 flex items-center justify-center text-white mr-3">
            PD
          </div>
          <div>
            <h2 className="text-white font-semibold">Police Department</h2>
            <p className="text-gray-400 text-xs">Keeping our streets safe</p>
          </div>
        </div>
      </div>

      <div className="p-3 bg-red-900/20 border-b border-red-900/20 mx-3 mt-3 rounded-md">
        <p className="text-xs text-red-400">This data is for demonstration purposes and is not real.</p>
      </div>

      <div className="px-4 py-3">
        <h3 className="text-sm font-medium text-white mb-1">Filters</h3>
        <p className="text-xs text-gray-400 mb-4">Use the following to filter the dashboard elements to a specific police division, district, crime type, or time period.</p>
      </div>

      <div className="px-4 mb-4">
        <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">Police Division</h4>
        <select className="w-full rounded-sm bg-gray-800 border border-gray-700 px-2 py-1 text-sm text-gray-300 focus:outline-none focus:ring-1 focus:ring-blue-500/50">
          <option>All</option>
          <option>Central</option>
          <option>North</option>
          <option>South</option>
        </select>
      </div>

      <div className="px-4 mb-4">
        <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">Police District</h4>
        <select className="w-full rounded-sm bg-gray-800 border border-gray-700 px-2 py-1 text-sm text-gray-300 focus:outline-none focus:ring-1 focus:ring-blue-500/50">
          <option>All</option>
          <option>District 1</option>
          <option>District 2</option>
          <option>District 3</option>
        </select>
      </div>

      <div className="px-4 mb-4">
        <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">Crime Type</h4>
        <div className="space-y-2">
          {crimeTypes.map((type) => (
            <div key={type.id} className="flex items-center">
              <input 
                type="radio" 
                id={`crime-${type.id}`} 
                name="crimeType" 
                className="h-4 w-4 text-blue-500 focus:ring-blue-400 border-gray-700 bg-gray-800"
                defaultChecked={type.id === 'all'}
              />
              <label htmlFor={`crime-${type.id}`} className="ml-2 text-sm text-gray-300">
                {type.name}
              </label>
            </div>
          ))}
        </div>
      </div>

      <nav className="flex-1 pt-2">
        <div className="px-4 mb-2">
          <h3 className="text-xs font-medium text-gray-400 uppercase tracking-wider">Navigation</h3>
        </div>
        <ul>
          {navItems.map((item) => (
            <li key={item.id} className="mb-1 px-2">
              <button
                className={`flex items-center w-full px-2 py-2 rounded-md text-sm font-medium transition-colors duration-150 
                  ${activeView === item.id 
                    ? 'bg-blue-700/20 text-blue-400 border-l-2 border-blue-500' 
                    : 'text-gray-400 hover:text-gray-100 hover:bg-gray-800/50'}`}
                onClick={() => setActiveView(item.id)}
              >
                {item.name}
              </button>
            </li>
          ))}
        </ul>
      </nav>

      <div className="p-4 border-t border-gray-800 mt-auto">
        <div className="flex items-center text-xs">
          <div className="text-gray-400">
            <span className="block">Last update:</span>
            <span className="text-blue-400 font-semibold">18 seconds ago</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar; 