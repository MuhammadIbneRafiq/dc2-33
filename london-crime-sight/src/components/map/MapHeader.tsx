
import React from 'react';

const MapHeader: React.FC = () => {
  return (
    <div className="flex items-center justify-between card-header px-4 py-3">
      <h3 className="text-white text-lg font-semibold flex items-center">
        <span className="text-blue-400 mr-2">🗺️</span>
        London Residential Burglary Risk Map
      </h3>
      
      <div className="flex items-center space-x-2">
        <div className="flex items-center text-sm text-gray-300 bg-gray-800/80 px-3 py-1 rounded-lg border border-gray-700/50">
          <div className="w-3 h-3 rounded-full bg-red-500 mr-1"></div>
          <span>High Risk</span>
        </div>
        
        <div className="flex items-center text-sm text-gray-300 bg-gray-800/80 px-3 py-1 rounded-lg border border-gray-700/50">
          <div className="w-3 h-3 rounded-full bg-yellow-500 mr-1"></div>
          <span>Medium Risk</span>
        </div>
      </div>
    </div>
  );
};

export default MapHeader;
