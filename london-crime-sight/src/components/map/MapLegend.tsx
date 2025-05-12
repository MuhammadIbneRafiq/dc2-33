
import React from 'react';

interface MapLegendProps {
  showPoliceAllocation: boolean;
}

const MapLegend: React.FC<MapLegendProps> = ({ showPoliceAllocation }) => {
  return (
    <>
      {showPoliceAllocation && (
        <div className="absolute top-4 left-4 bg-gray-900/80 backdrop-blur-sm p-3 rounded-lg border border-gray-700/50 shadow-lg">
          <div className="text-white text-sm font-semibold mb-2">Police Units</div>
          <div className="flex space-x-4">
            <div className="flex items-center">
              <div className="w-3 h-3 rounded-full bg-blue-500 mr-1"></div>
              <span className="text-xs text-gray-300">Vehicle</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 rounded-full bg-green-500 mr-1"></div>
              <span className="text-xs text-gray-300">Foot</span>
            </div>
          </div>
        </div>
      )}
      
      <div className="absolute bottom-4 right-4 bg-gray-900/80 backdrop-blur-sm p-3 rounded-lg border border-gray-700/50 shadow-lg">
        <div className="text-xs text-gray-400 mb-1">Click on a hotspot to view details</div>
        <div className="text-xs text-indigo-300">Data updated: May 2025</div>
      </div>
    </>
  );
};

export default MapLegend;
