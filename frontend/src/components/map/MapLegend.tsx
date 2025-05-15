import React from 'react';

interface MapLegendItem {
  color: string;
  label: string;
}

interface MapLegendProps {
  position?: 'topright' | 'topleft' | 'bottomright' | 'bottomleft';
  showPoliceAllocation?: boolean;
}

const MapLegend: React.FC<MapLegendProps> = ({ 
  position = 'bottomright',
  showPoliceAllocation
}) => {
  return (
    <div className={`absolute ${position === 'bottomright' ? 'bottom-5 right-5' : 
                       position === 'bottomleft' ? 'bottom-5 left-5' :
                       position === 'topleft' ? 'top-5 left-5' : 'top-5 right-5'} 
                     bg-gray-900/80 backdrop-blur-sm text-white p-3 rounded-lg shadow-lg z-[1000] max-w-[250px]`}>
      <h3 className="text-sm font-bold mb-2 border-b border-gray-700 pb-1">Map Legend</h3>
      
      <div className="mb-3">
        <h4 className="text-xs font-semibold mb-1">Risk Levels</h4>
        <div className="space-y-1">
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-red-500 mr-2"></div>
            <span className="text-xs">High Risk</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-yellow-500 mr-2"></div>
            <span className="text-xs">Medium Risk</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
            <span className="text-xs">Low Risk</span>
          </div>
        </div>
      </div>
      
      <div className="mb-2">
        <h4 className="text-xs font-semibold mb-1">Police Units</h4>
        <div className="space-y-1">
          <div className="flex items-center">
            <div className="mr-2 bg-blue-500 text-xs w-4 h-4 rounded-full flex items-center justify-center text-white">🚓</div>
            <span className="text-xs">Vehicle Patrol</span>
          </div>
          <div className="flex items-center">
            <div className="mr-2 bg-blue-800 text-xs w-4 h-4 rounded-full flex items-center justify-center text-white">👮</div>
            <span className="text-xs">Foot Patrol</span>
          </div>
        </div>
      </div>
      
      {/* Credits */}
      <div className="mt-3 pt-2 border-t border-gray-700 text-gray-400 text-[10px]">
        Based on CPTED policing strategies
      </div>
    </div>
  );
};

export default MapLegend;