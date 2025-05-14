import React from 'react';

interface MapLegendItem {
  color: string;
  label: string;
}

interface MapLegendProps {
  title?: string;
  items?: MapLegendItem[];
  showPoliceAllocation?: boolean;
  showPredictions?: boolean;
  predictionItems?: MapLegendItem[];
}

const MapLegend: React.FC<MapLegendProps> = ({ 
  title = "Legend", 
  items = [], 
  showPoliceAllocation = false,
  showPredictions = false,
  predictionItems = []
}) => {
  return (
    <>
      {/* Combined legend at the bottom of the map */}
      <div className="absolute bottom-4 left-0 right-0 mx-auto px-4">
        <div className="bg-gray-900/80 backdrop-blur-sm p-3 rounded-lg border border-gray-700/50 shadow-lg z-[1000] max-w-3xl mx-auto">
          <div className="flex flex-wrap justify-center gap-6">
            {/* Main LSOA Risk Legend */}
            <div>
              <div className="text-white text-sm font-semibold mb-2 text-center">{title}</div>
              {items.length > 0 && (
                <div className="flex flex-wrap gap-3">
                  {items.map((item, index) => (
                    <div key={index} className="flex items-center">
                      <div 
                        className="w-3 h-3 mr-1 rounded" 
                        style={{ backgroundColor: item.color }}
                      ></div>
                      <span className="text-xs text-gray-300">{item.label}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            {/* Police Allocation Legend */}
            {showPoliceAllocation && (
              <div>
                <div className="text-white text-sm font-semibold mb-2 text-center">Police Units</div>
                <div className="flex gap-3">
                  <div className="flex items-center">
                    <div className="w-3 h-3 rounded-full bg-blue-900 mr-1"></div>
                    <span className="text-xs text-gray-300">Officer</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-3 h-3 rounded-full bg-blue-600 mr-1"></div>
                    <span className="text-xs text-gray-300">Vehicle</span>
                  </div>
                </div>
              </div>
            )}
            
            {/* Prediction Legend */}
            {showPredictions && (
              <div>
                <div className="text-white text-sm font-semibold mb-2 text-center">Predictions</div>
                <div className="flex flex-wrap gap-3">
                  {predictionItems.map((item, index) => (
                    <div key={index} className="flex items-center">
                      <div 
                        className="w-3 h-3 mr-1 rounded-full" 
                        style={{ backgroundColor: item.color }}
                      ></div>
                      <span className="text-xs text-gray-300">{item.label}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          <div className="text-xs text-gray-400 mt-2 text-center">Data updated: May 2023</div>
        </div>
      </div>
    </>
  );
};

export default MapLegend;
