
import React from 'react';

interface MapboxTokenInputProps {
  tokenInput: string;
  setTokenInput: (token: string) => void;
  handleTokenSubmit: (e: React.FormEvent) => void;
}

const MapboxTokenInput: React.FC<MapboxTokenInputProps> = ({ 
  tokenInput, 
  setTokenInput, 
  handleTokenSubmit 
}) => {
  return (
    <div className="dashboard-card h-[600px] flex flex-col items-center justify-center p-6">
      <div className="w-full max-w-md">
        <h3 className="text-xl font-bold text-white mb-4 text-center">Mapbox API Key Required</h3>
        <p className="text-gray-300 mb-6 text-center">
          To use the map functionality, please enter your Mapbox public token.
        </p>
        
        <form onSubmit={handleTokenSubmit} className="space-y-4">
          <div>
            <label htmlFor="mapbox-token" className="block text-sm font-medium text-gray-300 mb-1">
              Mapbox Public Token
            </label>
            <input 
              id="mapbox-token"
              type="text" 
              value={tokenInput}
              onChange={(e) => setTokenInput(e.target.value)}
              placeholder="pk.eyJ1Ijoi..."
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-md text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>
          
          <div className="bg-blue-900/20 p-3 rounded-md border border-blue-800/30 text-sm text-blue-300">
            <p>You can get your Mapbox token by signing up at <a href="https://mapbox.com/" target="_blank" rel="noopener noreferrer" className="text-blue-400 underline">mapbox.com</a> and visiting the Access Tokens section.</p>
          </div>
          
          <button 
            type="submit"
            className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-md transition-colors duration-200"
          >
            Apply Token
          </button>
        </form>
      </div>
    </div>
  );
};

export default MapboxTokenInput;
