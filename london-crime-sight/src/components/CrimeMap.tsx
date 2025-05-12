
import React from 'react';
import 'mapbox-gl/dist/mapbox-gl.css';
import { useMapbox } from '../hooks/useMapbox';
import MapboxTokenInput from './map/MapboxTokenInput';
import MapHeader from './map/MapHeader';
import MapLegend from './map/MapLegend';

interface MapProps {
  showPoliceAllocation: boolean;
  policeData: any[] | null;
  onSelectLSOA: (lsoa: string) => void;
}

const CrimeMap: React.FC<MapProps> = ({ showPoliceAllocation, policeData, onSelectLSOA }) => {
  const {
    mapContainer,
    tokenInput,
    setTokenInput,
    tokenSubmitted,
    handleTokenSubmit,
    needsToken
  } = useMapbox(onSelectLSOA, showPoliceAllocation, policeData);

  if (needsToken) {
    return (
      <MapboxTokenInput 
        tokenInput={tokenInput}
        setTokenInput={setTokenInput}
        handleTokenSubmit={handleTokenSubmit}
      />
    );
  }

  return (
    <div className="dashboard-card">
      <MapHeader />
      
      <div className="relative h-[600px]">
        <div ref={mapContainer} className="absolute inset-0" />
        <MapLegend showPoliceAllocation={showPoliceAllocation} />
      </div>
    </div>
  );
};

export default CrimeMap;
