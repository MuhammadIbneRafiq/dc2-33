import React, { useState } from 'react';
import Header from './Header';
import Sidebar from './Sidebar';
import CrimeMap from './CrimeMap';
import PoliceAllocation from './PoliceAllocation';
import DashboardStats from './DashboardStats';
import DataAnalytics from './DataAnalytics';
import EmmieExplanation from './EmmieExplanation';
import PoliceChat from './PoliceChat';

const MapDashboard: React.FC = () => {
  const [activeView, setActiveView] = useState('map');
  const [showPoliceAllocation, setShowPoliceAllocation] = useState(false);
  const [policeData, setPoliceData] = useState<any[] | null>(null);
  const [selectedLSOA, setSelectedLSOA] = useState<string | null>(null);
  const [allocationMetrics, setAllocationMetrics] = useState<any | null>(null);

  const handleTogglePoliceAllocation = () => {
    setShowPoliceAllocation(!showPoliceAllocation);
  };

  const handlePoliceDataLoaded = (data: any[]) => {
    setPoliceData(data);
  };

  const handleSelectLSOA = (lsoa: string) => {
    setSelectedLSOA(lsoa);
  };

  const handleMetricsUpdate = (metrics: any) => {
    setAllocationMetrics(metrics);
  };

  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <Sidebar 
        activeView={activeView} 
        setActiveView={setActiveView} 
        showPoliceAllocation={showPoliceAllocation}
        onTogglePoliceAllocation={handleTogglePoliceAllocation}
        selectedLSOA={selectedLSOA}
      />

      {/* Main Content */}
      <div className="flex-1 ml-[280px]">
        <Header />
        
        <div className="container mx-auto p-6">
          {activeView === 'dashboard' && (
            <>
              <DashboardStats />
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="lg:col-span-1">
                  <CrimeMap 
                    showPoliceAllocation={showPoliceAllocation}
                    policeData={policeData}
                    onSelectLSOA={handleSelectLSOA}
                  />
                </div>
                <div className="lg:col-span-1">
                  <PoliceAllocation
                    showPoliceAllocation={showPoliceAllocation}
                    onTogglePoliceAllocation={handleTogglePoliceAllocation}
                    onPoliceDataLoaded={handlePoliceDataLoaded}
                    onMetricsUpdate={handleMetricsUpdate}
                  />
                </div>
              </div>
            </>
          )}
          
          {activeView === 'map' && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <CrimeMap 
                  showPoliceAllocation={showPoliceAllocation}
                  policeData={policeData}
                  onSelectLSOA={handleSelectLSOA}
                />
              </div>
              <div className="lg:col-span-1">
                <PoliceAllocation
                  showPoliceAllocation={showPoliceAllocation}
                  onTogglePoliceAllocation={handleTogglePoliceAllocation}
                  onPoliceDataLoaded={handlePoliceDataLoaded}
                  onMetricsUpdate={handleMetricsUpdate}
                />
              </div>
            </div>
          )}
          
          {activeView === 'allocation' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="lg:col-span-1">
                <PoliceAllocation
                  showPoliceAllocation={showPoliceAllocation}
                  onTogglePoliceAllocation={handleTogglePoliceAllocation}
                  onPoliceDataLoaded={handlePoliceDataLoaded}
                  onMetricsUpdate={handleMetricsUpdate}
                />
              </div>
              <div className="lg:col-span-1">
                <CrimeMap 
                  showPoliceAllocation={showPoliceAllocation}
                  policeData={policeData}
                  onSelectLSOA={handleSelectLSOA}
                />
              </div>
            </div>
          )}
          
          {activeView === 'analytics' && <DataAnalytics />}

          {activeView === 'emmie' && <EmmieExplanation />}
        </div>
      </div>

      {/* Police Chat Widget */}
      <PoliceChat 
        selectedLSOA={selectedLSOA}
        selectedAllocation={allocationMetrics}
      />
    </div>
  );
};

export default MapDashboard;
