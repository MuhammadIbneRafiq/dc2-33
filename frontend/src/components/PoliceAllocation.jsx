import React from 'react';

// Mock data for police allocation
const allocationData = {
  totalOfficers: 45,
  deploymentMethod: 'K-means Clustering',
  coveragePercentage: 78,
  estimatedReduction: 42,
  optimizedAreas: [
    { lsoa: 'E01000001', officerCount: 3, reductionPercentage: 38 },
    { lsoa: 'E01032740', officerCount: 5, reductionPercentage: 45 },
    { lsoa: 'E01000005', officerCount: 4, reductionPercentage: 39 },
    { lsoa: 'E01032739', officerCount: 2, reductionPercentage: 32 },
  ],
};

const PoliceAllocation = () => {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h3 className="text-lg font-semibold text-dark mb-3">
        Optimal Police Force Allocation
      </h3>
      
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-blue-50 p-3 rounded-lg">
          <div className="text-sm text-gray-600">Total Officers</div>
          <div className="text-2xl font-bold text-blue-700">{allocationData.totalOfficers}</div>
        </div>
        
        <div className="bg-green-50 p-3 rounded-lg">
          <div className="text-sm text-gray-600">Coverage</div>
          <div className="text-2xl font-bold text-green-700">{allocationData.coveragePercentage}%</div>
        </div>
        
        <div className="bg-purple-50 p-3 rounded-lg">
          <div className="text-sm text-gray-600">Allocation Method</div>
          <div className="text-lg font-semibold text-purple-700">{allocationData.deploymentMethod}</div>
        </div>
        
        <div className="bg-amber-50 p-3 rounded-lg">
          <div className="text-sm text-gray-600">Est. Crime Reduction</div>
          <div className="text-2xl font-bold text-amber-700">{allocationData.estimatedReduction}%</div>
        </div>
      </div>
      
      <div className="bg-gray-50 p-3 rounded-lg">
        <h4 className="font-medium text-gray-700 mb-2">Area-specific Allocation</h4>
        
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="bg-gray-100">
                <th className="py-2 px-3 text-left rounded-l-lg">LSOA Code</th>
                <th className="py-2 px-3 text-center">Officers</th>
                <th className="py-2 px-3 text-center rounded-r-lg">Est. Reduction</th>
              </tr>
            </thead>
            <tbody>
              {allocationData.optimizedAreas.map((area, index) => (
                <tr key={area.lsoa} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="py-2 px-3">{area.lsoa}</td>
                  <td className="py-2 px-3 text-center">
                    <span className="inline-flex items-center justify-center h-6 w-6 rounded-full bg-blue-100 text-blue-800 font-medium">
                      {area.officerCount}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-center">
                    <span className="inline-block px-2 py-1 rounded-full bg-green-100 text-green-800 text-xs">
                      {area.reductionPercentage}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      <div className="mt-4 bg-blue-50 p-3 rounded-lg">
        <h4 className="font-medium text-blue-800 mb-1">How does it work?</h4>
        <p className="text-sm text-gray-700">
          The system uses K-means clustering to identify optimal police officer allocation based on historical burglary data. 
          Officers are strategically placed in areas with high burglary rates to maximize prevention and response efficiency.
        </p>
        <p className="text-sm text-gray-700 mt-2">
          The algorithm analyzes patterns of criminal activity and determines the optimal number and positioning of officers for maximum coverage.
        </p>
      </div>
    </div>
  );
};

export default PoliceAllocation; 