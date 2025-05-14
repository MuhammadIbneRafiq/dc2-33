import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { getEmmieScores, getLsoaWellbeingData } from '../api/backendService';

interface EmmieScoreCardProps {
  selectedLSOA: string | null;
}

const EmmieScorecard: React.FC<EmmieScoreCardProps> = ({ selectedLSOA }) => {
  const [expandedSection, setExpandedSection] = useState('intervention');
  const [emmieFramework, setEmmieFramework] = useState<any>(null);
  const [housingData, setHousingData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Updated burglary factors based on criminology research
  const burglaryFactors = {
    environmental: {
      title: 'Environmental Factors',
      items: [
        { name: 'Lack of natural surveillance', score: 4, description: 'Limited visibility from street or neighbors', effectiveness: 4.2 },
        { name: 'Poor street lighting', score: 3, description: 'Dark areas create opportunities', effectiveness: 3.8 },
        { name: 'Secluded entry points', score: 4, description: 'Hidden access points to properties', effectiveness: 4.3 },
        { name: 'Vacant properties', score: 5, description: 'Unoccupied homes are prime targets', effectiveness: 4.7 }
      ]
    },
    socioeconomic: {
      title: 'Socioeconomic Factors',
      items: [
        { name: 'Income inequality', score: 4, description: 'Areas with high wealth disparity', effectiveness: 4.1 },
        { name: 'Housing density', score: 3, description: 'High-density housing areas', effectiveness: 3.5 },
        { name: 'Transient population', score: 4, description: 'Areas with short-term residents', effectiveness: 3.9 },
        { name: 'Unemployment rate', score: 5, description: 'Higher joblessness correlates with crime', effectiveness: 4.5 }
      ]
    },
    opportunity: {
      title: 'Opportunity Factors',
      items: [
        { name: 'Weak security measures', score: 5, description: 'Inadequate locks, alarms, cameras', effectiveness: 4.8 },
        { name: 'Predictable occupancy', score: 4, description: 'Obvious patterns when homes are empty', effectiveness: 4.2 },
        { name: 'Valuable possessions', score: 3, description: 'Visible high-value items', effectiveness: 3.7 },
        { name: 'Easy escape routes', score: 4, description: 'Multiple pathways to flee', effectiveness: 4.0 }
      ]
    },
    temporal: {
      title: 'Temporal Patterns',
      items: [
        { name: 'Winter darkness', score: 4, description: 'Early darkness in winter months', effectiveness: 4.1 },
        { name: 'Holiday seasons', score: 5, description: 'Christmas and summer holiday periods', effectiveness: 4.6 },
        { name: 'Weekend patterns', score: 3, description: 'Different patterns on weekends', effectiveness: 3.4 },
        { name: 'Evening hours', score: 4, description: 'Peak times between 6pm-12am', effectiveness: 4.3 }
      ]
    },
    cpted: {
      title: 'CPTED Principles',
      items: [
        { name: 'Territorial reinforcement', score: 4, description: 'Clear boundaries between public/private', effectiveness: 4.1 },
        { name: 'Access control', score: 5, description: 'Limiting entry/exit points', effectiveness: 4.7 },
        { name: 'Activity support', score: 3, description: 'Encouraging legitimate space use', effectiveness: 3.6 },
        { name: 'Maintenance', score: 4, description: 'Upkeep of properties and surroundings', effectiveness: 4.2 }
      ]
    }
  };

  // Load EMMIE framework data
  useEffect(() => {
    const loadEmmieData = async () => {
      try {
        setIsLoading(true);
        
        // Use updated burglary factors instead of API data
        setEmmieFramework(burglaryFactors);
        
        setIsLoading(false);
      } catch (error) {
        console.error("Error loading EMMIE data:", error);
        setError("Failed to load intervention data");
        setIsLoading(false);
      }
    };
    
    loadEmmieData();
  }, []);

  // Load wellbeing data when LSOA changes
  useEffect(() => {
    const loadWellbeingData = async () => {
      if (!selectedLSOA) {
        setHousingData(null);
        return;
      }
      
      try {
        setIsLoading(true);
        const wellbeingData = await getLsoaWellbeingData(selectedLSOA);
        
        if (wellbeingData) {
          // Process wellbeing data into the format we need
          setHousingData({
            imdScore: Math.round(wellbeingData.imd_score * 10) / 10,
            housingDensity: Math.round(wellbeingData.housing_score),
            socialHousingPercent: Math.round(wellbeingData.income_score * 100),
            crimeRank: Math.round(wellbeingData.crime_score * 30) + 1,
            wellbeingScore: (10 - wellbeingData.imd_score) / 10, // Invert so higher is better
            riskLevel: getLocalRiskLevel(wellbeingData.crime_score)
          });
        } else {
          setHousingData(null);
        }
        
        setIsLoading(false);
      } catch (error) {
        console.error(`Error loading wellbeing data for LSOA ${selectedLSOA}:`, error);
        setError(`Failed to load data for ${selectedLSOA}`);
        setIsLoading(false);
      }
    };
    
    loadWellbeingData();
  }, [selectedLSOA]);

  // Helper function to determine risk level
  const getLocalRiskLevel = (crimeScore: number) => {
    if (crimeScore > 0.8) return 2; // High
    if (crimeScore > 0.4) return 1; // Medium
    return 0; // Low
  };

  const renderStars = (score: number) => {
    const stars = [];
    const maxStars = 5;
    
    for (let i = 1; i <= maxStars; i++) {
      stars.push(
        <span 
          key={i} 
          className={`text-lg ${i <= score ? 'text-blue-400' : 'text-gray-600'}`}
        >
          ★
        </span>
      );
    }
    
    return stars;
  };

  const renderRiskLevel = (level: number) => {
    const colors = {
      0: 'text-green-400',
      1: 'text-yellow-400',
      2: 'text-red-400'
    };
    
    const labels = {
      0: 'Low',
      1: 'Medium',
      2: 'High'
    };
    
    return (
      <span className={colors[level as keyof typeof colors]}>
        {labels[level as keyof typeof labels]}
      </span>
    );
  };

  if (isLoading) {
    return (
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden mb-4 p-4">
        <div className="flex justify-center items-center py-8">
          <div className="w-6 h-6 border-2 border-blue-500 border-t-blue-100 rounded-full animate-spin"></div>
          <span className="ml-2 text-gray-400">Loading data...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden mb-4 p-4">
        <div className="text-center py-6">
          <div className="text-red-500 text-xl mb-2">⚠️</div>
          <p className="text-gray-300">{error}</p>
        </div>
      </div>
    );
  }

  if (!emmieFramework) {
    return (
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden mb-4 p-4">
        <div className="text-center py-6">
          <p className="text-gray-400">Burglary factors data unavailable.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden mb-4">
      <h3 className="text-white text-base font-semibold p-3 border-b border-gray-700 flex items-center">
        <span className="text-orange-400 mr-2">🎯</span>
        Burglary Risk Factors
      </h3>
      
      {/* Tabs */}
      <div className="flex border-b border-gray-700 overflow-x-auto scrollbar-thin">
        {emmieFramework && Object.keys(emmieFramework).map(key => (
          <button
            key={key}
            className={`py-2 px-4 text-sm font-medium transition-colors ${
              expandedSection === key 
                ? 'text-blue-400 border-b-2 border-blue-400' 
                : 'text-gray-400 hover:text-gray-300'
            }`}
            onClick={() => setExpandedSection(key)}
          >
            {emmieFramework[key]?.title || key.charAt(0).toUpperCase() + key.slice(1)}
          </button>
        ))}
      </div>
      
      {/* EMMIE Table */}
      <div className="p-2 bg-gray-900">
        <table className="w-full">
          <tbody>
            {emmieFramework && emmieFramework[expandedSection] && emmieFramework[expandedSection].items && 
             emmieFramework[expandedSection].items.length > 0 ? (
              emmieFramework[expandedSection].items.map((item: any, index: number) => (
                <motion.tr 
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="border-b border-gray-800"
                >
                  <td className="py-2 px-3 text-sm text-white">
                    <div className="flex flex-col">
                      <span>{item.name}</span>
                      {item.description && (
                        <span className="text-xs text-gray-400">{item.description}</span>
                      )}
                    </div>
                  </td>
                  <td className="py-2 px-3 text-right">
                    {renderStars(item.score)}
                  </td>
                </motion.tr>
              ))
            ) : (
              <tr>
                <td colSpan={2} className="py-4 text-center text-gray-400">
                  No items available for this section
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      
      <h3 className="text-white text-base font-semibold p-3 border-t border-b border-gray-700 flex items-center">
        <span className="text-green-400 mr-2">📊</span>
        Area Analysis
      </h3>
      
      <div className="p-3 bg-gray-900 text-sm">
        {selectedLSOA ? (
          <div>
            <div className="mb-3">
              <div className="text-gray-400 mb-1">Selected Area:</div>
              <div className="font-mono text-white">{selectedLSOA}</div>
            </div>
            
            {housingData ? (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5 }}
              >
                <div className="grid grid-cols-2 gap-2 mb-3">
                  <div className="bg-gray-800 p-2 rounded border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">IMD Score:</div>
                    <div className="text-white">{housingData.imdScore}/10</div>
                  </div>
                  <div className="bg-gray-800 p-2 rounded border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">Housing Density:</div>
                    <div className="text-white">{housingData.housingDensity}/ha</div>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2 mb-3">
                  <div className="bg-gray-800 p-2 rounded border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">Social Housing:</div>
                    <div className="text-white">{housingData.socialHousingPercent}%</div>
                  </div>
                  <div className="bg-gray-800 p-2 rounded border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">Crime Rank:</div>
                    <div className="text-white">{housingData.crimeRank}/30</div>
                  </div>
                </div>
                <div className="bg-gray-800 p-2 rounded border border-gray-700 mb-2">
                  <div className="text-xs text-gray-400 mb-1">Area Risk Level:</div>
                  <div className="text-white font-medium">
                    {renderRiskLevel(housingData.riskLevel)}
                  </div>
                </div>
                <div className="bg-gray-800 p-2 rounded border border-gray-700">
                  <div className="text-xs text-gray-400 mb-1">Recommended Interventions:</div>
                  <ul className="text-xs text-white mt-1 space-y-1">
                    {/* Show different recommendations based on risk level */}
                    {housingData.riskLevel === 2 ? (
                      <>
                        <li className="flex items-center"><span className="text-blue-400 mr-1">•</span> Targeted police patrols</li>
                        <li className="flex items-center"><span className="text-blue-400 mr-1">•</span> Enhanced CCTV coverage</li>
                        <li className="flex items-center"><span className="text-blue-400 mr-1">•</span> Community watch programs</li>
                      </>
                    ) : housingData.riskLevel === 1 ? (
                      <>
                        <li className="flex items-center"><span className="text-blue-400 mr-1">•</span> Improved street lighting</li>
                        <li className="flex items-center"><span className="text-blue-400 mr-1">•</span> Property marking schemes</li>
                        <li className="flex items-center"><span className="text-blue-400 mr-1">•</span> Security assessments</li>
                      </>
                    ) : (
                      <>
                        <li className="flex items-center"><span className="text-blue-400 mr-1">•</span> Maintain environmental design</li>
                        <li className="flex items-center"><span className="text-blue-400 mr-1">•</span> Community engagement</li>
                        <li className="flex items-center"><span className="text-blue-400 mr-1">•</span> Preventative education</li>
                      </>
                    )}
                  </ul>
                </div>
              </motion.div>
            ) : (
              <div className="text-center py-4 text-gray-400">
                Loading area data...
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-4 text-gray-400">
            <p>Select an area on the map to view details</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default EmmieScorecard;

