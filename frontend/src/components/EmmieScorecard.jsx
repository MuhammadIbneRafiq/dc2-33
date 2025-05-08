import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { getEmmieScores, getLsoaWellbeingData } from '../api/backendService';

const EmmieScorecard = ({ selectedLSOA }) => {
  const [expandedSection, setExpandedSection] = useState('intervention');
  const [emmieFramework, setEmmieFramework] = useState(null);
  const [housingData, setHousingData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Load EMMIE framework data
  useEffect(() => {
    const loadEmmieData = async () => {
      try {
        setIsLoading(true);
        const emmieData = await getEmmieScores();
        
        // Ensure each section has at least an empty items array to prevent mapping errors
        if (emmieData) {
          Object.keys(emmieData).forEach(key => {
            if (!emmieData[key]) {
              emmieData[key] = { title: key.charAt(0).toUpperCase() + key.slice(1), items: [] };
            } else if (!emmieData[key].items) {
              emmieData[key].items = [];
            }
          });
        }
        
        setEmmieFramework(emmieData);
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
  const getLocalRiskLevel = (crimeScore) => {
    if (crimeScore > 0.8) return 2; // High
    if (crimeScore > 0.4) return 1; // Medium
    return 0; // Low
  };

  const renderStars = (score) => {
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

  const renderRiskLevel = (level) => {
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
      <span className={colors[level]}>
        {labels[level]}
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
          <p className="text-gray-400">EMMIE framework data unavailable.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden mb-4">
      <h3 className="text-white text-base font-semibold p-3 border-b border-gray-700 flex items-center">
        <span className="text-orange-400 mr-2">🎯</span>
        EMMIE Scorecard Widget
      </h3>
      
      {/* Tabs */}
      <div className="flex border-b border-gray-700">
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
              emmieFramework[expandedSection].items.map((item, index) => (
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
                <td colSpan="2" className="py-4 text-center text-gray-400">
                  No items available for this section
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      
      <h3 className="text-white text-base font-semibold p-3 border-t border-b border-gray-700 flex items-center">
        <span className="text-green-400 mr-2">📊</span>
        Live Patrol Feed
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
                  <div className="bg-gray-800 p-2 rounded border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">Social Housing:</div>
                    <div className="text-white">{housingData.socialHousingPercent}%</div>
                  </div>
                  <div className="bg-gray-800 p-2 rounded border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">Crime Rank:</div>
                    <div className="text-white">{housingData.crimeRank}/32</div>
                  </div>
                </div>
                
                <div className="bg-gray-800 p-2 rounded border border-gray-700 mb-3">
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="text-xs text-gray-400 mb-1">Wellbeing Score:</div>
                      <div className="text-white">{housingData.wellbeingScore.toFixed(1)}</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400 mb-1">Risk Level:</div>
                      <div className="text-white">{renderRiskLevel(housingData.riskLevel)}</div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-blue-900/20 text-blue-400 border border-blue-900/30 rounded p-2 text-xs">
                  EMMIE scores indicate intervention effectiveness. The framework considers Effect, Mechanism, 
                  Moderators, Implementation and Economic impact for crime reduction strategies.
                </div>
              </motion.div>
            ) : (
              <div className="text-center py-4 text-gray-400">
                <p>Loading wellbeing data for this area...</p>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-4 text-gray-400">
            <p>Select an area on the map to view EMMIE data</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default EmmieScorecard; 