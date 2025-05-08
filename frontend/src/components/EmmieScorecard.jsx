import React, { useState } from 'react';
import { motion } from 'framer-motion';

const EmmieScorecard = ({ selectedLSOA }) => {
  const [expandedSection, setExpandedSection] = useState('intervention');

  // This is mock data - in a real application, this would come from the backend based on the CSV data
  const emmieFramework = {
    intervention: {
      title: 'Intervention',
      items: [
        { name: 'Deploy CCTV/Varifi', score: 4 },
        { name: 'Foot patrol', score: 4 },
      ]
    },
    prevention: {
      title: 'Prevention',
      items: [
        { name: 'Target hardening', score: 5 },
        { name: 'Environmental design', score: 4 },
      ]
    },
    diversion: {
      title: 'Diversion',
      items: [
        { name: 'Community engagement', score: 3 },
        { name: 'Youth programs', score: 3 },
      ]
    }
  };

  // Mock housing and wellbeing data for the selected LSOA
  // In a real app, this would be loaded from the CSV data
  const getHousingData = (lsoa) => {
    if (!lsoa) return null;
    
    // Generate deterministic but seemingly random data based on LSOA code
    const seed = lsoa.charCodeAt(6) * lsoa.charCodeAt(8);
    const rng = (min, max) => Math.floor(seed % 17 * (Math.random() + 0.5) * (max - min) / 17) + min;
    
    return {
      imdScore: rng(1, 10), // Index of Multiple Deprivation score (1-10)
      housingDensity: rng(10, 100), // Housing density per hectare
      socialHousingPercent: rng(5, 45), // % of social housing
      crimeRank: rng(1, 32), // Crime rank in London (1-32)
      wellbeingScore: rng(3, 9) / 10, // Wellbeing score (0.3-0.9)
      riskLevel: seed % 3 // 0 = low, 1 = medium, 2 = high
    };
  };

  const housingData = getHousingData(selectedLSOA);

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

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden mb-4">
      <h3 className="text-white text-base font-semibold p-3 border-b border-gray-700 flex items-center">
        <span className="text-orange-400 mr-2">🎯</span>
        EMMIE Scorecard Widget
      </h3>
      
      {/* Tabs */}
      <div className="flex border-b border-gray-700">
        {Object.keys(emmieFramework).map(key => (
          <button
            key={key}
            className={`py-2 px-4 text-sm font-medium transition-colors ${
              expandedSection === key 
                ? 'text-blue-400 border-b-2 border-blue-400' 
                : 'text-gray-400 hover:text-gray-300'
            }`}
            onClick={() => setExpandedSection(key)}
          >
            {emmieFramework[key].title}
          </button>
        ))}
      </div>
      
      {/* EMMIE Table */}
      <div className="p-2 bg-gray-900">
        <table className="w-full">
          <tbody>
            {emmieFramework[expandedSection].items.map((item, index) => (
              <motion.tr 
                key={index}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="border-b border-gray-800"
              >
                <td className="py-2 px-3 text-sm text-white">{item.name}</td>
                <td className="py-2 px-3 text-right">
                  {renderStars(item.score)}
                </td>
              </motion.tr>
            ))}
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
            
            {housingData && (
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