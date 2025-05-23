import React from 'react';
import { motion } from 'framer-motion';
import { Star } from 'lucide-react';

interface EmmieExplanationProps {
  selectedLSOA?: string | null;
  lsoaFactors?: string[];
}

const EmmieExplanation: React.FC<EmmieExplanationProps> = ({ selectedLSOA, lsoaFactors }) => {
  const emmieDetails = [
    {
      title: "Effect",
      letter: "E",
      color: "blue",
      description: "The impact of an intervention on residential burglary rates or other crime rates.",
      details: "This measures whether interventions actually work to reduce residential burglaries. It considers the statistical strength of evidence showing a reduction in crime rates following the intervention."
    },
    {
      title: "Mechanism",
      letter: "M",
      color: "purple",
      description: "How the intervention works to reduce residential burglary.",
      details: "This explains the causal process through which an intervention leads to a reduction in residential burglaries. It might involve deterrence, increasing risks to offenders, removing opportunities, or changing environmental factors."
    },
    {
      title: "Moderators",
      letter: "M",
      color: "pink",
      description: "The conditions under which the intervention works best.",
      details: "These are the contextual factors that influence how effective an intervention will be in different settings. This could include geographic features, community characteristics, or time-based factors that affect residential burglary rates."
    },
    {
      title: "Implementation",
      letter: "I",
      color: "green",
      description: "How the intervention should be implemented for greatest effect.",
      details: "This covers the practical aspects of putting the intervention into action, including training requirements, resource allocation, and operational considerations to ensure the intervention is delivered as intended."
    },
    {
      title: "Economic",
      letter: "E",
      color: "amber",
      description: "The cost-effectiveness of the intervention.",
      details: "This assesses whether the cost of implementing the intervention is justified by the benefits in terms of reduced residential burglaries, cost savings to victims and police, and improved community safety."
    }
  ];

  return (
    <div className="p-6">
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        {selectedLSOA && lsoaFactors && (
          <div className="mb-8 bg-blue-900/20 border border-blue-900/30 rounded-lg p-6">
            <h3 className="text-xl font-bold text-blue-300 mb-2">Risk Factors for {selectedLSOA}</h3>
            <ul className="list-disc pl-6 text-blue-100 text-sm">
              {lsoaFactors.map((factor, idx) => (
                <li key={idx}>{factor}</li>
              ))}
            </ul>
          </div>
        )}
        <div className="flex items-center mb-6">
          <h2 className="text-2xl font-bold text-white">Understanding EMMIE</h2>
          <div className="ml-2 bg-blue-500/20 text-blue-300 px-3 py-1 rounded-full text-xs font-medium">
            Evidence-Based Policing
          </div>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-6 mb-8 border border-gray-700">
          <p className="text-gray-300">
            The EMMIE framework is an evidence-based approach developed by the College of Policing to assess crime prevention interventions. 
            It helps police forces make informed decisions about which strategies to deploy against residential burglary by evaluating them across five dimensions.
          </p>
          <ul className="mt-4 text-gray-400 text-sm list-disc pl-6">
            <li>Socioeconomic status of the area</li>
            <li>Housing type and density</li>
            <li>Lighting and visibility in public spaces</li>
            <li>Community cohesion and social capital</li>
            <li>Prior crime rates and repeat victimization</li>
            <li>Proximity to transport and escape routes</li>
          </ul>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
          {emmieDetails.map((item, index) => (
            <motion.div
              key={index}
              className={`bg-gray-800 rounded-lg border border-${item.color}-900/30 p-6 flex flex-col relative overflow-hidden`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.03 }}
            >
              <div className={`text-${item.color}-400 text-5xl font-bold absolute right-4 top-4 opacity-10`}>
                {item.letter}
              </div>
              <h3 className={`text-${item.color}-400 text-xl font-bold mb-3`}>{item.title}</h3>
              <p className="text-gray-300 text-sm mb-4">{item.description}</p>
              <div className="mt-auto">
                <div className="flex items-center">
                  <Star className={`text-${item.color}-400 mr-1`} size={14} />
                  <Star className={`text-${item.color}-400 mr-1`} size={14} />
                  <Star className={`text-${item.color}-400 mr-1`} size={14} />
                  <Star className={`text-${item.color}-400 mr-1`} size={14} />
                  <Star className={`text-${item.color}-400/30`} size={14} />
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-xl font-bold text-white mb-4">How EMMIE Scoring Works</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div>
              <h4 className="text-blue-400 font-semibold mb-2">Score Calculation</h4>
              <p className="text-gray-300 text-sm">
                Each intervention is rated on a scale of 1-5 stars for each EMMIE dimension. 
                These scores are based on the quality and consistency of available evidence, 
                with higher scores indicating stronger evidence of effectiveness.
              </p>
            </div>
            
            <div>
              <h4 className="text-blue-400 font-semibold mb-2">Application to Residential Burglary</h4>
              <p className="text-gray-300 text-sm">
                For residential burglary prevention, EMMIE scores help identify which 
                interventions are most likely to succeed in specific contexts. High scores 
                across all dimensions indicate interventions that are both effective and 
                practical to implement.
              </p>
            </div>
          </div>

          <div className="mb-6">
            <h4 className="text-blue-400 font-semibold mb-2">EMMIE Framework Examples</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left py-2 px-3 text-gray-400">Intervention</th>
                    <th className="text-center py-2 px-3 text-blue-400">Effect</th>
                    <th className="text-center py-2 px-3 text-purple-400">Mechanism</th>
                    <th className="text-center py-2 px-3 text-pink-400">Moderators</th>
                    <th className="text-center py-2 px-3 text-green-400">Implementation</th>
                    <th className="text-center py-2 px-3 text-amber-400">Economic</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-gray-800 hover:bg-gray-700/30">
                    <td className="py-3 px-3 text-white">Security Hardware</td>
                    <td className="py-3 px-3 text-center text-blue-400">★★★★☆</td>
                    <td className="py-3 px-3 text-center text-purple-400">★★★★★</td>
                    <td className="py-3 px-3 text-center text-pink-400">★★★☆☆</td>
                    <td className="py-3 px-3 text-center text-green-400">★★★★☆</td>
                    <td className="py-3 px-3 text-center text-amber-400">★★★★☆</td>
                  </tr>
                  <tr className="border-b border-gray-800 hover:bg-gray-700/30">
                    <td className="py-3 px-3 text-white">Community Watch</td>
                    <td className="py-3 px-3 text-center text-blue-400">★★★☆☆</td>
                    <td className="py-3 px-3 text-center text-purple-400">★★★☆☆</td>
                    <td className="py-3 px-3 text-center text-pink-400">★★★★☆</td>
                    <td className="py-3 px-3 text-center text-green-400">★★★★☆</td>
                    <td className="py-3 px-3 text-center text-amber-400">★★★☆☆</td>
                  </tr>
                  <tr className="border-b border-gray-800 hover:bg-gray-700/30">
                    <td className="py-3 px-3 text-white">Target Hardening</td>
                    <td className="py-3 px-3 text-center text-blue-400">★★★★★</td>
                    <td className="py-3 px-3 text-center text-purple-400">★★★★★</td>
                    <td className="py-3 px-3 text-center text-pink-400">★★★★☆</td>
                    <td className="py-3 px-3 text-center text-green-400">★★★★★</td>
                    <td className="py-3 px-3 text-center text-amber-400">★★★★★</td>
                  </tr>
                  <tr className="hover:bg-gray-700/30">
                    <td className="py-3 px-3 text-white">PCSO Patrols</td>
                    <td className="py-3 px-3 text-center text-blue-400">★★☆☆☆</td>
                    <td className="py-3 px-3 text-center text-purple-400">★★★☆☆</td>
                    <td className="py-3 px-3 text-center text-pink-400">★★☆☆☆</td>
                    <td className="py-3 px-3 text-center text-green-400">★★★☆☆</td>
                    <td className="py-3 px-3 text-center text-amber-400">★★☆☆☆</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="p-4 bg-blue-900/20 border border-blue-900/30 rounded-lg text-blue-300 text-sm">
            <p>
              <strong>How to use EMMIE in this application:</strong> When selecting an area on the map, 
              you will see EMMIE recommendations tailored to that location based on its specific characteristics. 
              These recommendations are calculated by analyzing the area's risk factors against the evidence-based 
              EMMIE framework.
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default EmmieExplanation;
