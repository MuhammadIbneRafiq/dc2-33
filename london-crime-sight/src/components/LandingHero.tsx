
import React from 'react';
import { motion } from 'framer-motion';
import { Shield, ChevronRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import MapIcon from './MapIcon';
import BarChart from './BarChart';

const LandingHero: React.FC = () => {
  return (
    <section className="py-20 px-4 md:px-6 bg-gradient-to-b from-gray-900 to-gray-950 relative overflow-hidden">
      {/* Background graphics */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-10 left-1/4 w-64 h-64 bg-blue-500/5 rounded-full filter blur-3xl"></div>
        <div className="absolute -bottom-20 right-1/3 w-80 h-80 bg-indigo-500/5 rounded-full filter blur-3xl"></div>
        <div className="absolute top-1/2 right-1/4 w-40 h-40 bg-purple-500/5 rounded-full filter blur-3xl"></div>
      </div>
      
      {/* Grid overlay */}
      <div className="absolute inset-0 bg-grid-pattern opacity-[0.03]"></div>
      
      <div className="container mx-auto relative z-10">
        <div className="max-w-3xl mx-auto text-center mb-12">
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center px-3 py-1 mb-6 rounded-full bg-blue-900/30 border border-blue-700/30 text-blue-400 text-sm"
          >
            <Shield size={16} className="mr-2" />
            <span>Advanced Crime Forecasting Technology</span>
          </motion.div>
          
          <motion.h1
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-white via-blue-100 to-indigo-200"
          >
            London Residential Burglary Prevention System
          </motion.h1>
          
          <motion.p
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.4, duration: 0.5 }}
            className="text-lg text-gray-300 mb-8"
          >
            Leveraging predictive analytics and machine learning to forecast residential burglary hotspots,
            enabling targeted resource allocation and proactive crime prevention strategies.
          </motion.p>
          
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.6, duration: 0.5 }}
            className="flex flex-col sm:flex-row justify-center gap-4"
          >
            <Link to="/dashboard">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="px-8 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white rounded-lg font-medium flex items-center justify-center shadow-lg shadow-blue-900/30 border border-blue-700/50"
              >
                View Dashboard
                <ChevronRight size={18} className="ml-1" />
              </motion.button>
            </Link>
            
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-8 py-3 bg-gray-800 hover:bg-gray-700 text-gray-200 rounded-lg font-medium border border-gray-700/50 shadow-md"
            >
              Learn More
            </motion.button>
          </motion.div>
        </div>
        
        {/* Feature highlights */}
        <motion.div
          initial={{ y: 50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.8, duration: 0.6 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-16"
        >
          <div className="bg-gray-800/50 backdrop-blur-sm p-6 rounded-xl border border-gray-700/50 shadow-lg">
            <div className="w-12 h-12 bg-blue-900/30 rounded-lg flex items-center justify-center mb-4 text-blue-400">
              <MapIcon size={24} />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Predictive Mapping</h3>
            <p className="text-gray-300">Visualize projected burglary hotspots with advanced heatmapping and geospatial analysis.</p>
          </div>
          
          <div className="bg-gray-800/50 backdrop-blur-sm p-6 rounded-xl border border-gray-700/50 shadow-lg">
            <div className="w-12 h-12 bg-indigo-900/30 rounded-lg flex items-center justify-center mb-4 text-indigo-400">
              <Shield size={24} />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Resource Allocation</h3>
            <p className="text-gray-300">Optimize police unit deployment based on risk levels and operational constraints.</p>
          </div>
          
          <div className="bg-gray-800/50 backdrop-blur-sm p-6 rounded-xl border border-gray-700/50 shadow-lg">
            <div className="w-12 h-12 bg-purple-900/30 rounded-lg flex items-center justify-center mb-4 text-purple-400">
              <BarChart size={24} />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">EMMIE Framework</h3>
            <p className="text-gray-300">Evidence-based intervention assessment using the EMMIE scoring methodology.</p>
          </div>
        </motion.div>
        
        {/* Statistics */}
        <motion.div
          initial={{ y: 50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 1.0, duration: 0.6 }}
          className="mt-16 bg-gradient-to-r from-gray-900 to-gray-800/50 p-8 rounded-2xl border border-gray-800/50 shadow-xl"
        >
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-white">System Performance</h2>
            <p className="text-gray-400">Proven results in residential burglary prediction</p>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center">
              <p className="text-3xl font-bold text-blue-400">89%</p>
              <p className="text-sm text-gray-400">Prediction Accuracy</p>
            </div>
            
            <div className="text-center">
              <p className="text-3xl font-bold text-purple-400">26%</p>
              <p className="text-sm text-gray-400">Burglary Reduction</p>
            </div>
            
            <div className="text-center">
              <p className="text-3xl font-bold text-green-400">12.5k</p>
              <p className="text-sm text-gray-400">Data Points Analyzed</p>
            </div>
            
            <div className="text-center">
              <p className="text-3xl font-bold text-indigo-400">32</p>
              <p className="text-sm text-gray-400">London Boroughs</p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default LandingHero;
