import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  Area,
  AreaChart,
  ComposedChart,
  Label
} from 'recharts';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';

// Define props interface
interface LsoaWellbeingData {
  lsoa_code: string;
  lsoa_name?: string; // Or 'name' as per Dashboard.tsx useQuery for lsoaData
  imd_score?: number;
  income_score?: number;
  employment_score?: number;
  education_score?: number;
  health_score?: number;
  crime_score?: number;
  housing_score?: number;
  living_environment_score?: number;
  // Add other properties from your backend's LSOA wellbeing data structure
}

interface DataAnalyticsProps {
  selectedLsoaCode: string | null;
  lsoaWellbeingData: LsoaWellbeingData | null;
  isLoadingLsoaData: boolean;
}

const burglaryHistoricalData = [
  { month: 'Jan', actual: 145, predicted: 142 },
  { month: 'Feb', actual: 160, predicted: 155 },
  { month: 'Mar', actual: 175, predicted: 180 },
  { month: 'Apr', actual: 190, predicted: 185 },
  { month: 'May', actual: 210, predicted: 205 },
  { month: 'Jun', actual: 225, predicted: 230 },
];

const burglaryForecastData = [
  { month: 'Jul', count: 225, forecast: 205, allocation: 12 },
  { month: 'Aug', count: 240, forecast: 215, allocation: 14 },
  { month: 'Sep', count: 255, forecast: 220, allocation: 16 },
  { month: 'Oct', count: 270, forecast: 225, allocation: 18 },
  { month: 'Nov', count: 285, forecast: 230, allocation: 20 },
  { month: 'Dec', count: 300, forecast: 235, allocation: 22 },
];

const factorData = [
  { name: 'Access Points', value: 28 },
  { name: 'Foot Traffic', value: 22 },
  { name: 'Time of Day', value: 18 },
  { name: 'Economic Factors', value: 15 },
  { name: 'Lighting', value: 10 },
  { name: 'Other', value: 7 }
];

const timeDistribution = [
  { time: '00:00-04:00', burglaries: 12, risk: 'Low' },
  { time: '04:00-08:00', burglaries: 8, risk: 'Low' },
  { time: '08:00-12:00', burglaries: 25, risk: 'Medium' },
  { time: '12:00-16:00', burglaries: 45, risk: 'High' },
  { time: '16:00-20:00', burglaries: 52, risk: 'Very High' },
  { time: '20:00-00:00', burglaries: 31, risk: 'Medium' }
];

const predictionAccuracy = [
  { month: 'Jan', accuracy: 92.5 },
  { month: 'Feb', accuracy: 94.2 },
  { month: 'Mar', accuracy: 91.8 },
  { month: 'Apr', accuracy: 96.1 },
  { month: 'May', accuracy: 93.7 },
  { month: 'Jun', accuracy: 95.3 }
];

const reducedBurglaries = [
  { month: 'Jul', before: 225, after: 180 },
  { month: 'Aug', before: 240, after: 192 },
  { month: 'Sep', before: 255, after: 204 },
  { month: 'Oct', before: 270, after: 216 },
  { month: 'Nov', before: 285, after: 228 },
  { month: 'Dec', before: 300, after: 240 }
];

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

const DataAnalytics: React.FC<DataAnalyticsProps> = ({ 
  selectedLsoaCode,
  lsoaWellbeingData,
  isLoadingLsoaData 
}) => {
  // Determine title based on selected LSOA
  const analyticsTitle = selectedLsoaCode 
    ? `Analytics for ${lsoaWellbeingData?.lsoa_name || selectedLsoaCode}` 
    : "Overall London Burglary Analytics";

  return (
    <div className="p-6 bg-gray-800/70 rounded-xl border border-gray-700/50 shadow-lg">
      <motion.div 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h2 className="text-2xl font-bold text-white mb-6">{analyticsTitle}</h2>

        {isLoadingLsoaData && selectedLsoaCode && (
          <div className="text-center text-white py-4">
            <p>Loading wellbeing data for {selectedLsoaCode}...</p>
          </div>
        )}

        {!isLoadingLsoaData && selectedLsoaCode && !lsoaWellbeingData && (
          <div className="text-center text-orange-400 py-4">
            <p>No wellbeing data found for {selectedLsoaCode}.</p>
          </div>
        )}

        {selectedLsoaCode && lsoaWellbeingData && (
          <div className="mb-6 p-4 bg-gray-900/50 rounded-lg">
            <h3 className="text-lg font-semibold text-blue-300 mb-2">Wellbeing Scores for {lsoaWellbeingData.lsoa_name || selectedLsoaCode}:</h3>
            <ul className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
              {lsoaWellbeingData.imd_score && <li>IMD Score: <span className="font-semibold text-white">{lsoaWellbeingData.imd_score.toFixed(2)}</span></li>}
              {lsoaWellbeingData.income_score && <li>Income Score: <span className="font-semibold text-white">{lsoaWellbeingData.income_score.toFixed(2)}</span></li>}
              {lsoaWellbeingData.employment_score && <li>Employment: <span className="font-semibold text-white">{lsoaWellbeingData.employment_score.toFixed(2)}</span></li>}
              {lsoaWellbeingData.education_score && <li>Education: <span className="font-semibold text-white">{lsoaWellbeingData.education_score.toFixed(2)}</span></li>}
              {lsoaWellbeingData.health_score && <li>Health Score: <span className="font-semibold text-white">{lsoaWellbeingData.health_score.toFixed(2)}</span></li>}
              {lsoaWellbeingData.crime_score && <li>Crime Score: <span className="font-semibold text-white">{lsoaWellbeingData.crime_score.toFixed(2)}</span></li>}
              {lsoaWellbeingData.housing_score && <li>Housing Score: <span className="font-semibold text-white">{lsoaWellbeingData.housing_score.toFixed(2)}</span></li>}
              {lsoaWellbeingData.living_environment_score && <li>Environment: <span className="font-semibold text-white">{lsoaWellbeingData.living_environment_score.toFixed(2)}</span></li>}
            </ul>
          </div>
        )}

        {/* Analytics Content - All in one view */}
        <div className="space-y-8">
          {/* Forecasting Section */}
          <div>
            <h3 className="text-xl font-semibold text-white mb-4">Forecasting Analysis</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="dashboard-card p-5">
                <h4 className="text-lg font-semibold text-white mb-4">Historical vs Predicted Burglaries</h4>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={burglaryHistoricalData}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="month" stroke="#9ca3af" />
                      <YAxis stroke="#9ca3af">
                        <Label value="Residential Burglaries" angle={-90} position="insideLeft" style={{ textAnchor: 'middle', fill: '#9ca3af' }} />
                      </YAxis>
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '0.375rem' }} 
                        itemStyle={{ color: '#e5e7eb' }}
                        labelStyle={{ color: '#9ca3af' }}
                      />
                      <Legend />
                      <Line type="monotone" dataKey="actual" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} name="Actual" />
                      <Line type="monotone" dataKey="predicted" stroke="#f97316" strokeWidth={2} dot={{ r: 4 }} name="Predicted" activeDot={{ r: 8 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-4 p-3 rounded-lg bg-gray-800 border border-gray-700">
                  <div className="text-sm text-blue-400 font-semibold mb-1">Prediction Accuracy</div>
                  <div className="text-2xl font-bold text-white">89.7%</div>
                  <div className="text-xs text-gray-400 mt-1">Average over the last 12 months</div>
                </div>
              </div>

              <div className="dashboard-card p-5">
                <h4 className="text-lg font-semibold text-white mb-4">Future Forecast with Resource Allocation</h4>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart
                      data={burglaryForecastData}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="month" stroke="#9ca3af" />
                      <YAxis yAxisId="left" stroke="#9ca3af" />
                      <YAxis yAxisId="right" orientation="right" stroke="#9ca3af" />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '0.375rem' }} 
                        itemStyle={{ color: '#e5e7eb' }}
                        labelStyle={{ color: '#9ca3af' }}
                      />
                      <Legend />
                      <Area yAxisId="left" type="monotone" dataKey="count" fill="#3b82f6" stroke="#3b82f6" name="Historical" />
                      <Area yAxisId="left" type="monotone" dataKey="forecast" fill="url(#colorForecast)" stroke="#10b981" name="Forecast" />
                      <Bar yAxisId="right" dataKey="allocation" barSize={20} fill="#8884d8" name="Police Units" />
                      <defs>
                        <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
                          <stop offset="95%" stopColor="#10b981" stopOpacity={0.2}/>
                        </linearGradient>
                      </defs>
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-4 p-3 rounded-lg bg-gray-800 border border-gray-700">
                  <div className="text-sm text-green-400 font-semibold mb-1">Forecasted Reduction</div>
                  <div className="text-2xl font-bold text-white">20.3%</div>
                  <div className="text-xs text-gray-400 mt-1">With optimal resource allocation</div>
                </div>
              </div>
            </div>
          </div>

          {/* Risk Factors Section */}
          <div>
            <h3 className="text-xl font-semibold text-white mb-4">Risk Analysis</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="dashboard-card p-5">
                <h4 className="text-lg font-semibold text-white mb-4">Residential Burglary Risk Factors</h4>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={factorData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      >
                        {factorData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip 
                        formatter={(value) => [`${value}%`, 'Contribution']}
                        contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '0.375rem' }} 
                        itemStyle={{ color: '#e5e7eb' }}
                        labelStyle={{ color: '#9ca3af' }}
                      />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-4 p-3 rounded-lg bg-gray-800 border border-gray-700 text-sm text-gray-300">
                  <p>Risk factors contribute to likelihood of residential burglaries. Access points (28%) is the most significant factor.</p>
                </div>
              </div>

              <div className="dashboard-card p-5">
                <h4 className="text-lg font-semibold text-white mb-4">Intervention Effectiveness Analysis</h4>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={[
                        { name: 'Security Hardware', effectiveness: 4, reduction: 26 },
                        { name: 'Community Watch', effectiveness: 3, reduction: 15 },
                        { name: 'Target Hardening', effectiveness: 5, reduction: 32 },
                        { name: 'PCSO Patrols', effectiveness: 2, reduction: 12 }
                      ]}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="name" stroke="#9ca3af" />
                      <YAxis yAxisId="left" orientation="left" stroke="#9ca3af">
                        <Label value="Effectiveness Score (1-5)" angle={-90} position="insideLeft" style={{ textAnchor: 'middle', fill: '#9ca3af' }} />
                      </YAxis>
                      <YAxis yAxisId="right" orientation="right" stroke="#9ca3af">
                        <Label value="% Reduction" angle={90} position="insideRight" style={{ textAnchor: 'middle', fill: '#9ca3af' }} />
                      </YAxis>
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '0.375rem' }} 
                        itemStyle={{ color: '#e5e7eb' }}
                        labelStyle={{ color: '#9ca3af' }}
                      />
                      <Legend />
                      <Bar yAxisId="left" dataKey="effectiveness" fill="#3b82f6" name="Effectiveness Score" />
                      <Bar yAxisId="right" dataKey="reduction" fill="#10b981" name="% Reduction" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-4 p-3 rounded-lg bg-gray-800 border border-gray-700 text-sm text-gray-300">
                  <p>Effectiveness scores correlate with potential burglary reduction. Target hardening (score 5) shows highest reduction (32%).</p>
                </div>
              </div>
                </div>
              </div>

            </div>
      </motion.div>
    </div>
  );
};

export default DataAnalytics;
