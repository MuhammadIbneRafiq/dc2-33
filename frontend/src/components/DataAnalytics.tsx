import React, { useState } from 'react';
import { 
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Area, AreaChart, ComposedChart, Label
} from 'recharts';
import { motion } from 'framer-motion';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

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
  { month: 'Jan', actual: 823, predicted: 845 },
  { month: 'Feb', actual: 786, predicted: 801 },
  { month: 'Mar', actual: 832, predicted: 815 },
  { month: 'Apr', actual: 783, predicted: 790 },
  { month: 'May', actual: 768, predicted: 756 },
  { month: 'Jun', actual: 795, predicted: 814 },
  { month: 'Jul', actual: 856, predicted: 832 },
  { month: 'Aug', actual: 901, predicted: 876 },
  { month: 'Sep', actual: 834, predicted: 865 },
  { month: 'Oct', actual: 864, predicted: 843 },
  { month: 'Nov', actual: 812, predicted: 834 },
  { month: 'Dec', actual: 795, predicted: 780 }
];

const burglaryForecastData = [
  { month: 'Dec', count: 795, allocation: 0 },
  { month: 'Jan', count: 845, allocation: 0 },
  { month: 'Feb', count: 825, allocation: 0 },
  { month: 'Mar', count: 810, allocation: 100 },
  { month: 'Apr', forecast: 780, allocation: 120 },
  { month: 'May', forecast: 730, allocation: 150 },
  { month: 'Jun', forecast: 690, allocation: 150 },
];

const factorData = [
  { name: 'Access Points', value: 28 },
  { name: 'Street Lighting', value: 15 },
  { name: 'Social Housing', value: 22 },
  { name: 'Population Density', value: 18 },
  { name: 'Previous Incidents', value: 17 }
];

const timeDistribution = [
  { time: '00:00-04:00', burglaries: 92, risk: 'Medium' },
  { time: '04:00-08:00', burglaries: 54, risk: 'Low' },
  { time: '08:00-12:00', burglaries: 143, risk: 'High' },
  { time: '12:00-16:00', burglaries: 189, risk: 'High' },
  { time: '16:00-20:00', burglaries: 231, risk: 'Very High' },
  { time: '20:00-00:00', burglaries: 165, risk: 'High' }
];

const predictionAccuracy = [
  { month: 'Jan', accuracy: 97 },
  { month: 'Feb', accuracy: 98 },
  { month: 'Mar', accuracy: 98 },
  { month: 'Apr', accuracy: 99 },
  { month: 'May', accuracy: 98 },
  { month: 'Jun', accuracy: 97 },
  { month: 'Jul', accuracy: 97 },
  { month: 'Aug', accuracy: 97 },
  { month: 'Sep', accuracy: 97 },
  { month: 'Oct', accuracy: 98 },
  { month: 'Nov', accuracy: 97 },
  { month: 'Dec', accuracy: 98 }
];

const reducedBurglaries = [
  { month: 'Jan', before: 845, after: 676 },
  { month: 'Feb', before: 825, after: 660 },
  { month: 'Mar', before: 810, after: 648 },
  { month: 'Apr', before: 780, after: 624 },
  { month: 'May', before: 730, after: 584 },
  { month: 'Jun', before: 690, after: 552 }
];

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

const DataAnalytics: React.FC<DataAnalyticsProps> = ({ 
  selectedLsoaCode,
  lsoaWellbeingData,
  isLoadingLsoaData 
}) => {
  const [activeTab, setActiveTab] = useState('forecast');

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
            {/* You can add a spinner here */}
          </div>
        )}

        {!isLoadingLsoaData && selectedLsoaCode && !lsoaWellbeingData && (
          <div className="text-center text-orange-400 py-4">
            <p>No wellbeing data found for {selectedLsoaCode}.</p>
          </div>
        )}

        {/* TODO: Integrate lsoaWellbeingData into the charts below */}
        {/* For example, the 'Risk Factors' tab could display IMD scores */}
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

        <Tabs defaultValue="forecast" onValueChange={setActiveTab} className="w-full">
          <TabsList className="mb-6 w-full grid grid-cols-2 lg:grid-cols-4 bg-gray-800 p-1 rounded-md">
            <TabsTrigger value="forecast" className="data-[state=active]:bg-blue-800 data-[state=active]:text-white">
              Forecasting
            </TabsTrigger>
            <TabsTrigger value="factors" className="data-[state=active]:bg-blue-800 data-[state=active]:text-white">
              Risk Factors
            </TabsTrigger>
            <TabsTrigger value="patterns" className="data-[state=active]:bg-blue-800 data-[state=active]:text-white">
              Time Patterns
            </TabsTrigger>
            <TabsTrigger value="impact" className="data-[state=active]:bg-blue-800 data-[state=active]:text-white">
              Impact Analysis
            </TabsTrigger>
          </TabsList>

          <TabsContent value="forecast" className="mt-2">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="dashboard-card p-5">
                <h3 className="text-lg font-semibold text-white mb-4">Historical vs Predicted Burglaries</h3>
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
                <h3 className="text-lg font-semibold text-white mb-4">Future Forecast with Resource Allocation</h3>
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
          </TabsContent>

          <TabsContent value="factors">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="dashboard-card p-5">
                <h3 className="text-lg font-semibold text-white mb-4">Residential Burglary Risk Factors</h3>
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
                <h3 className="text-lg font-semibold text-white mb-4">EMMIE Scores and Impact on Burglary</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={[
                        { name: 'Security Hardware', score: 4, reduction: 26 },
                        { name: 'Community Watch', score: 3, reduction: 15 },
                        { name: 'Target Hardening', score: 5, reduction: 32 },
                        { name: 'PCSO Patrols', score: 2, reduction: 12 }
                      ]}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="name" stroke="#9ca3af" />
                      <YAxis yAxisId="left" orientation="left" stroke="#9ca3af">
                        <Label value="EMMIE Score (1-5)" angle={-90} position="insideLeft" style={{ textAnchor: 'middle', fill: '#9ca3af' }} />
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
                      <Bar yAxisId="left" dataKey="score" fill="#3b82f6" name="EMMIE Score" />
                      <Bar yAxisId="right" dataKey="reduction" fill="#10b981" name="% Reduction" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-4 p-3 rounded-lg bg-gray-800 border border-gray-700 text-sm text-gray-300">
                  <p>EMMIE scores correlate with potential burglary reduction. Target hardening (score 5) shows highest reduction (32%).</p>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="patterns">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="dashboard-card p-5">
                <h3 className="text-lg font-semibold text-white mb-4">Residential Burglary Time Distribution</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={timeDistribution}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="time" stroke="#9ca3af" />
                      <YAxis stroke="#9ca3af">
                        <Label value="Number of Burglaries" angle={-90} position="insideLeft" style={{ textAnchor: 'middle', fill: '#9ca3af' }} />
                      </YAxis>
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '0.375rem' }} 
                        itemStyle={{ color: '#e5e7eb' }}
                        labelStyle={{ color: '#9ca3af' }}
                      />
                      <Bar 
                        dataKey="burglaries" 
                        fill="#3b82f6"
                        name="Residential Burglaries"
                      >
                        {timeDistribution.map((entry, index) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={
                              entry.risk === 'Very High' ? '#ef4444' :
                              entry.risk === 'High' ? '#f97316' :
                              entry.risk === 'Medium' ? '#f59e0b' :
                              '#84cc16'
                            } 
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-4 grid grid-cols-4 gap-2">
                  <div className="p-2 rounded-lg bg-gray-800 border border-red-900/30">
                    <div className="text-xs text-gray-400">Very High</div>
                    <div className="text-sm text-red-400 font-semibold">16:00-20:00</div>
                  </div>
                  <div className="p-2 rounded-lg bg-gray-800 border border-orange-900/30">
                    <div className="text-xs text-gray-400">High</div>
                    <div className="text-sm text-orange-400 font-semibold">12:00-16:00</div>
                  </div>
                  <div className="p-2 rounded-lg bg-gray-800 border border-yellow-900/30">
                    <div className="text-xs text-gray-400">Medium</div>
                    <div className="text-sm text-yellow-400 font-semibold">20:00-00:00</div>
                  </div>
                  <div className="p-2 rounded-lg bg-gray-800 border border-green-900/30">
                    <div className="text-xs text-gray-400">Low</div>
                    <div className="text-sm text-green-400 font-semibold">04:00-08:00</div>
                  </div>
                </div>
              </div>

              <div className="dashboard-card p-5">
                <h3 className="text-lg font-semibold text-white mb-4">Prediction Accuracy Over Time</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={predictionAccuracy}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="month" stroke="#9ca3af" />
                      <YAxis domain={[90, 100]} stroke="#9ca3af">
                        <Label value="Accuracy (%)" angle={-90} position="insideLeft" style={{ textAnchor: 'middle', fill: '#9ca3af' }} />
                      </YAxis>
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '0.375rem' }} 
                        itemStyle={{ color: '#e5e7eb' }}
                        labelStyle={{ color: '#9ca3af' }}
                      />
                      <Line type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={2} dot={{ r: 4 }} />
                      <Legend />
                      <CartesianGrid stroke="#374151" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-4 p-3 rounded-lg bg-gray-800 border border-gray-700 text-sm text-gray-300">
                  <p>Prediction accuracy has remained consistently above 95%, peaking at 99% in April.</p>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="impact">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="dashboard-card p-5">
                <h3 className="text-lg font-semibold text-white mb-4">Projected Impact of Interventions</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={reducedBurglaries}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
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
                      <Bar dataKey="before" name="Without Intervention" fill="#ef4444" />
                      <Bar dataKey="after" name="With Intervention" fill="#10b981" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-4 p-3 rounded-lg bg-gray-800 border border-gray-700">
                  <div className="text-sm text-green-400 font-semibold mb-1">Average Reduction</div>
                  <div className="text-2xl font-bold text-white">20%</div>
                  <div className="text-xs text-gray-400 mt-1">Using EMMIE-based resource allocation</div>
                </div>
              </div>

              <div className="dashboard-card p-5">
                <h3 className="text-lg font-semibold text-white mb-4">Cumulative Impact on Public Safety</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={[
                        { month: 'Jan', burglaries: 169, cost: 423, fear: 82 },
                        { month: 'Feb', burglaries: 165, cost: 413, fear: 81 },
                        { month: 'Mar', burglaries: 162, cost: 405, fear: 80 },
                        { month: 'Apr', burglaries: 156, cost: 390, fear: 78 },
                        { month: 'May', burglaries: 146, cost: 365, fear: 75 },
                        { month: 'Jun', burglaries: 138, cost: 345, fear: 71 }
                      ]}
                      margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="month" stroke="#9ca3af" />
                      <YAxis stroke="#9ca3af" />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '0.375rem' }} 
                        itemStyle={{ color: '#e5e7eb' }}
                        labelStyle={{ color: '#9ca3af' }}
                      />
                      <Area type="monotone" dataKey="burglaries" stackId="1" stroke="#3b82f6" fill="#3b82f6" name="Burglaries Prevented" />
                      <Area type="monotone" dataKey="cost" stackId="1" stroke="#8884d8" fill="#8884d8" name="Cost Savings (£1000s)" />
                      <Area type="monotone" dataKey="fear" stackId="1" stroke="#82ca9d" fill="#82ca9d" name="Fear Reduction Index" />
                      <Legend />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-4 grid grid-cols-3 gap-2">
                  <div className="p-3 rounded-lg bg-gray-800 border border-blue-900/30">
                    <div className="text-xs text-gray-400">Burglaries Prevented</div>
                    <div className="text-lg text-blue-400 font-semibold">936</div>
                  </div>
                  <div className="p-3 rounded-lg bg-gray-800 border border-purple-900/30">
                    <div className="text-xs text-gray-400">Cost Savings</div>
                    <div className="text-lg text-purple-400 font-semibold">£2.3M</div>
                  </div>
                  <div className="p-3 rounded-lg bg-gray-800 border border-green-900/30">
                    <div className="text-xs text-gray-400">Fear Reduction</div>
                    <div className="text-lg text-green-400 font-semibold">13.4%</div>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </motion.div>
    </div>
  );
};

export default DataAnalytics;
