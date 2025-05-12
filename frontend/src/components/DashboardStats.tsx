
import React from 'react';
import StatisticsCard from './StatisticsCard';
import { BarChart, Map, AlertTriangle, TrendingUp, Shield } from 'lucide-react';

const DashboardStats: React.FC = () => {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-6">
      <StatisticsCard
        title="Total Forecasts"
        value="4,893"
        change={12.5}
        changeLabel="last month"
        color="blue"
        icon={<Map size={20} className="text-blue-400" />}
      />
      
      <StatisticsCard
        title="Residential Burglaries"
        value="783"
        change={-7.2}
        changeLabel="last month"
        color="orange"
        icon={<AlertTriangle size={20} className="text-orange-400" />}
      />
      
      <StatisticsCard
        title="Prediction Accuracy"
        value="89%"
        change={3.1}
        changeLabel="vs target"
        color="green"
        icon={<TrendingUp size={20} className="text-green-400" />}
      />
      
      <StatisticsCard
        title="Officers Deployed"
        value="342"
        change={5.8}
        changeLabel="last week"
        color="purple"
        icon={<Shield size={20} className="text-purple-400" />}
      />
      
      <StatisticsCard
        title="Hotspots Identified"
        value="128"
        change={-3.4}
        changeLabel="last month"
        color="red"
        icon={<BarChart size={20} className="text-red-400" />}
      />
    </div>
  );
};

export default DashboardStats;
