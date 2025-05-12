
import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface StatisticsCardProps {
  title: string;
  value: string | number;
  change: number;
  changeLabel: string;
  color: 'blue' | 'green' | 'red' | 'orange' | 'purple';
  icon: React.ReactNode;
}

const StatisticsCard: React.FC<StatisticsCardProps> = ({
  title,
  value,
  change,
  changeLabel,
  color,
  icon
}) => {
  const cardColors = {
    blue: {
      bg: 'bg-blue-900/20',
      border: 'border-blue-900/30',
      iconBg: 'bg-blue-900/30',
      valueColor: 'text-blue-400',
      changeColor: change >= 0 ? 'text-green-400' : 'text-red-400'
    },
    green: {
      bg: 'bg-green-900/20',
      border: 'border-green-900/30',
      iconBg: 'bg-green-900/30',
      valueColor: 'text-green-400',
      changeColor: change >= 0 ? 'text-green-400' : 'text-red-400'
    },
    red: {
      bg: 'bg-red-900/20',
      border: 'border-red-900/30',
      iconBg: 'bg-red-900/30',
      valueColor: 'text-red-400',
      changeColor: change >= 0 ? 'text-green-400' : 'text-red-400'
    },
    orange: {
      bg: 'bg-orange-900/20',
      border: 'border-orange-900/30',
      iconBg: 'bg-orange-900/30',
      valueColor: 'text-orange-400',
      changeColor: change >= 0 ? 'text-green-400' : 'text-red-400'
    },
    purple: {
      bg: 'bg-purple-900/20',
      border: 'border-purple-900/30',
      iconBg: 'bg-purple-900/30',
      valueColor: 'text-purple-400',
      changeColor: change >= 0 ? 'text-green-400' : 'text-red-400'
    }
  };
  
  const colors = cardColors[color];
  
  return (
    <motion.div 
      className={`dashboard-card ${colors.bg} border ${colors.border}`}
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.2 }}
    >
      <div className="p-5">
        <div className="flex justify-between items-start">
          <div>
            <p className="text-sm text-gray-400 mb-1">{title}</p>
            <h3 className={`text-2xl font-bold ${colors.valueColor}`}>{value}</h3>
          </div>
          
          <div className={`p-3 rounded-full ${colors.iconBg}`}>
            {icon}
          </div>
        </div>
        
        <div className="mt-4 flex items-center">
          {change >= 0 ? (
            <TrendingUp size={16} className="text-green-400 mr-1" />
          ) : (
            <TrendingDown size={16} className="text-red-400 mr-1" />
          )}
          
          <span className={colors.changeColor}>
            {change >= 0 ? '+' : ''}{change}%
          </span>
          <span className="text-gray-400 ml-1 text-sm">
            {changeLabel}
          </span>
        </div>
      </div>
    </motion.div>
  );
};

export default StatisticsCard;
