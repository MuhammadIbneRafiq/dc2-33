import React, { useEffect, useRef } from 'react';
import { Chart, registerables } from 'chart.js';
import 'chart.js/auto';

// Register all Chart.js components
Chart.register(...registerables);

const BurglaryChart = ({ data, showPoliceImpact }) => {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  useEffect(() => {
    // If no data, don't render the chart
    if (!data) return;
    
    // Clean up any existing chart
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }
    
    const ctx = chartRef.current.getContext('2d');
    
    // Create chart datasets
    const datasets = [
      {
        label: 'Burglaries (Historical)',
        data: data.countsBefore,
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239, 68, 68, 0.2)',
        borderWidth: 2,
        tension: 0.4,
        fill: true,
        pointBackgroundColor: '#ef4444',
        pointRadius: 3,
        pointHoverRadius: 5,
      }
    ];
    
    // Add the "after police allocation" dataset if needed
    if (showPoliceImpact) {
      datasets.push({
        label: 'Projected Burglaries (With Police)',
        data: data.countsAfter,
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.2)',
        borderWidth: 2,
        tension: 0.4,
        fill: true,
        pointBackgroundColor: '#10b981',
        pointRadius: 3,
        pointHoverRadius: 5,
      });
    }
    
    // Create the chart
    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.labels,
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animations: {
          tension: {
            duration: 1000,
            easing: 'linear'
          }
        },
        scales: {
          x: {
            grid: {
              color: 'rgba(255, 255, 255, 0.1)',
            },
            ticks: {
              color: 'rgba(255, 255, 255, 0.7)',
              font: {
                family: 'Inter, sans-serif'
              }
            }
          },
          y: {
            beginAtZero: true,
            grid: {
              color: 'rgba(255, 255, 255, 0.1)',
            },
            ticks: {
              color: 'rgba(255, 255, 255, 0.7)',
              font: {
                family: 'Inter, sans-serif'
              }
            },
            title: {
              display: true,
              text: 'Burglary Count',
              color: 'rgba(255, 255, 255, 0.7)',
              font: {
                family: 'Inter, sans-serif',
                weight: 'normal'
              }
            }
          }
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: 'rgba(255, 255, 255, 0.9)',
              font: {
                family: 'Inter, sans-serif'
              },
              usePointStyle: true,
              padding: 20
            }
          },
          tooltip: {
            backgroundColor: 'rgba(31, 41, 55, 0.8)',
            titleColor: 'rgba(255, 255, 255, 0.9)',
            bodyColor: 'rgba(255, 255, 255, 0.9)',
            borderColor: 'rgba(59, 130, 246, 0.5)',
            borderWidth: 1,
            bodyFont: {
              family: 'Inter, sans-serif'
            },
            titleFont: {
              family: 'Inter, sans-serif',
              weight: 'bold'
            },
            padding: 10,
            displayColors: false,
            callbacks: {
              afterLabel: function(context) {
                if (showPoliceImpact && context.datasetIndex === 1) {
                  const originalValue = data.countsBefore[context.dataIndex];
                  const newValue = data.countsAfter[context.dataIndex];
                  const reductionPercent = Math.round((originalValue - newValue) / originalValue * 100);
                  return `Crime reduction: ${reductionPercent}%`;
                }
                return '';
              }
            }
          }
        }
      }
    });
    
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [data, showPoliceImpact]);
  
  if (!data) {
    return (
      <div className="dashboard-card h-full p-4 flex items-center justify-center">
        <div className="text-gray-400 text-center">
          {/* <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 mx-auto mb-2 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg> */}
          <p className="text-sm">Select an LSOA on the map to view burglary trends</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-card">
      <div className="flex items-center justify-between card-header px-4 py-3">
        <h3 className="text-white text-lg font-semibold flex items-center">
          {/* <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
            <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z" />
          </svg> */}
          Burglary Time Trends
        </h3>
        
        {showPoliceImpact && (
          <div className="flex items-center px-3 py-1 bg-green-600/20 text-green-400 rounded-md text-sm border border-green-600/30">
            <span className="font-medium flex items-center">
              {/* <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg> */}
              Police Impact Shown
            </span>
          </div>
        )}
      </div>
      
      <div className="p-4 crime-time-chart">
        <canvas ref={chartRef}></canvas>
      </div>
      
      <div className="p-3 bg-gray-800 border-t border-gray-700">
        <div className="flex flex-wrap justify-between items-center">
          <div className="flex items-center text-sm text-gray-400">
            {/* <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg> */}
            <span>Data shows monthly burglary counts in the selected LSOA</span>
          </div>
          
          {data && data.reductionPercent && showPoliceImpact && (
            <div className="mt-2 sm:mt-0 flex items-center bg-blue-600/20 px-3 py-1 rounded-md text-sm">
              <span className="text-blue-400 font-medium mr-1">Estimated reduction:</span> 
              <span className="text-green-400 font-bold">{data.reductionPercent}%</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BurglaryChart; 