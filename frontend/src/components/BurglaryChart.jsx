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
        label: 'Burglaries (Current)',
        data: data.countsBefore,
        borderColor: 'rgb(235, 70, 70)',
        backgroundColor: 'rgba(235, 70, 70, 0.1)',
        fill: true,
        tension: 0.4,
        borderWidth: 2,
        pointBackgroundColor: 'rgb(235, 70, 70)',
        pointRadius: 3,
      }
    ];
    
    // Add the "with police allocation" dataset if toggle is on
    if (showPoliceImpact) {
      datasets.push({
        label: 'Burglaries (With Optimal Police Allocation)',
        data: data.countsAfter,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.1)',
        fill: true,
        tension: 0.4,
        borderWidth: 2,
        pointBackgroundColor: 'rgb(75, 192, 192)',
        pointRadius: 3,
      });
    }
    
    // Create new chart
    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.months,
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: {
              boxWidth: 12,
              usePointStyle: true,
              pointStyle: 'circle'
            }
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            callbacks: {
              footer: (tooltipItems) => {
                if (showPoliceImpact && tooltipItems.length > 1) {
                  const before = tooltipItems[0].raw;
                  const after = tooltipItems[1].raw;
                  const reduction = before - after;
                  const percentReduction = ((reduction / before) * 100).toFixed(1);
                  return `Reduction: ${reduction} (${percentReduction}%)`;
                }
                return '';
              }
            }
          }
        },
        scales: {
          x: {
            grid: {
              display: false
            },
            ticks: {
              maxRotation: 45,
              minRotation: 45
            }
          },
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Number of Burglaries'
            },
            grid: {
              color: 'rgba(0, 0, 0, 0.05)'
            }
          }
        },
        interaction: {
          mode: 'nearest',
          axis: 'x',
          intersect: false
        },
        animations: {
          tension: {
            duration: 1000,
            easing: 'linear'
          }
        }
      }
    });
    
    // Clean up chart on component unmount
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [data, showPoliceImpact]);

  return (
    <div className="crime-time-chart">
      {data ? (
        <canvas ref={chartRef}></canvas>
      ) : (
        <div className="flex items-center justify-center h-full">
          <p className="text-gray-500">No data available</p>
        </div>
      )}
    </div>
  );
};

export default BurglaryChart; 