// Mock service functions to simulate API calls

/**
 * Get optimized police allocation data based on the number of units, deployment hours, and time of day
 */
export const getPoliceAllocation = async (
  units: number, 
  deploymentHours: number = 2, 
  deploymentTime: string = '08:00'
): Promise<any[]> => {
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 1500));
  
  // Determine effectiveness modifier based on time of day
  let timeEffectivenessModifier = 0;
  
  // Higher effectiveness during high crime hours (evening/night)
  if (deploymentTime.startsWith('18:') || deploymentTime.startsWith('20:') || 
      deploymentTime.startsWith('22:') || deploymentTime.startsWith('00:')) {
    timeEffectivenessModifier = 10; // Better effectiveness during peak crime hours
  } else if (deploymentTime.startsWith('08:') || deploymentTime.startsWith('16:')) {
    timeEffectivenessModifier = 5; // Moderate effectiveness during transition hours
  }
  
  // Deployment hours modifier (more hours can lead to fatigue but better coverage)
  const hoursEffectivenessModifier = deploymentHours <= 4 ? 5 : 0;
  
  // Generate random police allocation data based on units and deployment parameters
  const policeUnits = Array.from({ length: units }, (_, index) => {
    const unitId = `PU-${String(index + 1).padStart(3, '0')}`;
    const estimatedBurglaries = Math.floor(Math.random() * 30) + 5;
    
    // Base effectiveness is 60-90, modified by time of day and deployment hours
    const baseEffectiveness = Math.floor(Math.random() * 30) + 60;
    const effectivenessScore = Math.min(99, baseEffectiveness + timeEffectivenessModifier + hoursEffectivenessModifier);
    
    const patrolType = Math.random() > 0.6 ? 'Vehicle' : 'Foot';
    
    return {
      unit_id: unitId,
      estimated_burglaries: estimatedBurglaries,
      effectiveness_score: effectivenessScore,
      patrol_type: patrolType,
      deployment_hours: deploymentHours,
      deployment_time: deploymentTime,
      latitude: 51.5 + (Math.random() * 0.1) - 0.05,
      longitude: -0.12 + (Math.random() * 0.2) - 0.1,
    };
  });
  
  return policeUnits;
};

/**
 * Get EMMIE framework scores for different intervention types
 */
export const getEmmieScores = async () => {
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  // Return simulated EMMIE framework data
  return {
    intervention: {
      title: "Intervention",
      items: [
        { name: "Security Hardware", description: "Locks, alarms, CCTV", score: 4 },
        { name: "Community Watch", description: "Neighborhood surveillance groups", score: 3 },
        { name: "Target Hardening", description: "Physical barriers to entry", score: 5 },
        { name: "PCSO Patrols", description: "Community Support Officers", score: 2 }
      ]
    },
    evidence: {
      title: "Evidence",
      items: [
        { name: "Systematic Reviews", description: "Academic literature reviews", score: 4 },
        { name: "Case Studies", description: "Real-world implementations", score: 3 },
        { name: "Statistical Analysis", description: "Crime pattern analysis", score: 5 },
        { name: "Predictive Modeling", description: "ML forecasting systems", score: 4 }
      ]
    },
    mechanism: {
      title: "Mechanism",
      items: [
        { name: "Deterrence", description: "Visible police presence", score: 4 },
        { name: "Crime Prevention", description: "Target hardening", score: 5 },
        { name: "Social Control", description: "Community engagement", score: 3 },
        { name: "Environmental Design", description: "Urban planning for safety", score: 4 }
      ]
    },
    implementation: {
      title: "Implementation",
      items: [
        { name: "Officer Training", description: "Specialist residential burglary reduction", score: 3 },
        { name: "Resource Allocation", description: "Targeted deployment", score: 5 },
        { name: "Stakeholder Engagement", description: "Working with communities", score: 4 },
        { name: "Temporal Targeting", description: "Time-based deployment", score: 4 }
      ]
    },
    economic: {
      title: "Economic",
      items: [
        { name: "Cost-Benefit Ratio", description: "Return on investment", score: 4 },
        { name: "Resource Efficiency", description: "Optimal unit deployment", score: 5 },
        { name: "Long-term Savings", description: "Reduced crime costs", score: 3 },
        { name: "Implementation Cost", description: "Initial expenditure", score: 2 }
      ]
    }
  };
};

/**
 * Get wellbeing data for a specific LSOA (Lower Super Output Area)
 */
export const getLsoaWellbeingData = async (lsoaCode: string) => {
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 800));
  
  // Mock data - normally this would be fetched from an API
  return {
    lsoa_code: lsoaCode,
    imd_score: Math.random() * 10,  // 0-10, higher is worse
    housing_score: Math.floor(Math.random() * 50) + 20,  // Housing density 20-70
    income_score: Math.random() * 0.4,  // 0-0.4 (will be displayed as 0-40%)
    crime_score: Math.random(),  // 0-1, higher is worse
    employment_score: Math.random(),  // 0-1
    education_score: Math.random(),  // 0-1
    health_score: Math.random(),  // 0-1
    access_services_score: Math.random(),  // 0-1
    living_environment_score: Math.random(),  // 0-1
  };
};

/**
 * Get crime forecast data for the London area
 */
export const getCrimeForecastData = async () => {
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  // Generate random crime hotspots across London
  const hotspots = Array.from({ length: 50 }, () => {
    // London coordinates with some randomization
    const latitude = 51.5 + (Math.random() * 0.2) - 0.1;
    const longitude = -0.12 + (Math.random() * 0.2) - 0.1;
    const intensity = Math.random();
    const burglaryRisk = Math.floor(Math.random() * 100) + 1;
    const lsoaCode = `E${String(Math.floor(Math.random() * 90000) + 10000).padStart(5, '0')}`;
    
    return {
      latitude,
      longitude,
      intensity,
      burglary_risk: burglaryRisk,
      lsoa_code: lsoaCode,
      trend: Math.random() > 0.5 ? 'up' : 'down',
      last_month_incidents: Math.floor(Math.random() * 30),
    };
  });
  
  return hotspots;
};
