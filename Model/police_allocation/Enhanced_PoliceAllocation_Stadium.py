# -*- coding: utf-8 -*-
"""
Enhanced Police Allocation System with Stadium Proximity Risk Factor
===================================================================

This system allocates police resources across London wards based on:
1. Original risk predictions (burglary forecasts)
2. Population density
3. Stadium proximity factor (NEW)

Stadium proximity affects crime patterns based on research showing:
- Tottenham & Arsenal areas: +9-13% burglary increase during match periods
- Chelsea area: -8% burglary decrease (protective effect)
- Stadium events create crowd dynamics affecting local crime

Author: Enhanced by AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, asin

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# === FILE PATHS ===
risk_predictions_path = r"monthly_risk_predictions_2025.csv"
lsoa_to_ward_path = r"LSOA11_WD21_LAD21_EW_LU_V2.xlsx"
population_path = r"sapelsoasyoa20192022.xlsx"

# === STADIUM RISK PARAMETERS ===
STADIUM_INFLUENCE_RADIUS_KM = 3.0  # Radius of stadium influence
STADIUM_RISK_MULTIPLIER = {
    # Based on empirical analysis of crime patterns around stadiums
    'high_risk': 1.15,      # +15% for Tottenham/Arsenal type areas
    'medium_risk': 1.05,    # +5% for general stadium areas
    'protective': 0.92      # -8% for Chelsea type protective areas
}

# === POLICE ALLOCATION PARAMETERS ===
TOTAL_HOURS_PER_WARD = 200  # Base hours per ward
HOURS_PER_OFFICER = 2       # Officer shift duration

# ============================================================================
# STADIUM DATA AND FUNCTIONS
# ============================================================================

def define_london_stadiums():
    """
    Define London football stadiums with empirically-derived risk factors
    
    Risk classifications based on empirical crime analysis:
    - high_risk: Areas showing significant crime increases during matches
    - protective: Areas showing crime decreases during matches  
    - medium_risk: Areas with minimal but positive correlation
    
    Returns:
        dict: Stadium data with coordinates and risk classifications
    """
    return {
        'Arsenal': {
            'venue': 'Emirates Stadium',
            'lat': 51.5549, 'lon': -0.1084,
            'risk_type': 'high_risk',  # +9.4% burglary increase
            'evidence': 'Marginal significance (p=0.0937), 9.4% increase during matches'
        },
        'Chelsea': {
            'venue': 'Stamford Bridge', 
            'lat': 51.4816, 'lon': -0.1909,
            'risk_type': 'protective',  # -7.8% burglary decrease
            'evidence': 'Significant decrease (p=0.0415), protective effect during matches'
        },
        'Tottenham': {
            'venue': 'Tottenham Hotspur Stadium',
            'lat': 51.6040, 'lon': -0.0670,
            'risk_type': 'high_risk',  # +13.2% burglary increase
            'evidence': 'Highly significant (p=0.0147), 13.2% increase during matches'
        },
        'West Ham': {
            'venue': 'London Stadium',
            'lat': 51.5386, 'lon': -0.0164,
            'risk_type': 'medium_risk',  # +4.2% (not significant)
            'evidence': 'No significant effect (p=0.5351), minimal correlation'
        }
    }

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth.
    
    Uses the Haversine formula for accurate distance calculation.
    
    Args:
        lat1, lon1: Latitude and longitude of first point (decimal degrees)
        lat2, lon2: Latitude and longitude of second point (decimal degrees)
    
    Returns:
        float: Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers
    earth_radius_km = 6371
    return c * earth_radius_km

def calculate_stadium_risk_factor(ward_lat, ward_lon, stadiums):
    """
    Calculate stadium-based risk factor for a ward based on proximity to stadiums.
    
    Logic:
    - Find closest stadium to ward centroid
    - If within influence radius, apply stadium-specific risk multiplier
    - Multiple stadiums can influence the same ward (additive effect)
    
    Args:
        ward_lat, ward_lon: Ward center coordinates
        stadiums: Dictionary of stadium data
    
    Returns:
        tuple: (risk_factor, explanation_string)
    """
    risk_factor = 1.0
    influences = []
    
    for team, stadium in stadiums.items():
        distance = haversine_distance(
            ward_lat, ward_lon,
            stadium['lat'], stadium['lon']
        )
        
        if distance <= STADIUM_INFLUENCE_RADIUS_KM:
            multiplier = STADIUM_RISK_MULTIPLIER[stadium['risk_type']]
            # Apply distance-weighted influence (closer = stronger effect)
            weight = 1 - (distance / STADIUM_INFLUENCE_RADIUS_KM)
            weighted_multiplier = 1 + (multiplier - 1) * weight
            
            risk_factor *= weighted_multiplier
            influences.append(f"{team} ({distance:.1f}km)")
    
    explanation = f"Stadium influences: {'; '.join(influences)}" if influences else "No stadium influence"
    return risk_factor, explanation

# ============================================================================
# MAIN ALLOCATION LOGIC
# ============================================================================

def create_ward_centroids(lsoa_to_ward_df):
    """
    Create approximate ward centroids for distance calculations.
    
    Note: This is a simplified approach. In production, use actual 
    ward boundary shapefiles for precise centroids.
    
    Args:
        lsoa_to_ward_df: DataFrame with LSOA to ward mappings
    
    Returns:
        dict: Ward code to approximate coordinates mapping
    """
    # For this demonstration, we'll use approximate coordinates for major wards
    # In production, extract from shapefile centroids
    ward_centroids = {
        # Central London wards (approximate)
        'E05009288': (51.5074, -0.1278),  # City of London
        'E05000026': (51.5074, -0.1278),  # Westminster
        # Add more ward centroids as needed
    }
    
    # For wards without specific coordinates, use London center as default
    london_center = (51.5074, -0.1278)
    
    return ward_centroids, london_center

def enhanced_police_allocation():
    """
    Main function implementing enhanced police allocation with stadium factors.
    
    Process:
    1. Load base risk data, population, and geographic mappings
    2. Calculate stadium proximity risk factors for each ward
    3. Combine base risk with stadium risk factors
    4. Allocate police resources proportionally
    5. Generate detailed allocation report
    """
    
    print("="*80)
    print("ENHANCED POLICE ALLOCATION SYSTEM")
    print("Incorporating Stadium Proximity Risk Factors")
    print("="*80)
    
    # === LOAD BASE DATA ===
    print("\n1. Loading base datasets...")
    
    risk_df = pd.read_csv(risk_predictions_path)
    lsoa_to_ward_df = pd.read_excel(lsoa_to_ward_path)
    population_df = pd.read_excel(
        population_path,
        sheet_name='Mid-2021 LSOA 2021',
        skiprows=6,
        usecols=[2, 4],
        names=["LSOA11CD", "population"]
    )
    
    stadiums = define_london_stadiums()
    
    print(f"   ✓ Risk predictions: {len(risk_df)} LSOAs")
    print(f"   ✓ LSOA-Ward mappings: {len(lsoa_to_ward_df)} records")
    print(f"   ✓ Population data: {len(population_df)} LSOAs")
    print(f"   ✓ Stadium data: {len(stadiums)} stadiums loaded")
    
    # === DATA CLEANING ===
    print("\n2. Processing and merging data...")
    
    risk_df.rename(columns={"LSOA_code": "LSOA11CD"}, inplace=True)
    lsoa_to_ward_df = lsoa_to_ward_df[['LSOA11CD', 'WD21CD', 'WD21NM']]
    
    # Merge datasets
    merged_df = risk_df.merge(lsoa_to_ward_df, on="LSOA11CD", how="left")
    merged_df = merged_df.merge(population_df, on="LSOA11CD", how="left")
    merged_df.dropna(subset=["population"], inplace=True)
    
    print(f"   ✓ Merged dataset: {len(merged_df)} records")
    
    # === CALCULATE BASE WEIGHTED RISK ===
    merged_df["base_weighted_risk"] = merged_df["average_risk_score_2025"] * merged_df["population"]
    
    # === AGGREGATE TO WARD LEVEL ===
    print("\n3. Aggregating to ward level...")
    
    ward_risk_df = merged_df.groupby(['WD21CD', 'WD21NM']).agg({
        "base_weighted_risk": "sum",
        "population": "sum",
        "average_risk_score_2025": "mean"  # Keep average risk for reference
    }).reset_index()
    
    ward_risk_df["base_population_weighted_risk"] = (
        ward_risk_df["base_weighted_risk"] / ward_risk_df["population"]
    )
    
    print(f"   ✓ Ward-level data: {len(ward_risk_df)} wards")
    
    # === CALCULATE STADIUM RISK FACTORS ===
    print("\n4. Calculating stadium proximity risk factors...")
    
    # Get ward centroids (simplified approach)
    ward_centroids, london_center = create_ward_centroids(lsoa_to_ward_df)
    
    stadium_factors = []
    stadium_explanations = []
    
    for _, ward in ward_risk_df.iterrows():
        ward_code = ward['WD21CD']
        
        # Get ward centroid (use lookup or default to London center)
        centroid = ward_centroids.get(ward_code, london_center)
        
        # Calculate stadium risk factor
        risk_factor, explanation = calculate_stadium_risk_factor(
            centroid[0], centroid[1], stadiums
        )
        
        stadium_factors.append(risk_factor)
        stadium_explanations.append(explanation)
    
    ward_risk_df["stadium_risk_factor"] = stadium_factors
    ward_risk_df["stadium_explanation"] = stadium_explanations
    
    # === APPLY STADIUM ADJUSTMENTS ===
    print("\n5. Applying stadium risk adjustments...")
    
    ward_risk_df["enhanced_weighted_risk"] = (
        ward_risk_df["base_weighted_risk"] * ward_risk_df["stadium_risk_factor"]
    )
    
    ward_risk_df["enhanced_population_weighted_risk"] = (
        ward_risk_df["enhanced_weighted_risk"] / ward_risk_df["population"]
    )
    
    # === NORMALIZE AND ALLOCATE RESOURCES ===
    print("\n6. Allocating police resources...")
    
    # Normalize enhanced risk scores
    total_enhanced_risk = ward_risk_df["enhanced_population_weighted_risk"].sum()
    ward_risk_df["normalized_enhanced_risk"] = (
        ward_risk_df["enhanced_population_weighted_risk"] / total_enhanced_risk
    )
    
    # Calculate total available hours
    total_hours_available = len(ward_risk_df) * TOTAL_HOURS_PER_WARD
    
    # Allocate hours proportionally to enhanced risk
    ward_risk_df["allocated_hours"] = (
        ward_risk_df["normalized_enhanced_risk"] * total_hours_available
    )
    
    # Convert to officer count
    ward_risk_df["officers_needed"] = (
        ward_risk_df["allocated_hours"] / HOURS_PER_OFFICER
    ).round().astype(int)
    
    # === CALCULATE IMPACT METRICS ===
    print("\n7. Calculating allocation impacts...")
    
    # Compare base vs enhanced allocations
    total_base_risk = ward_risk_df["base_population_weighted_risk"].sum()
    ward_risk_df["base_normalized_risk"] = (
        ward_risk_df["base_population_weighted_risk"] / total_base_risk
    )
    ward_risk_df["base_allocated_hours"] = (
        ward_risk_df["base_normalized_risk"] * total_hours_available
    )
    ward_risk_df["base_officers_needed"] = (
        ward_risk_df["base_allocated_hours"] / HOURS_PER_OFFICER
    ).round().astype(int)
    
    # Calculate changes
    ward_risk_df["officer_change"] = (
        ward_risk_df["officers_needed"] - ward_risk_df["base_officers_needed"]
    )
    ward_risk_df["hour_change"] = (
        ward_risk_df["allocated_hours"] - ward_risk_df["base_allocated_hours"]
    )
    
    # === GENERATE SUMMARY STATISTICS ===
    print("\n8. Generating summary statistics...")
    
    # Stadium influence summary
    stadium_influenced_wards = ward_risk_df[ward_risk_df["stadium_risk_factor"] != 1.0]
    
    print(f"\nSTADIUM INFLUENCE SUMMARY:")
    print(f"   • Total wards analyzed: {len(ward_risk_df)}")
    print(f"   • Wards influenced by stadiums: {len(stadium_influenced_wards)}")
    print(f"   • Average stadium risk factor: {ward_risk_df['stadium_risk_factor'].mean():.3f}")
    
    if len(stadium_influenced_wards) > 0:
        print(f"\nTOP STADIUM-INFLUENCED WARDS:")
        top_influenced = stadium_influenced_wards.nlargest(5, 'stadium_risk_factor')
        for _, ward in top_influenced.iterrows():
            print(f"   • {ward['WD21NM']}: {ward['stadium_risk_factor']:.3f}x risk")
            print(f"     {ward['stadium_explanation']}")
    
    # Resource reallocation summary
    total_officer_changes = abs(ward_risk_df["officer_change"]).sum()
    wards_with_increases = len(ward_risk_df[ward_risk_df["officer_change"] > 0])
    wards_with_decreases = len(ward_risk_df[ward_risk_df["officer_change"] < 0])
    
    print(f"\nRESOURCE REALLOCATION SUMMARY:")
    print(f"   • Total officer reallocations: {total_officer_changes}")
    print(f"   • Wards receiving more officers: {wards_with_increases}")
    print(f"   • Wards receiving fewer officers: {wards_with_decreases}")
    
    # === PREPARE FINAL OUTPUT ===
    print("\n9. Preparing final allocation plan...")
    
    # Rename columns for output
    final_df = ward_risk_df.rename(columns={
        "WD21CD": "ward_code",
        "WD21NM": "ward_name"
    })
    
    # Select key columns for output
    output_columns = [
        "ward_code", "ward_name", "population",
        "base_population_weighted_risk", "stadium_risk_factor", 
        "enhanced_population_weighted_risk",
        "base_officers_needed", "officers_needed", "officer_change",
        "allocated_hours", "stadium_explanation"
    ]
    
    final_df = final_df[output_columns]
    
    # === DISPLAY RESULTS ===
    print("\n" + "="*80)
    print("ENHANCED ALLOCATION RESULTS")
    print("="*80)
    
    print(f"\nTOP 10 WARDS BY ENHANCED RISK:")
    top_risk_wards = final_df.nlargest(10, 'enhanced_population_weighted_risk')
    for i, (_, ward) in enumerate(top_risk_wards.iterrows(), 1):
        stadium_note = " (Stadium Influenced)" if ward['stadium_risk_factor'] != 1.0 else ""
        print(f"{i:2d}. {ward['ward_name']}: {ward['officers_needed']} officers{stadium_note}")
    
    print(f"\nWARDS WITH LARGEST OFFICER INCREASES:")
    increased_wards = final_df[final_df['officer_change'] > 0].nlargest(5, 'officer_change')
    for _, ward in increased_wards.iterrows():
        print(f"   • {ward['ward_name']}: +{ward['officer_change']} officers")
        print(f"     Reason: {ward['stadium_explanation']}")
    
    # === SAVE RESULTS ===
    output_filename = "enhanced_ward_allocation_plan.csv"
    final_df.to_csv(output_filename, index=False)
    
    print(f"\n✓ Enhanced allocation plan saved to: {output_filename}")
    print(f"✓ Analysis complete!")
    
    return final_df

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run enhanced allocation
    allocation_results = enhanced_police_allocation()
    
    # Display sample results
    print("\n" + "="*50)
    print("SAMPLE ALLOCATION RESULTS")
    print("="*50)
    
    sample_results = allocation_results.head(10)
    display_columns = ["ward_name", "enhanced_population_weighted_risk", "officers_needed", "officer_change"]
    print(sample_results[display_columns].to_string(index=False)) 