# Stadium-Enhanced Police Allocation Demo
# ====================================

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, asin

def define_london_stadiums():
    """Define stadiums with evidence-based risk factors"""
    return {
        'Arsenal': {
            'venue': 'Emirates Stadium',
            'lat': 51.5549, 'lon': -0.1084,
            'risk_type': 'high_risk',
            'evidence': '+9.4% burglary increase (p=0.0937)'
        },
        'Chelsea': {
            'venue': 'Stamford Bridge', 
            'lat': 51.4816, 'lon': -0.1909,
            'risk_type': 'protective',
            'evidence': '-7.8% burglary decrease (p=0.0415)'
        },
        'Tottenham': {
            'venue': 'Tottenham Hotspur Stadium',
            'lat': 51.6040, 'lon': -0.0670,
            'risk_type': 'high_risk',
            'evidence': '+13.2% burglary increase (p=0.0147)'
        },
        'West Ham': {
            'venue': 'London Stadium',
            'lat': 51.5386, 'lon': -0.0164,
            'risk_type': 'medium_risk',
            'evidence': '+4.2% increase (not significant)'
        }
    }

# Configuration
STADIUM_INFLUENCE_RADIUS_KM = 3.0
STADIUM_RISK_MULTIPLIER = {
    'high_risk': 1.15,      # +15% for high-risk areas
    'medium_risk': 1.05,    # +5% for medium-risk areas
    'protective': 0.92      # -8% for protective areas
}

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance using Haversine formula"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * 6371  # Earth radius in km

def calculate_stadium_risk_factor(ward_lat, ward_lon, stadiums):
    """Calculate stadium-based risk factor"""
    risk_factor = 1.0
    influences = []
    
    for team, stadium in stadiums.items():
        distance = haversine_distance(ward_lat, ward_lon, stadium['lat'], stadium['lon'])
        
        if distance <= STADIUM_INFLUENCE_RADIUS_KM:
            multiplier = STADIUM_RISK_MULTIPLIER[stadium['risk_type']]
            weight = 1 - (distance / STADIUM_INFLUENCE_RADIUS_KM)
            weighted_multiplier = 1 + (multiplier - 1) * weight
            risk_factor *= weighted_multiplier
            influences.append(f"{team} ({distance:.1f}km, {weighted_multiplier:.3f}x)")
    
    explanation = f"Stadium influences: {'; '.join(influences)}" if influences else "No stadium influence"
    return risk_factor, explanation

def enhanced_police_allocation_demo():
    """Demonstration of enhanced allocation system"""
    print("=" * 80)
    print("STADIUM-ENHANCED POLICE ALLOCATION DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Load available data
        print("\n1. Loading available datasets...")
        risk_df = pd.read_csv("monthly_risk_predictions_2025.csv")
        lsoa_to_ward_df = pd.read_excel("LSOA11_WD21_LAD21_EW_LU_V2.xlsx")
        
        print(f"   ✓ Risk predictions: {len(risk_df)} LSOAs")
        print(f"   ✓ LSOA-Ward mappings: {len(lsoa_to_ward_df)} records")
        
        # Create synthetic population data for demo
        unique_lsoas = risk_df["LSOA_code"].unique() if "LSOA_code" in risk_df.columns else []
        if len(unique_lsoas) == 0:
            unique_lsoas = risk_df.iloc[:, 0].unique()
        
        population_df = pd.DataFrame({
            "LSOA11CD": unique_lsoas,
            "population": np.random.randint(1000, 5000, len(unique_lsoas))
        })
        print(f"   ✓ Population data (synthetic): {len(population_df)} LSOAs")
        
        stadiums = define_london_stadiums()
        print(f"   ✓ Stadium data: {len(stadiums)} stadiums loaded")
        
        # Data processing
        print("\n2. Processing and merging data...")
        
        # Ensure consistent column naming
        if "LSOA_code" in risk_df.columns:
            risk_df.rename(columns={"LSOA_code": "LSOA11CD"}, inplace=True)
        
        # Select relevant columns
        lsoa_to_ward_df = lsoa_to_ward_df[['LSOA11CD', 'WD21CD', 'WD21NM']]
        
        # Merge datasets
        merged_df = risk_df.merge(lsoa_to_ward_df, on="LSOA11CD", how="left")
        merged_df = merged_df.merge(population_df, on="LSOA11CD", how="left")
        merged_df.dropna(subset=["population"], inplace=True)
        
        print(f"   ✓ Merged dataset: {len(merged_df)} records")
        
        # Get risk column name
        risk_col = "average_risk_score_2025" if "average_risk_score_2025" in merged_df.columns else merged_df.select_dtypes(include=[np.number]).columns[0]
        
        # Calculate weighted risk
        merged_df["weighted_risk"] = merged_df[risk_col] * merged_df["population"]
        
        # Aggregate to ward level
        print("\n3. Aggregating to ward level...")
        
        ward_risk_df = merged_df.groupby(['WD21CD', 'WD21NM']).agg({
            "weighted_risk": "sum",
            "population": "sum"
        }).reset_index()
        
        ward_risk_df["population_weighted_risk"] = ward_risk_df["weighted_risk"] / ward_risk_df["population"]
        
        print(f"   ✓ Ward-level data: {len(ward_risk_df)} wards")
        
        # Calculate stadium risk factors
        print("\n4. Calculating stadium proximity risk factors...")
        
        # Use London center as default for all wards (simplified for demo)
        london_center = (51.5074, -0.1278)
        
        stadium_factors = []
        stadium_explanations = []
        
        for _, ward in ward_risk_df.iterrows():
            risk_factor, explanation = calculate_stadium_risk_factor(
                london_center[0], london_center[1], stadiums
            )
            stadium_factors.append(risk_factor)
            stadium_explanations.append(explanation)
        
        ward_risk_df["stadium_risk_factor"] = stadium_factors
        ward_risk_df["stadium_explanation"] = stadium_explanations
        
        # Apply stadium adjustments
        print("\n5. Applying stadium risk adjustments...")
        
        ward_risk_df["enhanced_risk"] = ward_risk_df["population_weighted_risk"] * ward_risk_df["stadium_risk_factor"]
        
        # Normalize and allocate
        total_risk = ward_risk_df["enhanced_risk"].sum()
        ward_risk_df["normalized_risk"] = ward_risk_df["enhanced_risk"] / total_risk
        
        total_hours = len(ward_risk_df) * 200
        ward_risk_df["allocated_hours"] = ward_risk_df["normalized_risk"] * total_hours
        ward_risk_df["officers_needed"] = (ward_risk_df["allocated_hours"] / 2).round()
        
        # Calculate base allocation for comparison
        base_total = ward_risk_df["population_weighted_risk"].sum()
        ward_risk_df["base_normalized"] = ward_risk_df["population_weighted_risk"] / base_total
        ward_risk_df["base_officers"] = (ward_risk_df["base_normalized"] * total_hours / 2).round()
        ward_risk_df["officer_change"] = ward_risk_df["officers_needed"] - ward_risk_df["base_officers"]
        
        # Display results
        print("\n" + "=" * 80)
        print("ENHANCED ALLOCATION RESULTS")
        print("=" * 80)
        
        # Stadium influence summary
        avg_stadium_factor = ward_risk_df["stadium_risk_factor"].mean()
        influenced_wards = len(ward_risk_df[ward_risk_df["stadium_risk_factor"] != 1.0])
        
        print(f"\nSTADIUM INFLUENCE SUMMARY:")
        print(f"   • Total wards analyzed: {len(ward_risk_df)}")
        print(f"   • Average stadium risk factor: {avg_stadium_factor:.3f}")
        print(f"   • Note: Using London center coordinates for demonstration")
        
        print(f"\nSTADIUM EVIDENCE BASE:")
        for team, stadium in stadiums.items():
            risk_type = stadium['risk_type']
            multiplier = STADIUM_RISK_MULTIPLIER[risk_type]
            print(f"   • {team} ({stadium['venue']}): {multiplier:.3f}x multiplier")
            print(f"     Evidence: {stadium['evidence']}")
        
        print(f"\nTOP 10 WARDS BY ENHANCED RISK:")
        top_wards = ward_risk_df.nlargest(10, 'enhanced_risk')
        for i, (_, ward) in enumerate(top_wards.iterrows(), 1):
            print(f"{i:2d}. {ward['WD21NM']}: {ward['officers_needed']:.0f} officers")
        
        # Resource changes
        total_changes = abs(ward_risk_df["officer_change"]).sum()
        increases = len(ward_risk_df[ward_risk_df["officer_change"] > 0])
        decreases = len(ward_risk_df[ward_risk_df["officer_change"] < 0])
        
        print(f"\nRESOURCE REALLOCATION IMPACT:")
        print(f"   • Total officer reallocations: {total_changes:.0f}")
        print(f"   • Wards with increases: {increases}")
        print(f"   • Wards with decreases: {decreases}")
        
        # Save results
        output_columns = [
            "WD21CD", "WD21NM", "population", "population_weighted_risk",
            "stadium_risk_factor", "enhanced_risk", "base_officers", 
            "officers_needed", "officer_change", "stadium_explanation"
        ]
        
        final_df = ward_risk_df[output_columns].rename(columns={
            "WD21CD": "ward_code",
            "WD21NM": "ward_name"
        })
        
        final_df.to_csv("enhanced_allocation_demo_results.csv", index=False)
        print(f"\n✓ Results saved to: enhanced_allocation_demo_results.csv")
        
        return final_df
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nCreating basic demonstration with synthetic data...")
        
        # Create synthetic demonstration data
        synthetic_wards = [
            ("W001", "Central Ward", 2500, 0.8),
            ("W002", "North Ward", 3200, 0.6),
            ("W003", "Arsenal Area", 2800, 0.9),  # Near Arsenal
            ("W004", "Chelsea Area", 3100, 0.7),  # Near Chelsea
            ("W005", "East Ward", 2600, 0.5),
        ]
        
        ward_data = []
        stadiums = define_london_stadiums()
        
        for ward_code, ward_name, pop, base_risk in synthetic_wards:
            # Calculate stadium effect
            if "Arsenal" in ward_name:
                stadium_factor = STADIUM_RISK_MULTIPLIER['high_risk']
                explanation = "Arsenal Stadium (high risk area)"
            elif "Chelsea" in ward_name:
                stadium_factor = STADIUM_RISK_MULTIPLIER['protective']
                explanation = "Chelsea Stadium (protective effect)"
            else:
                stadium_factor = 1.0
                explanation = "No stadium influence"
            
            enhanced_risk = base_risk * stadium_factor
            base_officers = (base_risk * 50)  # Simplified calculation
            enhanced_officers = (enhanced_risk * 50)
            
            ward_data.append({
                "ward_code": ward_code,
                "ward_name": ward_name,
                "population": pop,
                "base_risk": base_risk,
                "stadium_risk_factor": stadium_factor,
                "enhanced_risk": enhanced_risk,
                "base_officers": base_officers,
                "officers_needed": enhanced_officers,
                "officer_change": enhanced_officers - base_officers,
                "stadium_explanation": explanation
            })
        
        demo_df = pd.DataFrame(ward_data)
        
        print("\nSYNTHETIC DEMONSTRATION RESULTS:")
        print("=" * 50)
        
        for _, ward in demo_df.iterrows():
            change = ward['officer_change']
            change_str = f"+{change:.0f}" if change > 0 else f"{change:.0f}"
            print(f"{ward['ward_name']}: {ward['officers_needed']:.0f} officers ({change_str})")
            print(f"   {ward['stadium_explanation']}")
        
        demo_df.to_csv("synthetic_demo_results.csv", index=False)
        print(f"\n✓ Synthetic demo saved to: synthetic_demo_results.csv")
        
        return demo_df

if __name__ == "__main__":
    results = enhanced_police_allocation_demo()
    print("\n" + "=" * 50)
    print("DEMONSTRATION COMPLETE")
    print("This system shows how stadium proximity can be")
    print("integrated into police allocation decisions using")
    print("empirical evidence from crime pattern analysis.")
    print("=" * 50) 