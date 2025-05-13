import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from folium.features import GeoJsonTooltip
from folium.plugins import HeatMap, MarkerCluster
import glob
from datetime import datetime, timedelta
from shapely.geometry import Point, Polygon
import branca.colormap as cm
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Set paths
DATA_DIR = "."
SHAPEFILE_DIR = "Model/LB_shp"
RESULTS_DIR = "Model/results"
OUTPUT_DIR = "Model/police_allocation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define data models
class Ward:
    def __init__(self, id, name, geometry):
        self.id = id
        self.name = name
        self.geometry = geometry
        self.lsoas = []  # List of LSOA IDs contained in ward
        self.max_officers = 100  # Max officers per ward
        self.available_hours = 200  # 2hrs * 100 officers

class LSOA:
    def __init__(self, id, name, risk_score, geometry):
        self.id = id
        self.name = name
        self.risk_score = risk_score
        self.geometry = geometry
        self.ward_id = None  # Will be set when mapped to a ward
        self.temporal_pattern = {}  # Hour -> risk weight

class AllocationPlan:
    def __init__(self, ward_id, ward_name, date):
        self.ward_id = ward_id
        self.ward_name = ward_name
        self.date = date
        self.time_blocks = {}  # (start_hour, end_hour) -> officer_count
        self.targeted_lsoas = []  # (lsoa_id, lsoa_name, allocation_weight)

# Load LSOA boundaries from shapefiles
def load_lsoa_boundaries():
    print("Loading LSOA boundaries...")
    shp_files = glob.glob(os.path.join(SHAPEFILE_DIR, "*.shp"))
    
    all_boundaries = []
    for shp_file in shp_files:
        borough_name = os.path.basename(shp_file).replace(".shp", "")
        try:
            gdf = gpd.read_file(shp_file)
            
            # Add LSOA code with consistent naming
            if 'lsoa21cd' in gdf.columns:
                gdf['LSOA_code'] = gdf['lsoa21cd']
                
                # Add borough name if not present
                if 'borough' not in gdf.columns:
                    gdf['borough'] = borough_name
                
                # Add LSOA name if available, otherwise use code
                if 'lsoa21nm' in gdf.columns:
                    gdf['LSOA_name'] = gdf['lsoa21nm']
                else:
                    gdf['LSOA_name'] = gdf['LSOA_code']
                
                all_boundaries.append(gdf)
                print(f"Added {len(gdf)} LSOAs from {borough_name}")
            else:
                print(f"Warning: {borough_name} shapefile does not contain LSOA information.")
        except Exception as e:
            print(f"Error loading {shp_file}: {e}")
    
    if not all_boundaries:
        raise ValueError("No valid boundary files found.")
    
    london_boundaries = pd.concat(all_boundaries, ignore_index=True)
    print(f"Successfully loaded {len(london_boundaries)} LSOA boundaries")
    return london_boundaries

# Generate synthetic ward boundaries if not available
def generate_ward_boundaries(lsoa_boundaries, n_wards=32):
    """
    Generate ward boundaries by clustering LSOAs.
    This is a fallback if actual ward boundaries aren't available.
    """
    print("Generating synthetic ward boundaries...")
    
    # Simplify by grouping by borough first
    lsoa_by_borough = lsoa_boundaries.groupby('borough')
    
    # Initialize ward data
    ward_data = []
    ward_id = 0
    
    # For each borough, create wards
    for borough_name, borough_lsoas in lsoa_by_borough:
        # Calculate number of wards for this borough based on size
        borough_size = len(borough_lsoas)
        n_borough_wards = max(1, int(borough_size / len(lsoa_boundaries) * n_wards))
        
        # Determine clusters within borough (simplified - adjacent LSOAs)
        if borough_size <= n_borough_wards:
            # One LSOA per ward
            for idx, lsoa in borough_lsoas.iterrows():
                ward_id += 1
                ward_name = f"Ward_{ward_id}_{borough_name}"
                
                ward_data.append({
                    'ward_id': f"W{ward_id:03d}",
                    'ward_name': ward_name,
                    'borough': borough_name,
                    'geometry': lsoa.geometry,
                    'lsoas': [lsoa['LSOA_code']]
                })
        else:
            # Group LSOAs by proximity
            grouped_lsoas = []
            remaining = list(borough_lsoas.index)
            
            while len(grouped_lsoas) < n_borough_wards and remaining:
                group = [remaining.pop(0)]
                seed_geometry = borough_lsoas.loc[group[0]].geometry
                
                # Find adjacent LSOAs
                i = 0
                while i < len(remaining) and len(group) < borough_size / n_borough_wards:
                    idx = remaining[i]
                    if borough_lsoas.loc[idx].geometry.touches(seed_geometry):
                        group.append(idx)
                        seed_geometry = seed_geometry.union(borough_lsoas.loc[idx].geometry)
                        remaining.remove(idx)
                    else:
                        i += 1
                
                grouped_lsoas.append(group)
            
            # Assign any remaining LSOAs to nearest group
            if remaining:
                for idx in remaining:
                    min_dist = float('inf')
                    best_group = 0
                    
                    for i, group in enumerate(grouped_lsoas):
                        group_geom = borough_lsoas.loc[group[0]].geometry
                        dist = borough_lsoas.loc[idx].geometry.distance(group_geom)
                        if dist < min_dist:
                            min_dist = dist
                            best_group = i
                    
                    grouped_lsoas[best_group].append(idx)
            
            # Create wards from groups
            for group in grouped_lsoas:
                ward_id += 1
                ward_name = f"Ward_{ward_id}_{borough_name}"
                
                # Union of geometries for the ward
                ward_geom = borough_lsoas.loc[group[0]].geometry
                for idx in group[1:]:
                    ward_geom = ward_geom.union(borough_lsoas.loc[idx].geometry)
                
                # List of LSOA codes in this ward
                ward_lsoas = [borough_lsoas.loc[idx]['LSOA_code'] for idx in group]
                
                ward_data.append({
                    'ward_id': f"W{ward_id:03d}",
                    'ward_name': ward_name,
                    'borough': borough_name,
                    'geometry': ward_geom,
                    'lsoas': ward_lsoas
                })
    
    # Create GeoDataFrame from ward data
    wards_gdf = gpd.GeoDataFrame(ward_data)
    print(f"Generated {len(wards_gdf)} ward boundaries")
    return wards_gdf

# Map LSOAs to wards
def map_lsoas_to_wards(lsoa_boundaries, ward_boundaries):
    """
    Create mapping between LSOAs and wards based on spatial relationships.
    Returns a dictionary mapping ward_id -> list of LSOA_ids
    """
    print("Mapping LSOAs to wards...")
    
    # Initialize mapping
    lsoa_ward_mapping = {}
    ward_lsoa_mapping = {row['ward_id']: [] for _, row in ward_boundaries.iterrows()}
    
    # For each LSOA, find the ward it belongs to
    for idx, lsoa in lsoa_boundaries.iterrows():
        lsoa_centroid = lsoa.geometry.centroid
        
        # Find containing ward
        for ward_idx, ward in ward_boundaries.iterrows():
            if ward.geometry.contains(lsoa_centroid):
                lsoa_ward_mapping[lsoa['LSOA_code']] = ward['ward_id']
                ward_lsoa_mapping[ward['ward_id']].append(lsoa['LSOA_code'])
                break
    
    # For any unassigned LSOAs, use nearest ward
    unassigned = []
    for idx, lsoa in lsoa_boundaries.iterrows():
        if lsoa['LSOA_code'] not in lsoa_ward_mapping:
            unassigned.append(lsoa)
    
    if unassigned:
        print(f"Assigning {len(unassigned)} LSOAs to nearest ward...")
        for lsoa in unassigned:
            nearest_ward = None
            min_distance = float('inf')
            
            for ward_idx, ward in ward_boundaries.iterrows():
                distance = lsoa.geometry.distance(ward.geometry)
                if distance < min_distance:
                    min_distance = distance
                    nearest_ward = ward
            
            lsoa_ward_mapping[lsoa['LSOA_code']] = nearest_ward['ward_id']
            ward_lsoa_mapping[nearest_ward['ward_id']].append(lsoa['LSOA_code'])
    
    # Check mapping coverage
    mapped_lsoas = len(lsoa_ward_mapping)
    total_lsoas = len(lsoa_boundaries)
    print(f"Mapped {mapped_lsoas} out of {total_lsoas} LSOAs to wards ({mapped_lsoas/total_lsoas*100:.1f}%)")
    
    # Check if any wards have no LSOAs
    empty_wards = sum(1 for lsoas in ward_lsoa_mapping.values() if not lsoas)
    if empty_wards:
        print(f"Warning: {empty_wards} wards have no assigned LSOAs")
    
    return ward_lsoa_mapping, lsoa_ward_mapping

# Calculate ward risk scores
def calculate_ward_risk_scores(risk_predictions, lsoa_boundaries, ward_lsoa_mapping):
    """
    Aggregate LSOA risk scores to ward level.
    Returns a dictionary of ward_id -> ward risk data
    """
    print("Calculating ward risk scores...")
    
    # Merge risk predictions with LSOA boundaries
    lsoa_data = lsoa_boundaries.merge(risk_predictions, on='LSOA_code', how='inner')
    
    # Calculate ward risk scores
    ward_risks = {}
    for ward_id, lsoa_ids in ward_lsoa_mapping.items():
        # Filter to LSOAs in this ward that have risk predictions
        ward_lsoas = lsoa_data[lsoa_data['LSOA_code'].isin(lsoa_ids)]
        
        if len(ward_lsoas) == 0:
            print(f"Warning: Ward {ward_id} has no risk data")
            continue
        
        # Calculate ward-level statistics
        avg_risk = ward_lsoas['risk_score'].mean()
        max_risk = ward_lsoas['risk_score'].max()
        total_risk = ward_lsoas['risk_score'].sum()
        
        # Create list of LSOAs with their risk scores
        lsoa_risks = []
        for _, lsoa in ward_lsoas.iterrows():
            lsoa_risks.append({
                'id': lsoa['LSOA_code'],
                'name': lsoa['LSOA_name'] if 'LSOA_name' in lsoa else lsoa['LSOA_code'],
                'risk_score': lsoa['risk_score'],
                'predicted_risk': lsoa['predicted_risk'] if 'predicted_risk' in lsoa else lsoa['risk_score'],
            })
        
        # Sort LSOAs by risk score
        lsoa_risks.sort(key=lambda x: x['risk_score'], reverse=True)
        
        ward_risks[ward_id] = {
            'ward_id': ward_id,
            'avg_risk': avg_risk,
            'max_risk': max_risk,
            'total_risk': total_risk,
            'num_lsoas': len(ward_lsoas),
            'lsoas': lsoa_risks
        }
    
    print(f"Calculated risk scores for {len(ward_risks)} wards")
    return ward_risks

# Generate synthetic temporal patterns if not available
def generate_synthetic_temporal_patterns(lsoa_boundaries, risk_predictions=None):
    """
    Generate synthetic temporal patterns for LSOAs.
    This is a fallback if real temporal data isn't available.
    """
    print("Generating synthetic temporal patterns...")
    
    # Create dictionary to store temporal patterns
    temporal_patterns = {}
    
    # Define typical burglary pattern across hours of day
    # Higher values in evening/night hours based on criminology research
    base_pattern = {
        0: 0.7,   # Midnight
        1: 0.6,
        2: 0.5,
        3: 0.4,
        4: 0.3,
        5: 0.2,
        6: 0.1,   # Early morning (low activity)
        7: 0.2,
        8: 0.3,
        9: 0.4,
        10: 0.5,
        11: 0.6,
        12: 0.7,  # Noon
        13: 0.8,
        14: 0.9,
        15: 1.0,   # Afternoon peak (schools out)
        16: 1.1,
        17: 1.2,
        18: 1.3,
        19: 1.5,  # Evening peak (people returning from work)
        20: 1.7,
        21: 1.6,  # Prime time for burglaries
        22: 1.4,
        23: 1.0   # Late night
    }
    
    # Define day of week pattern (higher on weekends)
    dow_pattern = {
        0: 0.9,  # Monday
        1: 0.8,  # Tuesday
        2: 0.85, # Wednesday
        3: 0.9,  # Thursday
        4: 1.1,  # Friday
        5: 1.2,  # Saturday
        6: 1.0   # Sunday
    }
    
    # If risk predictions are available, use them to modulate patterns
    if risk_predictions is not None:
        # Merge with LSOA boundaries
        lsoa_data = lsoa_boundaries.merge(risk_predictions, on='LSOA_code', how='inner')
        
        # Scale risk scores to [0.5, 1.5] range to modulate patterns
        if 'risk_score' in lsoa_data.columns:
            scaler = MinMaxScaler(feature_range=(0.5, 1.5))
            risk_factors = scaler.fit_transform(lsoa_data[['risk_score']]).flatten()
            
            # Generate temporal patterns for each LSOA
            for i, (_, lsoa) in enumerate(lsoa_data.iterrows()):
                risk_factor = risk_factors[i]
                
                # Create a slightly randomized pattern based on risk score
                lsoa_pattern = {}
                
                # For each day of week
                for day in range(7):
                    day_factor = dow_pattern[day] * risk_factor
                    
                    # For each hour
                    for hour in range(24):
                        # Apply small random variation
                        random_factor = np.random.normal(1.0, 0.1)
                        hour_weight = base_pattern[hour] * day_factor * random_factor
                        
                        # Store in pattern dictionary
                        time_key = (day, hour)
                        lsoa_pattern[time_key] = max(0.1, hour_weight)  # Ensure positive weight
                
                temporal_patterns[lsoa['LSOA_code']] = lsoa_pattern
        else:
            print("Warning: No risk_score column found, using uniform risk factors")
            for _, lsoa in lsoa_boundaries.iterrows():
                lsoa_pattern = {}
                risk_factor = np.random.uniform(0.7, 1.3)  # Random risk factor
                
                # Generate pattern
                for day in range(7):
                    day_factor = dow_pattern[day] * risk_factor
                    for hour in range(24):
                        random_factor = np.random.normal(1.0, 0.1)
                        hour_weight = base_pattern[hour] * day_factor * random_factor
                        time_key = (day, hour)
                        lsoa_pattern[time_key] = max(0.1, hour_weight)
                
                temporal_patterns[lsoa['LSOA_code']] = lsoa_pattern
    else:
        # No risk predictions, use random risk factors
        for _, lsoa in lsoa_boundaries.iterrows():
            lsoa_pattern = {}
            risk_factor = np.random.uniform(0.7, 1.3)  # Random risk factor
            
            # Generate pattern
            for day in range(7):
                day_factor = dow_pattern[day] * risk_factor
                for hour in range(24):
                    random_factor = np.random.normal(1.0, 0.1)
                    hour_weight = base_pattern[hour] * day_factor * random_factor
                    time_key = (day, hour)
                    lsoa_pattern[time_key] = max(0.1, hour_weight)
            
            temporal_patterns[lsoa['LSOA_code']] = lsoa_pattern
    
    print(f"Generated temporal patterns for {len(temporal_patterns)} LSOAs")
    return temporal_patterns

# Calculate hourly risks using temporal patterns
def calculate_hourly_risks(lsoas, temporal_patterns, target_date):
    """
    Calculate hourly risk scores for a ward based on LSOA risk scores and temporal patterns.
    """
    # Get day of week and initialize hourly risks
    day_of_week = target_date.weekday()  # 0=Monday, 6=Sunday
    hourly_risks = {hour: 0.0 for hour in range(24)}
    
    # For each LSOA in the ward
    for lsoa in lsoas:
        lsoa_id = lsoa['id']
        base_risk = lsoa['risk_score']
        
        # If temporal pattern exists for this LSOA
        if lsoa_id in temporal_patterns:
            lsoa_pattern = temporal_patterns[lsoa_id]
            
            # For each hour of the day
            for hour in range(24):
                time_key = (day_of_week, hour)
                if time_key in lsoa_pattern:
                    hour_factor = lsoa_pattern[time_key]
                    hourly_risks[hour] += base_risk * hour_factor
        else:
            # No temporal pattern, distribute risk evenly
            for hour in range(24):
                hourly_risks[hour] += base_risk / 24
    
    return hourly_risks

# Find optimal 2-hour blocks for police allocation
def select_best_hours(hourly_risks, block_size=2, num_blocks=4, start_hour=6, end_hour=22):
    """
    Select the best 2-hour blocks for police allocation.
    Returns a list of (start_hour, end_hour) tuples.
    """
    # Filter to patrol hours (6:00-22:00)
    patrol_hours = {h: hourly_risks[h] for h in range(start_hour, end_hour)}
    
    # Calculate risk for each possible block
    block_risks = {}
    for start in range(start_hour, end_hour - block_size + 1):
        end = start + block_size
        block_risk = sum(patrol_hours[h] for h in range(start, end))
        block_risks[(start, end)] = block_risk
    
    # Sort blocks by risk and select top ones
    sorted_blocks = sorted(block_risks.items(), key=lambda x: x[1], reverse=True)
    best_blocks = []
    
    # Select non-overlapping blocks
    remaining_blocks = sorted_blocks.copy()
    while len(best_blocks) < num_blocks and remaining_blocks:
        # Get highest risk remaining block
        best_block = remaining_blocks.pop(0)
        best_blocks.append(best_block)
        
        # Remove overlapping blocks
        remaining_blocks = [b for b in remaining_blocks if 
                           not (b[0][0] < best_block[0][1] and b[0][1] > best_block[0][0])]
    
    return best_blocks

# Allocate officers based on risk scores
def allocate_officers_to_lsoas(lsoas, max_officers=100):
    """
    Allocate officers to LSOAs based on risk scores.
    Returns a list of (lsoa_id, lsoa_name, officer_count) tuples.
    """
    # Calculate total risk
    total_risk = sum(lsoa['risk_score'] for lsoa in lsoas)
    
    # Allocate officers proportionally to risk
    allocations = []
    remaining_officers = max_officers
    
    for lsoa in lsoas:
        # Calculate proportion of risk
        if total_risk > 0:
            proportion = lsoa['risk_score'] / total_risk
            officer_count = max(1, int(proportion * max_officers))
        else:
            # If no risk data, allocate equally
            officer_count = max_officers // len(lsoas)
        
        # Ensure we don't exceed remaining officers
        officer_count = min(officer_count, remaining_officers)
        
        if officer_count > 0:
            allocations.append((lsoa['id'], lsoa['name'], officer_count))
            remaining_officers -= officer_count
        
        if remaining_officers <= 0:
            break
    
    # If we have remaining officers, allocate them to highest risk areas
    if remaining_officers > 0 and allocations:
        sorted_allocations = sorted(allocations, key=lambda x: lsoas[next(i for i, lsoa in enumerate(lsoas) if lsoa['id'] == x[0])]['risk_score'], reverse=True)
        
        i = 0
        while remaining_officers > 0 and i < len(sorted_allocations):
            lsoa_id, lsoa_name, officers = sorted_allocations[i]
            sorted_allocations[i] = (lsoa_id, lsoa_name, officers + 1)
            remaining_officers -= 1
            i = (i + 1) % len(sorted_allocations)
        
        # Update original allocations
        allocations_dict = {lsoa_id: (lsoa_id, lsoa_name, officers) for lsoa_id, lsoa_name, officers in sorted_allocations}
        allocations = [allocations_dict[lsoa_id] for lsoa_id, _, _ in allocations]
    
    return allocations

# Generate allocation plan for wards
def generate_allocation_plans(ward_risks, temporal_patterns, allocation_date, ward_boundaries):
    """
    Generate police allocation plans for wards.
    """
    print(f"Generating allocation plans for {allocation_date.strftime('%Y-%m-%d')}...")
    
    # Initialize allocation plans
    allocation_plans = {}
    
    # For each ward with risk data
    for ward_id, risk_data in ward_risks.items():
        # Get ward name
        ward_name = ward_id
        for _, ward in ward_boundaries.iterrows():
            if ward['ward_id'] == ward_id:
                ward_name = ward['ward_name']
                break
        
        # Calculate hourly risks
        hourly_risks = calculate_hourly_risks(risk_data['lsoas'], temporal_patterns, allocation_date)
        
        # Select best time blocks
        best_blocks = select_best_hours(hourly_risks)
        
        # Create allocation plan
        plan = AllocationPlan(ward_id, ward_name, allocation_date)
        
        # Set time blocks
        for (start_hour, end_hour), risk_score in best_blocks:
            plan.time_blocks[(start_hour, end_hour)] = 100  # Assign all 100 officers to each block
        
        # Allocate officers to LSOAs
        plan.targeted_lsoas = allocate_officers_to_lsoas(risk_data['lsoas'])
        
        allocation_plans[ward_id] = plan
    
    print(f"Generated allocation plans for {len(allocation_plans)} wards")
    return allocation_plans

# Create allocation summary table
def create_allocation_summary(allocation_plans, ward_risks):
    """
    Create a summary table of the allocation plans.
    """
    # Initialize summary data
    summary_data = []
    
    # For each ward
    for ward_id, plan in allocation_plans.items():
        # Get ward risk data
        risk_data = ward_risks.get(ward_id, {})
        avg_risk = risk_data.get('avg_risk', 0)
        max_risk = risk_data.get('max_risk', 0)
        
        # Count total officer-hours
        total_officer_hours = sum(100 * 2 for _ in plan.time_blocks)
        
        # Get top 3 highest risk LSOAs
        top_lsoas = [f"{lsoa_name} ({officers} officers)" 
                    for _, lsoa_name, officers in sorted(plan.targeted_lsoas, key=lambda x: x[2], reverse=True)[:3]]
        
        # Get patrol times
        patrol_times = [f"{start:02d}:00-{end:02d}:00" for (start, end) in plan.time_blocks.keys()]
        
        # Add to summary
        summary_data.append({
            'Ward ID': ward_id,
            'Ward Name': plan.ward_name,
            'Average Risk': avg_risk,
            'Max Risk': max_risk,
            'Officer-Hours': total_officer_hours,
            'Patrol Times': ", ".join(patrol_times),
            'Top Target LSOAs': ", ".join(top_lsoas)
        })
    
    # Convert to DataFrame and sort by risk
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Average Risk', ascending=False)
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, f"allocation_summary_{allocation_plans[next(iter(allocation_plans))].date.strftime('%Y%m%d')}.csv")
    summary_df.to_csv(csv_path, index=False)
    
    print(f"Allocation summary saved to {csv_path}")
    return summary_df

# Create patrol time table
def create_patrol_time_table(allocation_plans):
    """
    Create a table showing all patrol times by ward.
    """
    # Get unique patrol time blocks
    all_blocks = set()
    for plan in allocation_plans.values():
        all_blocks.update(plan.time_blocks.keys())
    
    # Sort time blocks
    time_blocks = sorted(all_blocks)
    
    # Initialize table data
    table_data = []
    
    # For each ward
    for ward_id, plan in allocation_plans.items():
        row = {'Ward': plan.ward_name or ward_id}
        
        # Add columns for each time block
        for start_hour, end_hour in time_blocks:
            block_key = f"{start_hour:02d}:00-{end_hour:02d}:00"
            row[block_key] = plan.time_blocks.get((start_hour, end_hour), 0)
        
        table_data.append(row)
    
    # Convert to DataFrame
    table_df = pd.DataFrame(table_data)
    
    # Sort by ward name
    table_df = table_df.sort_values('Ward')
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, f"patrol_times_{allocation_plans[next(iter(allocation_plans))].date.strftime('%Y%m%d')}.csv")
    table_df.to_csv(csv_path, index=False)
    
    print(f"Patrol time table saved to {csv_path}")
    return table_df

def create_allocation_map(allocation_plans, ward_boundaries, lsoa_boundaries, lsoa_ward_mapping, risk_predictions):
    """Reliable version for displaying allocation"""
    print("Creating simplified allocation map...")
    
    # Create base map
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles='CartoDB positron')
    
    # Get allocation date
    date_str = allocation_plans[next(iter(allocation_plans))].date.strftime('%Y%m%d')
    
    # Load CSV data instead of attempting GeoJSON conversion
    csv_path = os.path.join(OUTPUT_DIR, f"allocation_summary_{date_str}.csv")
    allocation_data = pd.read_csv(csv_path)
    
    # Add circles for each ward
    for _, ward in allocation_data.iterrows():
        # Extract coordinates - use average for London boroughs if needed
        lat, lon = 51.5074, -0.1278  # Default to London center
        
        # Get borough name from ward name if possible
        borough = ward['Ward Name'].split('_')[-1] if '_' in ward['Ward Name'] else ward['Ward Name']
        
        # Add larger circle for ward
        folium.CircleMarker(
            location=[lat + (hash(ward['Ward ID']) % 100) / 1000, 
                     lon + (hash(ward['Ward ID']) % 100) / 1000],  # Slight offset for each ward
            radius=30,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.2,
            tooltip=f"Ward: {ward['Ward Name']}<br>Risk: {ward['Average Risk']:.2f}"
        ).add_to(m)
        
        # Add circle for officer allocation
        folium.CircleMarker(
            location=[lat + (hash(ward['Ward ID']) % 100) / 1000, 
                     lon + (hash(ward['Ward ID']) % 100) / 1000],
            radius=10,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.6,
            tooltip=f"Officers: 100<br>Patrol Times: {ward['Patrol Times']}"
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; background-color: white;
                border-radius: 5px; padding: 10px; opacity: 0.8; font-family: Arial;">
        <p><strong>Burglary Risk</strong></p>
        <p><i style="background: #bd0026; width: 15px; height: 15px; display: inline-block;"></i> Very High</p>
        <p><i style="background: #f03b20; width: 15px; height: 15px; display: inline-block;"></i> High</p>
        <p><i style="background: #fd8d3c; width: 15px; height: 15px; display: inline-block;"></i> Medium</p>
        <p><i style="background: #fecc5c; width: 15px; height: 15px; display: inline-block;"></i> Low</p>
        <p><i style="background: #ffffb2; width: 15px; height: 15px; display: inline-block;"></i> Very Low</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    output_path = os.path.join(OUTPUT_DIR, f"police_allocation_map_simple_{date_str}.html")
    m.save(output_path)
    
    print(f"Simplified map saved to {output_path}")
    return output_path

def create_static_allocation_charts():
    """Create static visualizations from allocation data"""
    allocation_csv = os.path.join(OUTPUT_DIR, "allocation_summary_20250514.csv")
    allocation_data = pd.read_csv(allocation_csv)
    
    # Create bar chart of ward risk levels
    plt.figure(figsize=(15, 8))
    bars = plt.bar(allocation_data['Ward Name'], allocation_data['Average Risk'])
    plt.xticks(rotation=90)
    plt.xlabel('Ward')
    plt.ylabel('Risk Score')
    plt.title('Burglary Risk by Ward')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ward_risk_scores.png'))
    
    # Create visualization of patrol times
    patrol_csv = os.path.join(OUTPUT_DIR, "patrol_times_20250514.csv")
    patrol_data = pd.read_csv(patrol_csv)
    
    # Create heatmap of patrol times
    plt.figure(figsize=(15, 10))
    patrol_matrix = patrol_data.set_index('Ward').T
    sns.heatmap(patrol_matrix, cmap='Blues', cbar_kws={'label': 'Officers Allocated'})
    plt.title('Patrol Time Allocation by Ward')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'patrol_time_heatmap.png'))
    
    return os.path.join(OUTPUT_DIR, 'ward_risk_scores.png'), os.path.join(OUTPUT_DIR, 'patrol_time_heatmap.png')

# Main function
def main():
    try:

        # Set allocation date (default to tomorrow)
        allocation_date = datetime.now() + timedelta(days=1)
        
        print(f"Police resource allocation planning for {allocation_date.strftime('%Y-%m-%d')}")
        print("-" * 80)
        
        # Load risk predictions from existing model
        predictions_file = os.path.join(RESULTS_DIR, "risk_predictions.csv")
        if os.path.exists(predictions_file):
            risk_predictions = pd.read_csv(predictions_file)
            print(f"Loaded risk predictions for {len(risk_predictions)} LSOAs")
        else:
            print(f"Warning: Risk predictions file not found at {predictions_file}")
            print("Generating synthetic risk data...")
            
            # Create synthetic risk data
            lsoa_boundaries = load_lsoa_boundaries()
            risk_predictions = pd.DataFrame({
                'LSOA_code': lsoa_boundaries['LSOA_code'],
                'risk_score': np.random.uniform(0, 100, size=len(lsoa_boundaries))
            })
        
        # Load LSOA boundaries
        lsoa_boundaries = load_lsoa_boundaries()
        
        # Generate ward boundaries
        # Try to load from predefined file first
        ward_file = os.path.join(SHAPEFILE_DIR, "London_Ward.shp")
        if os.path.exists(ward_file):
            try:
                ward_boundaries = gpd.read_file(ward_file)
                print(f"Loaded ward boundaries from {ward_file}")
                
                # Ensure ward ID and name fields exist
                if 'ward_id' not in ward_boundaries.columns:
                    ward_boundaries['ward_id'] = [f"W{i:03d}" for i in range(len(ward_boundaries))]
                if 'ward_name' not in ward_boundaries.columns:
                    ward_boundaries['ward_name'] = ward_boundaries.index.astype(str).map(lambda x: f"Ward_{x}")
            except Exception as e:
                print(f"Error loading ward file: {e}")
                print("Generating synthetic ward boundaries...")
                ward_boundaries = generate_ward_boundaries(lsoa_boundaries)
        else:
            print("Ward boundary file not found, generating synthetic boundaries...")
            ward_boundaries = generate_ward_boundaries(lsoa_boundaries)
        
        # Map LSOAs to wards
        ward_lsoa_mapping, lsoa_ward_mapping = map_lsoas_to_wards(lsoa_boundaries, ward_boundaries)
        
        # Calculate ward risk scores
        ward_risks = calculate_ward_risk_scores(risk_predictions, lsoa_boundaries, ward_lsoa_mapping)
        
        # Generate temporal patterns
        # Try to load from predefined file first
        temporal_file = os.path.join(RESULTS_DIR, "temporal_patterns.pkl")
        if os.path.exists(temporal_file):
            try:
                import pickle
                with open(temporal_file, 'rb') as f:
                    temporal_patterns = pickle.load(f)
                print(f"Loaded temporal patterns from {temporal_file}")
            except Exception as e:
                print(f"Error loading temporal patterns: {e}")
                print("Generating synthetic temporal patterns...")
                temporal_patterns = generate_synthetic_temporal_patterns(lsoa_boundaries, risk_predictions)
        else:
            print("Temporal patterns file not found, generating synthetic patterns...")
            temporal_patterns = generate_synthetic_temporal_patterns(lsoa_boundaries, risk_predictions)
        
        # Generate allocation plans
        allocation_plans = generate_allocation_plans(ward_risks, temporal_patterns, allocation_date, ward_boundaries)
        
        # Create allocation summary
        allocation_summary = create_allocation_summary(allocation_plans, ward_risks)
        
        # Create patrol time table
        patrol_times = create_patrol_time_table(allocation_plans)
        
        # Create interactive allocation map
        map_path = create_allocation_map(allocation_plans, ward_boundaries, lsoa_boundaries, lsoa_ward_mapping, risk_predictions)
        
        print("\nAllocation planning completed successfully!")
        print(f"Output files saved to {OUTPUT_DIR}")
        print(f"Interactive map: {map_path}")
        # Add visualization with static charts
        
        # Return allocation data
        return {
            'allocation_plans': allocation_plans,
            'ward_risks': ward_risks,
            'allocation_summary': allocation_summary,
            'patrol_times': patrol_times,
            'map_path': map_path
        }

    except Exception as e:
        print(f"Error in allocation planning: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()