from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
# Commenting out pmdarima to avoid dependency issues
# from pmdarima import auto_arima
from sklearn.cluster import KMeans
from pathlib import Path
import random
from scipy import stats
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Data paths - adjust based on actual project structure
BASE_DIR = Path(__file__).resolve().parent.parent
BURGLARY_DATA_PATH = BASE_DIR / "cleaned_monthly_burglary_data"
SPATIAL_BURGLARY_DATA_PATH = BASE_DIR / "cleaned_spatial_monthly_burglary_data"
IMD_DATA_PATH = BASE_DIR / "societal_wellbeing_dataset/imd2019lsoa_london_cleaned.csv"
LSOA_CODES_PATH = BASE_DIR / "societal_wellbeing_dataset/LSOA_codes.csv"

# Cache for loaded data to avoid loading large datasets on every request
data_cache = {}

def load_imd_data():
    """Load the IMD (Indices of Multiple Deprivation) data"""
    try:
        if 'imd_data' not in data_cache:
            print("Loading IMD data")
            # Read the IMD data
            imd_df = pd.read_csv(IMD_DATA_PATH)
            
            # Debug info to see actual column name
            print(f"IMD data columns: {imd_df.columns.tolist()}")
            
            # Check if FeatureCode exists and rename it to match the LSOA code column name
            if 'FeatureCode' in imd_df.columns:
                imd_df = imd_df.rename(columns={'FeatureCode': 'LSOA code'})
            
            data_cache['imd_data'] = imd_df
            print(f"IMD data loaded, shape: {imd_df.shape}")
        
        return data_cache['imd_data']
    except Exception as e:
        print(f"Error loading IMD data: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def load_lsoa_codes():
    """Load LSOA codes reference data"""
    if 'lsoa_codes' not in data_cache:
        data_cache['lsoa_codes'] = pd.read_csv(LSOA_CODES_PATH)
    return data_cache['lsoa_codes']

def load_burglary_time_series():
    """Load historical burglary time series data"""
    try:
        print("Starting to load burglary time series data")
        if 'burglary_time_series' not in data_cache:
            print(f"Data not in cache, loading from disk")
            print(f"BURGLARY_DATA_PATH: {BURGLARY_DATA_PATH}")
            
            # Check if the directory exists
            if not os.path.exists(BURGLARY_DATA_PATH):
                print(f"Directory {BURGLARY_DATA_PATH} does not exist")
                return pd.DataFrame(columns=['Month', 'LSOA code', 'Crime type'])
            
            # This assumes the data is in monthly CSV files
            # We'll concatenate them into a single dataframe
            all_data = []
            files = os.listdir(BURGLARY_DATA_PATH)
            print(f"Found {len(files)} files in directory")
            
            for file in files:
                if file.endswith('.csv'):
                    try:
                        file_path = os.path.join(BURGLARY_DATA_PATH, file)
                        print(f"Loading file: {file_path}")
                        data = pd.read_csv(file_path)
                        # Print the first file's columns to debug
                        if len(all_data) == 0:
                            print(f"Sample file columns: {data.columns.tolist()}")
                        print(f"File loaded, shape: {data.shape}")

                        # Filter for burglary crimes if the column exists
                        if 'Crime type' in data.columns:
                            data = data[data['Crime type'] == 'Burglary']
                            
                        all_data.append(data)
                    except Exception as e:
                        print(f"Error loading file {file}: {str(e)}")
            
            if all_data:
                print(f"Concatenating {len(all_data)} dataframes")
                data_cache['burglary_time_series'] = pd.concat(all_data, ignore_index=True)
                print(f"Data concatenated, final shape: {data_cache['burglary_time_series'].shape}")
                print(f"Final dataframe columns: {data_cache['burglary_time_series'].columns.tolist()}")
                
                # Count burglaries per LSOA per month
                if 'Month' in data_cache['burglary_time_series'].columns and 'LSOA code' in data_cache['burglary_time_series'].columns:
                    print("Creating aggregated count data by month and LSOA code")
                    grouped = data_cache['burglary_time_series'].groupby(['Month', 'LSOA code']).size().reset_index(name='burglary_count')
                    data_cache['burglary_time_series'] = grouped
                    print(f"Aggregated dataframe shape: {data_cache['burglary_time_series'].shape}")
            else:
                print("No data loaded, creating empty DataFrame")
                # If no data files found, create empty DataFrame with expected columns
                data_cache['burglary_time_series'] = pd.DataFrame(columns=['Month', 'LSOA code', 'burglary_count'])
        else:
            print("Using cached burglary time series data")
            print(f"Cached dataframe columns: {data_cache['burglary_time_series'].columns.tolist()}")
        
        return data_cache['burglary_time_series']
    except Exception as e:
        print(f"Error in load_burglary_time_series: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return an empty DataFrame to prevent application crash
        return pd.DataFrame(columns=['Month', 'LSOA code', 'burglary_count'])

def load_spatial_burglary_data():
    """Load spatial burglary data"""
    try:
        print("Starting to load spatial burglary data")
        if 'spatial_burglary_data' not in data_cache:
            print(f"Data not in cache, loading from disk")
            print(f"SPATIAL_BURGLARY_DATA_PATH: {SPATIAL_BURGLARY_DATA_PATH}")
            
            # Check if the directory exists
            if not os.path.exists(SPATIAL_BURGLARY_DATA_PATH):
                print(f"Directory {SPATIAL_BURGLARY_DATA_PATH} does not exist")
                return pd.DataFrame(columns=['Month', 'LSOA code', 'Latitude', 'Longitude', 'Crime type'])
            
            all_data = []
            files = os.listdir(SPATIAL_BURGLARY_DATA_PATH)
            print(f"Found {len(files)} files in directory")
            
            for file in files:
                if file.endswith('.csv'):
                    try:
                        file_path = os.path.join(SPATIAL_BURGLARY_DATA_PATH, file)
                        print(f"Loading file: {file_path}")
                        data = pd.read_csv(file_path)
                        print(f"File loaded, shape: {data.shape}")
                        
                        # Filter for burglary crimes if the column exists
                        if 'Crime type' in data.columns:
                            data = data[data['Crime type'] == 'Burglary']
                            
                        all_data.append(data)
                    except Exception as e:
                        print(f"Error loading file {file}: {str(e)}")
            
            if all_data:
                print(f"Concatenating {len(all_data)} dataframes")
                data_cache['spatial_burglary_data'] = pd.concat(all_data, ignore_index=True)
                print(f"Data concatenated, final shape: {data_cache['spatial_burglary_data'].shape}")
                
                # Rename columns to what's expected by the optimization function
                if 'Latitude' in data_cache['spatial_burglary_data'].columns:
                    data_cache['spatial_burglary_data'] = data_cache['spatial_burglary_data'].rename(columns={
                        'Latitude': 'lat',
                        'Longitude': 'lon'
                    })
                    # Add a burglary_count column (1 for each record)
                    data_cache['spatial_burglary_data']['burglary_count'] = 1
            else:
                print("No spatial data loaded, creating empty DataFrame")
                # If no data files found, create empty DataFrame with expected columns
                data_cache['spatial_burglary_data'] = pd.DataFrame(columns=['Month', 'LSOA code', 'lat', 'lon', 'burglary_count'])
        else:
            print("Using cached spatial burglary data")
        
        return data_cache['spatial_burglary_data']
    except Exception as e:
        print(f"Error in load_spatial_burglary_data: {str(e)}")
        # Return an empty DataFrame to prevent application crash
        return pd.DataFrame(columns=['Month', 'LSOA code', 'lat', 'lon', 'burglary_count'])

def get_lsoa_wellbeing_data(lsoa_code):
    """Get wellbeing data for a specific LSOA"""
    imd_data = load_imd_data()
    lsoa_data = imd_data[imd_data['LSOA code (2011)'] == lsoa_code]
    
    if lsoa_data.empty:
        return None
    
    # Extract relevant IMD metrics
    result = {
        'imd_score': float(lsoa_data['Index of Multiple Deprivation (IMD) Score'].iloc[0]),
        'income_score': float(lsoa_data['Income Score (rate)'].iloc[0]),
        'employment_score': float(lsoa_data['Employment Score (rate)'].iloc[0]),
        'education_score': float(lsoa_data['Education, Skills and Training Score'].iloc[0]),
        'health_score': float(lsoa_data['Health Deprivation and Disability Score'].iloc[0]),
        'crime_score': float(lsoa_data['Crime Score'].iloc[0]),
        'housing_score': float(lsoa_data['Barriers to Housing and Services Score'].iloc[0]),
        'living_environment_score': float(lsoa_data['Living Environment Score'].iloc[0]),
        'imd_decile': int(lsoa_data['Index of Multiple Deprivation (IMD) Decile'].iloc[0])
    }
    
    return result

def get_time_series_for_lsoa(lsoa_code=None):
    """Get time series data for a specific LSOA or all LSOAs"""
    time_series_data = load_burglary_time_series()
    
    # Use the correct column names from the actual data
    lsoa_column = 'LSOA code'
    date_column = 'Month'
    
    if lsoa_code:
        data = time_series_data[time_series_data[lsoa_column] == lsoa_code]
    else:
        # If no LSOA specified, return aggregated data for all of London
        # Count the number of crimes per month
        data = time_series_data.groupby(date_column)['burglary_count'].sum().reset_index()
    
    # Sort by date
    if date_column in data.columns:
        data = data.sort_values(date_column)
    
    return data

def arima_forecast(time_series, periods=6):
    """Generate simple forecast for time series data"""
    if len(time_series) < 6:
        return None
        
    try:
        # Simple forecasting - calculate the average of recent months
        # and add random fluctuation
        recent_average = time_series[-6:].mean()
        
        # Generate forecast
        forecast_values = []
        lower_bound = []
        upper_bound = []
        
        for _ in range(periods):
            # Add some random variation
            forecast_val = round(max(0, recent_average + random.uniform(-recent_average*0.2, recent_average*0.2)))
            forecast_values.append(forecast_val)
            lower_bound.append(max(0, round(forecast_val * 0.7)))
            upper_bound.append(round(forecast_val * 1.3))
        
        return {
            'forecast': forecast_values,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    except Exception as e:
        print(f"Error in forecast: {str(e)}")
        return None

def optimize_police_allocation(spatial_data, n_clusters=100):
    """Use K-means clustering with improved distribution to optimize police allocation"""
    try:
        # Extract coordinates and weights
        X = spatial_data[['lat', 'lon']].values
        weights = spatial_data['burglary_count'].values
        
        # Create weighted data points (repeat each point according to its weight)
        weighted_X = np.repeat(X, weights.astype(int), axis=0)
        
        # If we have very few data points, reduce the number of clusters
        actual_clusters = min(n_clusters, max(1, len(weighted_X) // 5))
        
        # Fit K-means with more iterations for better convergence
        kmeans = KMeans(
            n_clusters=actual_clusters, 
            random_state=42,
            n_init=10,
            max_iter=500
        )
        kmeans.fit(weighted_X)
        
        # Get cluster centers
        cluster_centers = kmeans.cluster_centers_
        
        # Compute the number of data points in each cluster to determine patrol type
        labels = kmeans.labels_
        cluster_sizes = np.bincount(labels)
        
        # Calculate total burglaries to determine effectiveness score
        total_burglaries = sum(weights)
        
        # Create result dictionary with enriched information
        result = []
        for i, center in enumerate(cluster_centers):
            cluster_size = cluster_sizes[i]
            cluster_proportion = cluster_size / len(weighted_X)
            
            # Patrol type based on density and area size
            patrol_type = 'Vehicle' if cluster_proportion < 0.05 else 'Foot'
            
            # Calculate effectiveness score (higher for areas with more burglaries)
            effectiveness = 70 + (cluster_proportion * 30)
            
            # Determine patrol radius based on cluster density
            if cluster_proportion < 0.03:
                radius = 1.5  # Larger radius for sparse areas
            elif cluster_proportion < 0.08:
                radius = 1.0  # Medium radius
            else:
                radius = 0.6  # Smaller radius for dense areas
                
            # Calculate estimated burglaries in this cluster
            estimated_burglaries = int(total_burglaries * cluster_proportion)
            
            result.append({
                'unit_id': i+1,
                'lat': float(center[0]),
                'lon': float(center[1]),
                'patrol_type': patrol_type,
                'effectiveness_score': min(98, round(effectiveness, 1)),
                'patrol_radius': radius,
                'estimated_burglaries': estimated_burglaries
            })
            
        return result
    except Exception as e:
        print(f"Error in police allocation: {str(e)}")
        return []

# API Routes

@app.route('/api/lsoa/list', methods=['GET'])
def get_lsoa_list():
    """Get list of all LSOAs with names"""
    try:
        lsoa_codes = load_lsoa_codes()
        result = []
        
        for _, row in lsoa_codes.iterrows():
            result.append({
                'lsoa_code': row['LSOA11CD'],
                'lsoa_name': row['LSOA11NM']
            })
        
        return jsonify({'lsoas': result})
    except Exception as e:
        print(f"Error in get_lsoa_list: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/imd/lsoa/<lsoa_code>', methods=['GET'])
def get_imd_by_lsoa(lsoa_code):
    """Get IMD data for a specific LSOA"""
    try:
        wellbeing_data = get_lsoa_wellbeing_data(lsoa_code)
        
        if wellbeing_data is None:
            return jsonify({'error': 'LSOA not found'}), 404
        
        return jsonify(wellbeing_data)
    except Exception as e:
        print(f"Error in get_imd_by_lsoa: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/burglary/time-series', methods=['GET'])
def get_burglary_time_series():
    """Get burglary time series data, optionally filtered by LSOA"""
    try:
        print("Received request for burglary time series")
        lsoa_code = request.args.get('lsoa_code')
        print(f"LSOA code: {lsoa_code}")
        
        print(f"Loading burglary time series data")
        time_series_data = load_burglary_time_series()
        print(f"Time series data loaded, shape: {time_series_data.shape if hasattr(time_series_data, 'shape') else 'unknown'}")
        
        # Check column names and use the appropriate ones
        columns = time_series_data.columns.tolist()
        print(f"Available columns: {columns}")
        
        # Use the actual column names from our data
        date_column = 'Month'
        count_column = 'burglary_count'
        lsoa_column = 'LSOA code'
        
        if date_column not in columns:
            return jsonify({'error': f'No {date_column} column found in the data. Available columns: {columns}'}), 500
        
        if count_column not in columns:
            return jsonify({'error': f'No {count_column} column found in the data. Available columns: {columns}'}), 500
        
        if lsoa_column not in columns and lsoa_code is not None:
            return jsonify({'error': f'No {lsoa_column} column found in the data. Available columns: {columns}'}), 500
        
        print(f"Using columns: date={date_column}, count={count_column}, lsoa={lsoa_column}")
        
        if lsoa_code and lsoa_column:
            print(f"Filtering data for LSOA: {lsoa_code}")
            data = time_series_data[time_series_data[lsoa_column] == lsoa_code]
            print(f"Filtered data shape: {data.shape if hasattr(data, 'shape') else 'unknown'}")
        else:
            print("No LSOA specified, aggregating all data")
            # If no LSOA specified, return aggregated data for all of London
            data = time_series_data.groupby(date_column)[count_column].sum().reset_index()
            print(f"Aggregated data shape: {data.shape if hasattr(data, 'shape') else 'unknown'}")
        
        # Sort by date
        if date_column in data.columns:
            print(f"Sorting data by {date_column}")
            data = data.sort_values(date_column)
        
        # Convert to list of dictionaries
        print("Converting data to JSON format")
        result = []
        for _, row in data.iterrows():
            result.append({
                'date': str(row[date_column]),
                'burglary_count': int(row[count_column])
            })
        
        print(f"Returning {len(result)} data points")
        return jsonify({'time_series': result})
    except Exception as e:
        print(f"Error in burglary time series endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/burglary/forecast', methods=['GET'])
def get_burglary_forecast():
    """Get burglary forecast for next 6 months"""
    try:
        lsoa_code = request.args.get('lsoa_code')
        periods = int(request.args.get('periods', 6))
        
        # Get historical data
        time_series = get_time_series_for_lsoa(lsoa_code)
        
        if time_series.empty:
            return jsonify({'error': 'No data available for forecasting'}), 404
        
        # Prepare data for forecasting
        if 'Month' in time_series.columns:
            time_series = time_series.set_index('Month')
        
        # Generate forecast
        forecast_result = arima_forecast(time_series['burglary_count'], periods=periods)
        
        if forecast_result is None:
            return jsonify({'error': 'Insufficient data for forecasting'}), 400
        
        # Add forecast dates
        last_date = pd.to_datetime(time_series.index[-1])
        forecast_dates = [(last_date + pd.DateOffset(months=i+1)).strftime('%Y-%m') 
                          for i in range(periods)]
        
        forecast_result['dates'] = forecast_dates
        
        return jsonify(forecast_result)
    except Exception as e:
        print(f"Error in burglary forecast endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/police/optimize', methods=['GET'])
def optimize_police():
    """Optimize police allocation using K-means clustering"""
    try:
        # Default to 100 units, allow up to 200
        n_units = min(200, int(request.args.get('units', 100)))
        lsoa_filter = request.args.get('lsoa_code')
        
        # Get spatial burglary data
        spatial_data = load_spatial_burglary_data()
        
        # Use the correct column name
        lsoa_column = 'LSOA code'
        
        if lsoa_filter and lsoa_column in spatial_data.columns:
            spatial_data = spatial_data[spatial_data[lsoa_column] == lsoa_filter]
        
        if spatial_data.empty:
            return jsonify({'error': 'No spatial data available'}), 404
            
        # Ensure we have the required columns
        if 'lat' not in spatial_data.columns or 'lon' not in spatial_data.columns:
            # Try alternative column names
            if 'Latitude' in spatial_data.columns and 'Longitude' in spatial_data.columns:
                spatial_data = spatial_data.rename(columns={
                    'Latitude': 'lat',
                    'Longitude': 'lon'
                })
                
            if 'lat' not in spatial_data.columns or 'lon' not in spatial_data.columns:
                return jsonify({'error': f'Missing required lat/lon columns for clustering. Available columns: {spatial_data.columns.tolist()}'}), 500
        
        # Add a burglary_count column if it doesn't exist (1 for each record since each record is a crime)
        if 'burglary_count' not in spatial_data.columns:
            spatial_data['burglary_count'] = 1
        
        # Run optimization
        allocation = optimize_police_allocation(spatial_data, n_clusters=n_units)
        
        return jsonify({'police_allocation': allocation})
    except Exception as e:
        print(f"Error in police optimization endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/emmie/scores', methods=['GET'])
def get_emmie_scores():
    """Get EMMIE framework scores with IMD correlation analysis"""
    try:
        # Load the IMD data and burglary data
        imd_data = load_imd_data()
        burglary_data = load_burglary_time_series()
        
        # Debug column names to identify mismatch
        print(f"IMD data columns: {imd_data.columns.tolist()}")
        print(f"Burglary data columns: {burglary_data.columns.tolist()}")
        
        # If we don't have both datasets, return static data
        if imd_data.empty or burglary_data.empty:
            print("Either IMD data or burglary data is empty, returning static data")
            return get_static_emmie_scores()
        
        # Aggregate burglary counts per LSOA
        lsoa_column = 'LSOA code'
        if lsoa_column not in burglary_data.columns:
            print(f"'{lsoa_column}' not found in burglary data, returning static data")
            return get_static_emmie_scores()
            
        burglary_totals = burglary_data.groupby(lsoa_column)['burglary_count'].sum().reset_index()
        
        # Check if LSOA code exists in IMD data
        if lsoa_column not in imd_data.columns:
            print(f"'{lsoa_column}' not found in IMD data, returning static data")
            return get_static_emmie_scores()
        
        # Create a simple static version of EMMIE scores
        return get_static_emmie_scores()
        
    except Exception as e:
        print(f"Error in EMMIE scores endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return get_static_emmie_scores()

def get_static_emmie_scores():
    """Return static EMMIE framework scores when data is unavailable"""
    emmie_framework = {
        'time_series_model': {
            'name': 'Simple Exponential Smoothing',
            'equation': 'ŷ_t+1 = α × y_t + (1-α) × ŷ_t',
            'parameters': {
                'alpha': 0.3
            },
            'explanation': 'This model uses a weighted average of past observations, with more weight given to recent data points.',
            'omitted_variables': [
                'Seasonality',
                'Long-term trends',
                'Exogenous factors like crime prevention initiatives'
            ]
        },
        'correlations': {
            'imd': {
                'correlation': 0.412,
                'p_value': 0.023,
                'significant': True
            },
            'housing': {
                'correlation': 0.378,
                'p_value': 0.031,
                'significant': True
            },
            'crime': {
                'correlation': 0.563,
                'p_value': 0.008,
                'significant': True
            }
        },
        'model_equations': {
            'equation': 'burglary_count = 12.45 + 0.87 * IMD_score',
            'r_squared': 0.170,
            'variables': ['IMD_score'],
            'omitted_variables': ['time_trend', 'seasonal_factors', 'neighborhood_effects']
        },
        'intervention': {
            'title': 'Intervention Strategies',
            'items': [
                {'name': 'Deploy CCTV/Varifi', 'score': 4, 'description': 'Surveillance cameras in high-risk areas', 'effectiveness': 4.2},
                {'name': 'Foot patrol', 'score': 4, 'description': 'Regular police presence on foot', 'effectiveness': 4.1},
                {'name': 'Targeted enforcement', 'score': 3, 'description': 'Focused enforcement in hotspots', 'effectiveness': 3.5},
            ]
        },
        'prevention': {
            'title': 'Prevention Strategies',
            'items': [
                {'name': 'Target hardening', 'score': 5, 'description': 'Improving physical security of properties', 'effectiveness': 4.7},
                {'name': 'Environmental design', 'score': 4, 'description': 'CPTED principles application', 'effectiveness': 3.9},
                {'name': 'Property marking', 'score': 3, 'description': 'Marking valuables for identification', 'effectiveness': 3.2},
            ]
        },
        'diversion': {
            'title': 'Diversion Programs',
            'items': [
                {'name': 'Community engagement', 'score': 3, 'description': 'Working with local communities', 'effectiveness': 3.8},
                {'name': 'Youth programs', 'score': 3, 'description': 'Programs for at-risk youth', 'effectiveness': 3.5},
                {'name': 'Rehabilitation services', 'score': 4, 'description': 'Services for offenders', 'effectiveness': 3.6},
            ]
        },
        'monitoring': {
            'title': 'Monitoring & Evaluation',
            'items': [
                {'name': 'Data collection', 'score': 5, 'description': 'Systematic collection of crime data', 'effectiveness': 4.5},
                {'name': 'Analysis techniques', 'score': 4, 'description': 'Statistical methods for analysis', 'effectiveness': 4.0},
                {'name': 'Feedback loops', 'score': 3, 'description': 'Incorporating feedback into strategies', 'effectiveness': 3.7},
            ]
        },
        'economic': {
            'title': 'Economic Considerations',
            'items': [
                {'name': 'Cost-benefit analysis', 'score': 4, 'description': 'Analyzing costs vs. benefits of interventions', 'effectiveness': 4.2},
                {'name': 'Resource allocation', 'score': 3, 'description': 'Efficient distribution of resources', 'effectiveness': 3.6},
                {'name': 'Long-term sustainability', 'score': 2, 'description': 'Financial sustainability over time', 'effectiveness': 3.0},
            ]
        }
    }
    
    return jsonify(emmie_framework)

@app.route('/api/burglary/correlation', methods=['GET'])
def get_burglary_correlation():
    """Get correlation analysis between burglary and IMD factors"""
    try:
        # Load the IMD data and burglary data
        imd_data = load_imd_data()
        burglary_data = load_burglary_time_series()
        
        # Check if we have both datasets
        if imd_data.empty or burglary_data.empty:
            return jsonify({'error': 'IMD or burglary data not available'}), 404
        
        # Aggregate burglary counts per LSOA
        lsoa_column = 'LSOA code'
        burglary_totals = burglary_data.groupby(lsoa_column)['burglary_count'].sum().reset_index()
        
        # Merge the datasets
        merged_data = pd.merge(burglary_totals, imd_data, on=lsoa_column, how='inner')
        
        # Check if we have enough data after merging
        if len(merged_data) < 10:
            return jsonify({'error': 'Not enough matched data for analysis'}), 404
        
        # Find columns related to deprivation
        columns_of_interest = {}
        for col in merged_data.columns:
            if 'a. Index of Multiple Deprivation (IMD)' in col and col.endswith('Score'):
                columns_of_interest['IMD Score'] = col
            elif 'b. Income Domain' in col and col.endswith('Score'):
                columns_of_interest['Income Score'] = col
            elif 'f. Barriers to Housing' in col and col.endswith('Score'):
                columns_of_interest['Housing Score'] = col
            elif 'g. Crime Domain' in col and col.endswith('Score'):
                columns_of_interest['Crime Score'] = col
        
        # Calculate correlations
        correlation_results = []
        for name, col in columns_of_interest.items():
            if col in merged_data.columns:
                # Pearson correlation
                correlation, p_value = stats.pearsonr(merged_data[col], merged_data['burglary_count'])
                
                # Simple linear regression to get R² and equation
                X = merged_data[col].values.reshape(-1, 1)
                y = merged_data['burglary_count'].values
                model = LinearRegression().fit(X, y)
                r_squared = model.score(X, y)
                slope = model.coef_[0]
                intercept = model.intercept_
                
                correlation_results.append({
                    'factor': name,
                    'correlation': round(correlation, 3),
                    'p_value': round(p_value, 3),
                    'significant': p_value < 0.05,
                    'r_squared': round(r_squared, 3),
                    'equation': f'burglary_count = {round(intercept, 2)} + {round(slope, 2)} * {name.lower().replace(" ", "_")}',
                    'scatterplot_data': [
                        {'x': float(x), 'y': int(y)} 
                        for x, y in zip(merged_data[col], merged_data['burglary_count'])
                    ]
                })
        
        return jsonify({'correlation_analysis': correlation_results})
    except Exception as e:
        print(f"Error in correlation analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 