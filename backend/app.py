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
    if 'imd_data' not in data_cache:
        data_cache['imd_data'] = pd.read_csv(IMD_DATA_PATH)
    return data_cache['imd_data']

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

def optimize_police_allocation(spatial_data, n_clusters=8):
    """Use K-means clustering to optimize police allocation"""
    try:
        # Extract coordinates and weights
        X = spatial_data[['lat', 'lon']].values
        weights = spatial_data['burglary_count'].values
        
        # Create weighted data points (repeat each point according to its weight)
        weighted_X = np.repeat(X, weights.astype(int), axis=0)
        
        # Fit K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(weighted_X)
        
        # Get cluster centers
        cluster_centers = kmeans.cluster_centers_
        
        # Create result dictionary
        result = []
        for i, center in enumerate(cluster_centers):
            result.append({
                'unit_id': i+1,
                'lat': float(center[0]),
                'lon': float(center[1])
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
        n_units = int(request.args.get('units', 8))
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
    """Get EMMIE framework scores"""
    try:
        # The EMMIE framework scores would typically come from a database
        # Here we're using static data for the demo
        emmie_framework = {
            'intervention': {
                'items': [
                    {'name': 'Deploy CCTV/Varifi', 'score': 4, 'description': 'Surveillance cameras in high-risk areas'},
                    {'name': 'Foot patrol', 'score': 4, 'description': 'Regular police presence on foot'},
                    {'name': 'Targeted enforcement', 'score': 3, 'description': 'Focused enforcement in hotspots'},
                ]
            },
            'prevention': {
                'items': [
                    {'name': 'Target hardening', 'score': 5, 'description': 'Improving physical security of properties'},
                    {'name': 'Environmental design', 'score': 4, 'description': 'CPTED principles application'},
                    {'name': 'Property marking', 'score': 3, 'description': 'Marking valuables for identification'},
                ]
            },
            'diversion': {
                'items': [
                    {'name': 'Community engagement', 'score': 3, 'description': 'Working with local communities'},
                    {'name': 'Youth programs', 'score': 3, 'description': 'Programs for at-risk youth'},
                    {'name': 'Rehabilitation services', 'score': 4, 'description': 'Services for offenders'},
                ]
            }
        }
        
        return jsonify(emmie_framework)
    except Exception as e:
        print(f"Error in EMMIE scores endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/geojson/lsoa', methods=['GET'])
def get_lsoa_geojson():
    """Get GeoJSON for London LSOAs with burglary data"""
    try:
        # Load burglary data to get counts per LSOA
        burglary_data = load_burglary_time_series()
        
        # Use correct column name
        lsoa_column = 'LSOA code'
        
        if lsoa_column not in burglary_data.columns:
            return jsonify({'error': f'No {lsoa_column} column found in the data. Available columns: {burglary_data.columns.tolist()}'}), 500
        
        # Count burglaries per LSOA
        lsoa_counts = burglary_data.groupby(lsoa_column)['burglary_count'].sum().reset_index()
        
        # Create a simple feature collection
        features = []
        for _, row in lsoa_counts.iterrows():
            lsoa_code = row[lsoa_column]
            burglary_count = int(row['burglary_count'])
            
            # Determine risk level based on burglary count
            if burglary_count > 100:
                risk_level = 'Very High'
            elif burglary_count > 50:
                risk_level = 'High'
            elif burglary_count > 20:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            # In a real implementation, you would get the actual geometry from a GeoJSON file
            # Here we're creating a simple placeholder polygon
            features.append({
                'type': 'Feature',
                'properties': {
                    'lsoa_code': lsoa_code,
                    'lsoa_name': f'LSOA {lsoa_code}',  # This would be replaced with actual names
                    'burglary_count': burglary_count,
                    'risk_level': risk_level
                },
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]  # Placeholder
                }
            })
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        return jsonify(geojson)
    except Exception as e:
        print(f"Error in LSOA GeoJSON endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 