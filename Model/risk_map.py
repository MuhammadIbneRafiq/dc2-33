import os
import pandas as pd
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip
from folium.plugins import Search, FeatureGroupSubGroup, MarkerCluster
from branca.colormap import linear
import glob

# Set paths based on project structure
RESULTS_DIR = "Model/results"
SHAPEFILE_DIR = "Model/LB_shp"
OUTPUT_DIR = "Model/interactive_map"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_interactive_risk_map(predictions_file='risk_predictions.csv', output_file='interactive_burglary_risk_map.html'):
    """
    Creates an interactive web-based map of London showing burglary risk by LSOA.
    """
    print("Creating interactive burglary risk map for London...")
    
    # Step 1: Load risk predictions
    try:
        predictions_path = os.path.join(RESULTS_DIR, predictions_file)
        risk_data = pd.read_csv(predictions_path)
        print(f"Loaded risk predictions for {len(risk_data)} LSOAs")
        
        # Debug: Check risk data
        print(f"Risk data columns: {risk_data.columns.tolist()}")
        print(f"Risk score range: {risk_data['risk_score'].min()} to {risk_data['risk_score'].max()}")
    except FileNotFoundError:
        print(f"Error: Predictions file not found at {predictions_path}")
        print("Please run the GCN model first or check the file path.")
        return None
    
    # Step 2: Load LSOA boundary data
    print("Loading LSOA boundaries from shapefiles...")
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
                
                all_boundaries.append(gdf)
                print(f"Added {len(gdf)} LSOAs from {borough_name}")
            else:
                print(f"Warning: {borough_name} shapefile does not contain LSOA information.")
        except Exception as e:
            print(f"Error loading {shp_file}: {e}")
    
    # Combine all boundaries
    if not all_boundaries:
        print("Error: No valid boundary files found.")
        return None
    
    london_boundaries = pd.concat(all_boundaries, ignore_index=True)
    print(f"Successfully loaded {len(london_boundaries)} LSOA boundaries")
    
    # Debug: Check LSOA codes in boundaries
    print(f"Boundary data LSOA code column: {london_boundaries['LSOA_code'].name}")
    print(f"Sample LSOA codes in boundaries: {london_boundaries['LSOA_code'].head(3).tolist()}")
    print(f"Sample LSOA codes in predictions: {risk_data['LSOA_code'].head(3).tolist()}")
    
    # Step 3: Merge boundaries with risk predictions
    london_risk = london_boundaries.merge(risk_data, on='LSOA_code', how='inner')
    print(f"Matched risk data for {len(london_risk)} LSOAs out of {len(london_boundaries)}")
    
    # Debug: Check if merge worked
    if len(london_risk) == 0:
        print("ERROR: No matches between boundary and risk data!")
        print("Attempting alternative approach with case-insensitive matching...")
        
        # Try case-insensitive matching
        london_boundaries['LSOA_code_lower'] = london_boundaries['LSOA_code'].str.lower()
        risk_data['LSOA_code_lower'] = risk_data['LSOA_code'].str.lower()
        london_risk = london_boundaries.merge(risk_data, on='LSOA_code_lower', how='inner')
        print(f"After case-insensitive matching: {len(london_risk)} matches")
    
    # Step 4: Create risk categories
    london_risk['risk_score'] = london_risk['risk_score'].astype(float)
    london_risk['risk_category'] = pd.qcut(london_risk['risk_score'], 
                                           q=5, 
                                           labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Get LSOA name if available, otherwise use LSOA code
    if 'lsoa21nm' in london_risk.columns:
        london_risk['display_name'] = london_risk['lsoa21nm']
    else:
        london_risk['display_name'] = london_risk['LSOA_code']
    
    # Ensure geometry is valid
    london_risk = london_risk[london_risk.geometry.is_valid]
    
    # Step 5: Create the interactive map
    print("Creating interactive map...")
    
    # Initialize map centered on London
    london_map = folium.Map(location=[51.5074, -0.1278], 
                            zoom_start=10,
                            tiles='CartoDB positron')
    
    # Debug: Print risk score summary for coloring
    print(f"Risk score summary for map coloring:")
    print(f"  Min: {london_risk['risk_score'].min()}")
    print(f"  Max: {london_risk['risk_score'].max()}")
    print(f"  Mean: {london_risk['risk_score'].mean()}")
    
    # Create direct color map function instead of using branca
    def get_color(score):
        # Red-Yellow-Green color scheme (reversed)
        if score >= 80:
            return '#800026'  # Dark red for highest risk
        elif score >= 60:
            return '#BD0026'  # Red
        elif score >= 40:
            return '#FC4E2A'  # Orange
        elif score >= 20:
            return '#FD8D3C'  # Light orange
        else:
            return '#FFEDA0'  # Yellow for lowest risk
    
    # Create chloropleth map directly through GeoJson
    folium.GeoJson(
        london_risk,
        name="Risk Levels",
        style_function=lambda feature: {
            'fillColor': get_color(feature['properties']['risk_score']),
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['display_name', 'risk_score', 'risk_category'],
            aliases=['LSOA:', 'Risk Score:', 'Risk Level:'],
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
        )
    ).add_to(london_map)
    
    # Add legend manually
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; background-color: white;
                border-radius: 5px; padding: 10px; opacity: 0.8; font-family: Arial;">
        <p><strong>Burglary Risk Level</strong></p>
        <p><i style="background: #800026; width: 15px; height: 15px; display: inline-block;"></i> Very High (80-100)</p>
        <p><i style="background: #BD0026; width: 15px; height: 15px; display: inline-block;"></i> High (60-80)</p>
        <p><i style="background: #FC4E2A; width: 15px; height: 15px; display: inline-block;"></i> Medium (40-60)</p>
        <p><i style="background: #FD8D3C; width: 15px; height: 15px; display: inline-block;"></i> Low (20-40)</p>
        <p><i style="background: #FFEDA0; width: 15px; height: 15px; display: inline-block;"></i> Very Low (0-20)</p>
    </div>
    '''
    london_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 800px; height: 60px; 
                background-color: white; border-radius: 5px; z-index: 9999; padding: 10px; 
                font-family: Arial; font-size: 14px; text-align: center;">
        <h3>London Burglary Risk Prediction Map</h3>
        <p>Graph Convolutional Network model incorporating spatial relationships between areas</p>
    </div>
    '''
    london_map.get_root().html.add_child(folium.Element(title_html))
    
    # Save the map
    output_path = os.path.join(OUTPUT_DIR, output_file)
    london_map.save(output_path)
    print(f"Interactive map saved to {output_path}")
    
    return output_path
def create_high_risk_areas_map(predictions_file='risk_predictions.csv', 
                             n=50, 
                             output_file='high_risk_areas_map.html'):
    """
    Creates an interactive map highlighting the highest risk areas.
    
    Args:
        predictions_file: CSV file containing risk predictions
        n: Number of highest risk areas to highlight
        output_file: Output HTML file name
    """
    print(f"Creating map of top {n} highest risk areas...")
    
    # Load risk predictions
    try:
        predictions_path = os.path.join(RESULTS_DIR, predictions_file)
        risk_data = pd.read_csv(predictions_path)
    except FileNotFoundError:
        print(f"Error: Predictions file not found at {predictions_path}")
        return None
    
    # Identify top N highest risk areas
    top_risk = risk_data.sort_values('risk_score', ascending=False).head(n)
    
    # Load LSOA boundaries
    shp_files = glob.glob(os.path.join(SHAPEFILE_DIR, "*.shp"))
    all_boundaries = []
    
    for shp_file in shp_files:
        borough_name = os.path.basename(shp_file).replace(".shp", "")
        try:
            gdf = gpd.read_file(shp_file)
            if 'lsoa21cd' in gdf.columns:
                gdf['LSOA_code'] = gdf['lsoa21cd']
                if 'borough' not in gdf.columns:
                    gdf['borough'] = borough_name
                all_boundaries.append(gdf)
        except Exception:
            continue
    
    # Combine boundaries
    london_boundaries = pd.concat(all_boundaries, ignore_index=True)
    
    # Filter to only include high risk areas
    high_risk_areas = london_boundaries[london_boundaries['LSOA_code'].isin(top_risk['LSOA_code'])]
    
    # Merge with risk data
    high_risk_areas = high_risk_areas.merge(top_risk, on='LSOA_code', how='inner')
    
    # Get LSOA name if available
    if 'lsoa21nm' in high_risk_areas.columns:
        high_risk_areas['display_name'] = high_risk_areas['lsoa21nm']
    else:
        high_risk_areas['display_name'] = high_risk_areas['LSOA_code']
    
    # Create map centered on London
    high_risk_map = folium.Map(location=[51.5074, -0.1278], 
                              zoom_start=10,
                              tiles='CartoDB positron')
    
    # Add title
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 800px; height: 60px; 
                background-color: white; border-radius: 5px; z-index: 9999; padding: 10px; 
                font-family: Arial; font-size: 14px; text-align: center;">
        <h3>Top {n} Highest Risk Areas for Burglary in London</h3>
        <p>Based on Graph Convolutional Network predictions</p>
    </div>
    '''
    high_risk_map.get_root().html.add_child(folium.Element(title_html))
    
    # Create feature group
    high_risk_group = folium.FeatureGroup(name="High Risk Areas")
    
    # Add high risk areas with ranking
    for rank, (idx, row) in enumerate(high_risk_areas.iterrows(), 1):
        # Style based on risk ranking
        color = f'rgb({255}, {max(0, 255 - rank*5)}, 0)'
        
        # Add area polygon
        folium.GeoJson(
            row['geometry'],
            style_function=lambda x, color=color: {
                'fillColor': color,
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7
            },
            tooltip=f"Rank: {rank}<br>{row['display_name']}<br>Risk Score: {row['risk_score']:.1f}"
        ).add_to(high_risk_group)
        
        # Add rank marker
        folium.Marker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            tooltip=f"Rank: {rank}",
            icon=folium.DivIcon(html=f'<div style="background-color: white; width: 20px; height: 20px; border-radius: 10px; text-align: center; line-height: 20px; font-weight: bold;">{rank}</div>')
        ).add_to(high_risk_group)
    
    # Add feature group to map
    high_risk_map.add_child(high_risk_group)
    
    # Add layer control
    folium.LayerControl().add_to(high_risk_map)
    
    # Save map
    output_path = os.path.join(OUTPUT_DIR, output_file)
    high_risk_map.save(output_path)
    print(f"High risk areas map saved to {output_path}")
    
    return output_path

def main():
    """Main function to generate all interactive maps."""
    try:
        # Create standard risk map
        risk_map_path = create_interactive_risk_map()
        
        if risk_map_path:
            print("\nInteractive maps created successfully!")
            
    except Exception as e:
        print(f"Error generating maps: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()