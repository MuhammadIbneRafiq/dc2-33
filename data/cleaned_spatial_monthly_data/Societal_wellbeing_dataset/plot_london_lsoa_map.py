#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import glob

# Set paths
data_path = 'final_merged_cleaned_lsoa_london_social_dataset.csv'
shapefile_dir = 'LB_LSOA2021_shp/LB_shp'

print("Loading LSOA data...")
# Load the merged dataset with LSOA codes
lsoa_data = pd.read_csv(data_path, encoding='latin1', low_memory=False)

# Check if the dataset contains the LSOA codes
if 'Lower Super Output Area' not in lsoa_data.columns:
    raise ValueError("LSOA code column not found in the dataset!")

print(f"Loaded {len(lsoa_data)} LSOAs.")

# Select a column to visualize as a heatmap (e.g., crime rate)
# Let's check if we have burglary data and use it for the visualization
if 'Crime (numbers);Burglary;2012/13' in lsoa_data.columns:
    value_column = 'Crime (numbers);Burglary;2012/13'
elif 'Crime (rates);Burglary;2012/13' in lsoa_data.columns:
    value_column = 'Crime (rates);Burglary;2012/13'
else:
    # Use a different column if burglary data is not available
    # We'll look for any numeric column that might be useful for visualization
    numeric_cols = lsoa_data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if lsoa_data[col].notna().sum() > len(lsoa_data) * 0.8:  # At least 80% non-null values
            value_column = col
            break
    else:
        # If no suitable column is found, use the Index of Multiple Deprivation if available
        if 'The Indices of Deprivation 2010;Domains and sub-domains;IMD Score' in lsoa_data.columns:
            value_column = 'The Indices of Deprivation 2010;Domains and sub-domains;IMD Score'
        else:
            # Otherwise use a simple count
            lsoa_data['count'] = 1
            value_column = 'count'

print(f"Using '{value_column}' for visualization.")

# Try to load and merge all borough shapefiles
print("Loading London borough shapefiles...")
all_boroughs_gdf = None

# Get all shapefile paths
shp_files = glob.glob(os.path.join(shapefile_dir, "*.shp"))

for shp_file in shp_files:
    try:
        borough_name = os.path.basename(shp_file).replace(".shp", "")
        print(f"Processing {borough_name}...")
        
        # Load shapefile
        gdf = gpd.read_file(shp_file)
        
        # Check if the GeoDataFrame contains LSOA information with modern column naming
        has_lsoa = 'lsoa21cd' in gdf.columns
        
        if has_lsoa:
            print(f"Found LSOA information in {borough_name} shapefile")
            if all_boroughs_gdf is None:
                all_boroughs_gdf = gdf
            else:
                all_boroughs_gdf = pd.concat([all_boroughs_gdf, gdf])
        else:
            print(f"Warning: {borough_name} shapefile does not contain LSOA information.")
    except Exception as e:
        print(f"Error processing {shp_file}: {e}")

if all_boroughs_gdf is None or len(all_boroughs_gdf) == 0:
    print("No valid borough shapefiles found with LSOA information.")
    print("Creating a basic map without LSOA boundaries...")
    
    # Create a basic map of London
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)
    
    # Add a title
    title_html = '''
    <h3 align="center" style="font-size:16px"><b>London Societal Wellbeing Map</b></h3>
    <h4 align="center" style="font-size:12px"><i>No LSOA boundaries available</i></h4>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save the map
    map_file = 'london_societal_wellbeing_map.html'
    m.save(map_file)
    print(f"Basic map saved as {map_file}")
else:
    print(f"Successfully loaded {len(all_boroughs_gdf)} LSOA boundaries.")
    
    # Now we know the LSOA column is 'lsoa21cd'
    lsoa_col = 'lsoa21cd'
    print(f"Using LSOA column: {lsoa_col}")
    
    # Convert LSOA codes if needed (check if they match format)
    if lsoa_data['Lower Super Output Area'].iloc[0].startswith('E') and not all_boroughs_gdf[lsoa_col].iloc[0].startswith('E'):
        # Format conversion may be needed here if formats don't match
        print("LSOA code format mismatch, attempting to harmonize...")
    
    # Merge the LSOA data with the shapefile data
    merged_gdf = all_boroughs_gdf.merge(
        lsoa_data, 
        left_on=lsoa_col, 
        right_on='Lower Super Output Area',
        how='left'
    )
    
    # Check if the merge worked
    match_count = merged_gdf[value_column].notna().sum()
    print(f"Matched {match_count} out of {len(all_boroughs_gdf)} LSOAs with data.")
    
    if match_count > 0:
        # Create a Folium map
        m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)
        
        # Create a colormap
        colormap = LinearSegmentedColormap.from_list(
            'custom_cmap', ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026']
        )
        
        # Add a choropleth layer
        folium.Choropleth(
            geo_data=merged_gdf,
            name='choropleth',
            data=merged_gdf,
            columns=[lsoa_col, value_column],
            key_on=f'feature.properties.{lsoa_col}',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=value_column
        ).add_to(m)
        
        # Add hover functionality
        style_function = lambda x: {'fillColor': '#ffffff', 
                                    'color': '#000000', 
                                    'fillOpacity': 0.1, 
                                    'weight': 0.1}
        highlight_function = lambda x: {'fillColor': '#000000', 
                                       'color': '#000000', 
                                       'fillOpacity': 0.5, 
                                       'weight': 0.3}
        
        # Ensure LSOA name column is available for tooltips
        lsoa_name_col = 'lsoa21nm' if 'lsoa21nm' in merged_gdf.columns else lsoa_col
        
        folium.features.GeoJson(
            merged_gdf,
            style_function=style_function,
            control=False,
            highlight_function=highlight_function,
            tooltip=folium.features.GeoJsonTooltip(
                fields=[lsoa_col, lsoa_name_col, value_column],
                aliases=['LSOA Code', 'LSOA Name', value_column],
                style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
            )
        ).add_to(m)
        
        # Add a title
        title_html = f'''
        <h3 align="center" style="font-size:16px"><b>London {value_column} by LSOA</b></h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Save the map
        map_file = 'london_lsoa_map.html'
        m.save(map_file)
        print(f"Interactive map saved as {map_file}")
        
        # Create a static plot for demonstration
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        merged_gdf.plot(column=value_column, cmap='YlOrRd', linewidth=0.1, ax=ax, edgecolor='0.8', legend=True)
        ax.set_title(f'London {value_column} by LSOA')
        ax.set_axis_off()
        plt.tight_layout()
        
        # Save the static map
        static_map_file = 'london_lsoa_map_static.png'
        plt.savefig(static_map_file, dpi=300, bbox_inches='tight')
        print(f"Static map saved as {static_map_file}")
    else:
        print("Failed to match LSOAs with data. Trying with a different approach...")
        
        # Create a map with borough boundaries instead since we can't match LSOAs
        print("Creating a map with borough boundaries...")
        
        # First, dissolve by borough
        borough_col = 'lad22nm'
        boroughs_gdf = all_boroughs_gdf.dissolve(by=borough_col).reset_index()
        
        # Aggregate data by borough
        borough_data = []
        for borough in boroughs_gdf[borough_col]:
            lsoas_in_borough = all_boroughs_gdf[all_boroughs_gdf[borough_col] == borough][lsoa_col].tolist()
            # Match these LSOAs in our data
            matched_data = lsoa_data[lsoa_data['Lower Super Output Area'].isin(lsoas_in_borough)]
            
            if not matched_data.empty and value_column in matched_data.columns:
                value = matched_data[value_column].mean()
            else:
                value = None
                
            borough_data.append({'Borough': borough, 'Value': value})
        
        # Create borough dataframe
        borough_df = pd.DataFrame(borough_data)
        
        # Merge with the boroughs GeoDataFrame
        merged_boroughs = boroughs_gdf.merge(borough_df, left_on=borough_col, right_on='Borough', how='left')
        
        # Now create a map using borough boundaries
        m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)
        
        # Check if we have any valid data to show
        if merged_boroughs['Value'].notna().sum() > 0:
            # Add a choropleth layer
            folium.Choropleth(
                geo_data=merged_boroughs,
                name='choropleth',
                data=merged_boroughs,
                columns=[borough_col, 'Value'],
                key_on=f'feature.properties.{borough_col}',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.5,
                legend_name=f'Average {value_column} by Borough'
            ).add_to(m)
            
            # Add hover functionality
            folium.features.GeoJson(
                merged_boroughs,
                style_function=lambda x: {'fillColor': '#ffffff', 'color': '#000000', 'fillOpacity': 0.1, 'weight': 0.5},
                control=False,
                highlight_function=lambda x: {'fillColor': '#000000', 'color': '#000000', 'fillOpacity': 0.5, 'weight': 1},
                tooltip=folium.features.GeoJsonTooltip(
                    fields=[borough_col, 'Value'],
                    aliases=['Borough', f'Average {value_column}'],
                    style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
                )
            ).add_to(m)
            
            title = f'London Borough Average {value_column}'
        else:
            # Just show borough boundaries without data
            for _, row in merged_boroughs.iterrows():
                folium.GeoJson(
                    row['geometry'],
                    style_function=lambda x: {'fillColor': '#cccccc', 'color': '#000000', 'fillOpacity': 0.1, 'weight': 0.7},
                ).add_to(m)
                folium.Marker(
                    location=[row['geometry'].centroid.y, row['geometry'].centroid.x],
                    popup=row[borough_col],
                    icon=folium.DivIcon(html=f"""<div style="font-size: 8pt">{row[borough_col]}</div>""")
                ).add_to(m)
            
            title = 'London Borough Boundaries'
        
        # Add a title
        title_html = f'''
        <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Save the map
        map_file = 'london_borough_map.html'
        m.save(map_file)
        print(f"Borough map saved as {map_file}")

print("donenenfydfyd") 