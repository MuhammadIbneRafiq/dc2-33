import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
from pathlib import Path
import datetime

# Set the plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Create output directory if it doesn't exist
if not os.path.exists('burglary_type_analysis'):
    os.makedirs('burglary_type_analysis')

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

# Load all three datasets
recent_data = load_data("MPS LSOA Level Crime (most recent 24 months).csv")
historical_data = load_data("MPS LSOA Level Crime (Historical).csv")
historical_earlier_data = load_data("MPS LSOA Level Crime (Historical earlier).csv")

# Get the time columns from each dataset
recent_time_cols = [col for col in recent_data.columns if col.isdigit()]
historical_time_cols = [col for col in historical_data.columns if col.isdigit()]
historical_earlier_time_cols = [col for col in historical_earlier_data.columns if col.isdigit()]

# Check burglary subcategories
burglary_categories_recent = recent_data[recent_data['Major Category'] == 'BURGLARY']['Minor Category'].unique()
burglary_categories_historical = historical_data[historical_data['Major Category'] == 'BURGLARY']['Minor Category'].unique()
burglary_categories_earlier = historical_earlier_data[historical_earlier_data['Major Category'] == 'BURGLARY']['Minor Category'].unique()

all_burglary_categories = list(set(list(burglary_categories_recent) + list(burglary_categories_historical) + list(burglary_categories_earlier)))
print("\nAll burglary subcategories across datasets:")
for cat in sorted(all_burglary_categories):
    print(f"- {cat}")

# Combine the datasets
# First, we need to identify common columns
common_cols = ['LSOA Code', 'LSOA Name', 'Borough', 'Major Category', 'Minor Category']

# For recent data, keep only the common columns and the time columns
recent_subset = recent_data[common_cols + recent_time_cols]

# For historical data, keep only the common columns and the time columns
historical_subset = historical_data[common_cols + historical_time_cols]

# For historical_earlier data, keep only the common columns and the time columns
historical_earlier_subset = historical_earlier_data[common_cols + historical_earlier_time_cols]

# Find all unique time columns across the three datasets
all_time_cols = list(set(recent_time_cols + historical_time_cols + historical_earlier_time_cols))
all_time_cols.sort()  # Sort to ensure chronological order

print(f"\nTotal number of time periods available: {len(all_time_cols)}")
print(f"Time range: {min(all_time_cols)} to {max(all_time_cols)}")

# Filter for burglary-related crimes only
def filter_burglary_data(df):
    return df[df['Major Category'] == 'BURGLARY']

recent_burglary = filter_burglary_data(recent_subset)
historical_burglary = filter_burglary_data(historical_subset)
historical_earlier_burglary = filter_burglary_data(historical_earlier_subset)

# Function to map burglary categories based on the police classification change in April 2017
def map_burglary_category(row):
    # Extract year and month from the date column
    date_str = row['date']
    year = int(date_str[:4])
    month = int(date_str[4:])
    
    category = row['Minor Category']
    
    # Properly categorize burglaries:
    # - "Burglary in a dwelling" and "Residential burglary" should both be "Residential Burglary"
    # - Everything else (Business, Non-dwelling, Other) should keep their respective categories
    
    if 'DWELLING' in category or 'RESIDENTIAL' in category:
        return 'Residential Burglary'
    elif 'BUSINESS' in category:
        return 'Business and Community Burglary'
    elif 'NON-DWELLING' in category:
        return 'Non-Dwelling Burglary'
    else:
        return 'Other Burglary'

# Function to aggregate by LSOA and burglary type
def aggregate_by_lsoa_and_type(df, time_cols):
    result = []
    
    # For each time period (month)
    for col in time_cols:
        if col in df.columns:
            # Get data for this month
            monthly_data = df[['LSOA Code', 'LSOA Name', 'Borough', 'Minor Category', col]].copy()
            monthly_data = monthly_data[monthly_data[col] > 0]  # Only keep rows with burglaries
            
            if not monthly_data.empty:
                # Add date column
                monthly_data['date'] = col
                
                # Rename count column
                monthly_data.rename(columns={col: 'count'}, inplace=True)
                
                # Add to result
                result.append(monthly_data)
    
    # Combine all months
    if result:
        combined = pd.concat(result)
        
        # Add categorized burglary type
        combined['burglary_category'] = combined.apply(map_burglary_category, axis=1)
        
        return combined
    else:
        return pd.DataFrame()

# Process each dataset
recent_processed = aggregate_by_lsoa_and_type(recent_burglary, recent_time_cols)
historical_processed = aggregate_by_lsoa_and_type(historical_burglary, historical_time_cols)
historical_earlier_processed = aggregate_by_lsoa_and_type(historical_earlier_burglary, historical_earlier_time_cols)

# Combine all processed data
all_processed = pd.concat([recent_processed, historical_processed, historical_earlier_processed])

# Convert date to datetime for better sorting and grouping
all_processed['date_dt'] = pd.to_datetime(all_processed['date'], format='%Y%m')
all_processed['year'] = all_processed['date_dt'].dt.year
all_processed['month'] = all_processed['date_dt'].dt.month

print(f"\nTotal records after processing: {len(all_processed)}")

# ----------------------
# Task 1: Annual trends by burglary type
# ----------------------

# Group by year and burglary category to get annual totals
annual_totals = all_processed.groupby(['year', 'burglary_category'])['count'].sum().reset_index()

# Check data
print("\nAnnual totals by category (sample):")
print(annual_totals.head(10))

# Pivot for easier plotting
annual_pivot = annual_totals.pivot(index='year', columns='burglary_category', values='count').fillna(0)

# Create the annual trends plot
plt.figure(figsize=(16, 10))

# Color mapping for categories
colors = {
    'Residential Burglary': '#e63946',  # Red
    'Business and Community Burglary': '#1d3557',  # Blue
    'Other Burglary': '#2a9d8f'  # Teal
}

# Plot each burglary category
for category in annual_pivot.columns:
    color = colors.get(category, 'gray')
    plt.plot(annual_pivot.index, annual_pivot[category], marker='o', linewidth=3, 
             markersize=8, label=category, color=color)

plt.title('Annual Burglary Trends by Type', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Number of Burglaries', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(title='Burglary Type', fontsize=14)
plt.xticks(annual_pivot.index, rotation=45)

# Add data labels for key points
for category in annual_pivot.columns:
    for year in [2010, 2017, 2023]:  # Add labels for start, classification change year, and end
        if year in annual_pivot.index:
            value = annual_pivot.loc[year, category]
            plt.text(year, value + (value * 0.02), f'{int(value):,}', 
                    ha='center', va='bottom', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

plt.tight_layout()

# Save the plot with high resolution
plt.savefig('burglary_type_analysis/updated_annual_burglary_trends.png', dpi=300, bbox_inches='tight')
plt.close()

# ----------------------
# Task 2: Top 10 LSOAs by burglary type
# ----------------------

# Group by LSOA and burglary category to get totals
lsoa_totals = all_processed.groupby(['LSOA Code', 'LSOA Name', 'Borough', 'burglary_category'])['count'].sum().reset_index()

# Get overall top 10 LSOAs
top_lsoas_overall = lsoa_totals.groupby(['LSOA Code', 'LSOA Name', 'Borough'])['count'].sum().reset_index()
top_lsoas_overall = top_lsoas_overall.sort_values('count', ascending=False).head(10)

print("\nTop 10 LSOAs by total burglaries:")
print(top_lsoas_overall[['LSOA Name', 'Borough', 'count']])

# Filter for just the top 10 LSOAs
top_lsoa_codes = top_lsoas_overall['LSOA Code'].tolist()
top_lsoa_data = lsoa_totals[lsoa_totals['LSOA Code'].isin(top_lsoa_codes)].copy()

# Create labels for LSOAs (combining LSOA name and borough)
top_lsoa_data['LSOA_Label'] = top_lsoa_data['LSOA Name'] + ' (' + top_lsoa_data['Borough'] + ')'

# Pivot for plotting
top_lsoa_pivot = top_lsoa_data.pivot_table(
    index='LSOA_Label',
    columns='burglary_category',
    values='count',
    aggfunc='sum'
).fillna(0)

# Reindex to ensure consistent order based on total burglaries
lsoa_order = top_lsoas_overall['LSOA Name'] + ' (' + top_lsoas_overall['Borough'] + ')'
top_lsoa_pivot = top_lsoa_pivot.reindex(lsoa_order)

# Sort LSOAs by residential burglary count for clearer visualization
residential_sorted = top_lsoa_pivot.sort_values('Residential Burglary', ascending=False)

# Create the plot
plt.figure(figsize=(16, 12))

# Plot as grouped bar chart with custom colors
ax = residential_sorted.plot(
    kind='bar', 
    figsize=(16, 12),
    color=[colors.get(cat, 'gray') for cat in residential_sorted.columns]
)

# Add value labels on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt='%d', fontsize=10)

plt.title('Top 10 LSOAs by Burglary Type', fontsize=20)
plt.xlabel('LSOA', fontsize=16)
plt.ylabel('Number of Burglaries', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Burglary Type', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('burglary_type_analysis/updated_top10_lsoas_by_type.png', dpi=300, bbox_inches='tight')
plt.close()

# Also create stacked bar chart for comparison
plt.figure(figsize=(16, 12))

# Plot as stacked bar chart with custom colors
ax = residential_sorted.plot(
    kind='bar', 
    stacked=True, 
    figsize=(16, 12),
    color=[colors.get(cat, 'gray') for cat in residential_sorted.columns]
)

# Add total value labels on top of stacked bars
totals = residential_sorted.sum(axis=1)
for i, total in enumerate(totals):
    plt.text(i, total + (total * 0.01), f'Total: {int(total):,}', 
             ha='center', va='bottom', fontsize=12, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

plt.title('Top 10 LSOAs by Burglary Type (Stacked)', fontsize=20)
plt.xlabel('LSOA', fontsize=16)
plt.ylabel('Number of Burglaries', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Burglary Type', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('burglary_type_analysis/updated_top10_lsoas_by_type_stacked.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAnalysis complete! Updated visualizations have been saved to the 'burglary_type_analysis' directory.") 