import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict

# Set the plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Create output directory if it doesn't exist
if not os.path.exists('residential_burglary_trends'):
    os.makedirs('residential_burglary_trends')

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
burglary_categories_earlier = historical_earlier_data[historical_data['Major Category'] == 'BURGLARY']['Minor Category'].unique()

all_burglary_categories = list(set(list(burglary_categories_recent) + list(burglary_categories_historical) + list(burglary_categories_earlier)))
print("\nAll burglary subcategories across datasets:")
print(all_burglary_categories)

# Filter for TRUE residential burglary types
residential_types = [cat for cat in all_burglary_categories if ('RESIDENTIAL' in cat or 'DWELLING' in cat) and 'NON' not in cat]
non_residential_types = [cat for cat in all_burglary_categories if cat not in residential_types]

print("\nResidential burglary types:")
print(residential_types)
print("\nNon-residential burglary types:")
print(non_residential_types)

# Combine the datasets
# First, identify common columns
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

# Filter for residential burglary data only
def filter_residential_burglary(df):
    return df[(df['Major Category'] == 'BURGLARY') & 
              (df['Minor Category'].isin(residential_types))]

recent_residential = filter_residential_burglary(recent_subset)
historical_residential = filter_residential_burglary(historical_subset)
historical_earlier_residential = filter_residential_burglary(historical_earlier_subset)

# ----------------------
# Task 1: Grouped bar chart of residential burglary types by LSOA
# ----------------------

# Aggregate the data by LSOA and burglary type
def aggregate_by_lsoa_and_type(df, time_cols):
    # Create a copy of the dataframe
    df_copy = df.copy()
    
    # Calculate the sum of burglaries for each row
    df_copy['total'] = df_copy[time_cols].sum(axis=1)
    
    # Group by LSOA and burglary type, and calculate total burglaries
    grouped = df_copy.groupby(['LSOA Code', 'LSOA Name', 'Borough', 'Minor Category'])['total'].sum().reset_index()
    
    return grouped

# Aggregate data from each dataset
recent_by_lsoa_type = aggregate_by_lsoa_and_type(recent_residential, recent_time_cols)
historical_by_lsoa_type = aggregate_by_lsoa_and_type(historical_residential, historical_time_cols)
historical_earlier_by_lsoa_type = aggregate_by_lsoa_and_type(historical_earlier_residential, historical_earlier_time_cols)

# Combine the aggregated data
recent_by_lsoa_type['dataset'] = 'recent'
historical_by_lsoa_type['dataset'] = 'historical'
historical_earlier_by_lsoa_type['dataset'] = 'historical_earlier'

combined_by_lsoa_type = pd.concat([recent_by_lsoa_type, historical_by_lsoa_type, historical_earlier_by_lsoa_type])

# Identify the top LSOAs by total residential burglaries
lsoa_totals = combined_by_lsoa_type.groupby(['LSOA Code', 'LSOA Name', 'Borough'])['total'].sum().reset_index()
top_lsoas = lsoa_totals.sort_values('total', ascending=False).head(10)
print("\nTop 10 LSOAs by total residential burglaries:")
print(top_lsoas[['LSOA Name', 'Borough', 'total']])

# Create grouped bar charts for top 10 LSOAs
for i, lsoa in top_lsoas.iterrows():
    lsoa_code = lsoa['LSOA Code']
    lsoa_name = lsoa['LSOA Name']
    borough = lsoa['Borough']
    
    # Filter data for this LSOA
    lsoa_data = combined_by_lsoa_type[combined_by_lsoa_type['LSOA Code'] == lsoa_code]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(lsoa_data['Minor Category'], lsoa_data['total'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:,.0f}', ha='center', va='bottom')
    
    plt.title(f'Residential Burglary Types in {lsoa_name} ({borough})', fontsize=16)
    plt.xlabel('Burglary Type', fontsize=14)
    plt.ylabel('Total Residential Burglaries over 10-15 years', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    safe_name = lsoa_name.replace(' ', '_').replace('/', '_')
    plt.savefig(f'residential_burglary_trends/residential_types_{safe_name}.png', dpi=300)
    plt.close()

# Create a combined plot for all top 10 LSOAs
top_lsoa_codes = top_lsoas['LSOA Code'].tolist()
top_lsoa_data = combined_by_lsoa_type[combined_by_lsoa_type['LSOA Code'].isin(top_lsoa_codes)]

# Pivot the data for better visualization
pivot_data = top_lsoa_data.pivot_table(
    index=['LSOA Name', 'Borough'],
    columns='Minor Category',
    values='total',
    aggfunc='sum'
).fillna(0)

# Create the plot
plt.figure(figsize=(18, 12))
pivot_data.plot(kind='bar', stacked=False, figsize=(18, 12))
plt.title('Residential Burglary Types by LSOA (Top 10 LSOAs)', fontsize=18)
plt.xlabel('LSOA', fontsize=16)
plt.ylabel('Total Residential Burglaries over 10-15 years', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Burglary Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('residential_burglary_trends/top10_lsoas_residential_types.png', dpi=300)
plt.close()

# ----------------------
# Task 2: Line chart of residential burglary types over time
# ----------------------

# Function to calculate monthly totals by burglary type
def calculate_monthly_totals_by_type(df, time_cols):
    # Initialize a dictionary to store the results
    result = defaultdict(lambda: defaultdict(int))
    
    # For each burglary type
    for btype in df['Minor Category'].unique():
        # Filter data for this type
        type_data = df[df['Minor Category'] == btype]
        
        # For each time period, calculate the total
        for col in time_cols:
            if col in df.columns:
                result[btype][col] = type_data[col].sum()
    
    return result

# Calculate monthly totals for residential burglary types
recent_monthly_by_type = calculate_monthly_totals_by_type(recent_residential, recent_time_cols)
historical_monthly_by_type = calculate_monthly_totals_by_type(historical_residential, historical_time_cols)
historical_earlier_monthly_by_type = calculate_monthly_totals_by_type(historical_earlier_residential, historical_earlier_time_cols)

# Combine the results
combined_monthly = defaultdict(lambda: defaultdict(int))

for btype in residential_types:
    for period in all_time_cols:
        # Check if the period exists in any of the datasets and add the count
        if period in recent_monthly_by_type[btype]:
            combined_monthly[btype][period] += recent_monthly_by_type[btype][period]
        if period in historical_monthly_by_type[btype]:
            combined_monthly[btype][period] += historical_monthly_by_type[btype][period]
        if period in historical_earlier_monthly_by_type[btype]:
            combined_monthly[btype][period] += historical_earlier_monthly_by_type[btype][period]

# Convert to DataFrame for plotting
monthly_data = []

for btype in combined_monthly:
    for period in combined_monthly[btype]:
        monthly_data.append({
            'burglary_type': btype,
            'month': period,
            'count': combined_monthly[btype][period]
        })

monthly_df = pd.DataFrame(monthly_data)

# Convert month to datetime for better plotting
monthly_df['month'] = pd.to_datetime(monthly_df['month'], format='%Y%m')
monthly_df = monthly_df.sort_values(['burglary_type', 'month'])

# Create line chart for residential burglary types - smoothed version
plt.figure(figsize=(20, 10))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Use distinct colors for clarity
markers = ['o', 's', '^']
line_styles = ['-', '--', '-.']

for i, btype in enumerate(residential_types):
    type_data = monthly_df[monthly_df['burglary_type'] == btype]
    if not type_data.empty:
        # Add smoothed line (rolling average)
        type_data = type_data.sort_values('month')
        type_data['smooth_count'] = type_data['count'].rolling(window=3, min_periods=1).mean()
        
        plt.plot(type_data['month'], type_data['smooth_count'], 
                 linewidth=2.5, 
                 color=colors[i % len(colors)],
                 marker=markers[i % len(markers)], 
                 markersize=7,
                 linestyle=line_styles[i % len(line_styles)],
                 label=btype)

plt.title('Residential Burglary Types Over Time (10-15 years)', fontsize=18)
plt.xlabel('Month', fontsize=16)
plt.ylabel('Total Burglaries', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(title='Burglary Type', fontsize=14, title_fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('residential_burglary_trends/residential_burglary_trends_smooth.png', dpi=300)
plt.close()

# Create line chart for residential burglary types - raw data
plt.figure(figsize=(20, 10))

for i, btype in enumerate(residential_types):
    type_data = monthly_df[monthly_df['burglary_type'] == btype]
    if not type_data.empty:
        plt.plot(type_data['month'], type_data['count'], 
                 linewidth=2, 
                 color=colors[i % len(colors)],
                 marker=markers[i % len(markers)], 
                 markersize=5,
                 label=btype)

plt.title('Residential Burglary Types Over Time - Raw Data (10-15 years)', fontsize=18)
plt.xlabel('Month', fontsize=16)
plt.ylabel('Total Burglaries', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(title='Burglary Type', fontsize=14, title_fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('residential_burglary_trends/residential_burglary_trends_raw.png', dpi=300)
plt.close()

# Calculate annual totals for clearer long-term trend
monthly_df['year'] = monthly_df['month'].dt.year
annual_totals = monthly_df.groupby(['burglary_type', 'year'])['count'].sum().reset_index()

# Create line chart for residential burglary types by year
plt.figure(figsize=(16, 10))

for i, btype in enumerate(residential_types):
    type_data = annual_totals[annual_totals['burglary_type'] == btype]
    if not type_data.empty:
        plt.plot(type_data['year'], type_data['count'], 
                 linewidth=3.5, 
                 color=colors[i % len(colors)],
                 marker=markers[i % len(markers)], 
                 markersize=10,
                 label=btype)
        
        # Add data points
        for x, y in zip(type_data['year'], type_data['count']):
            plt.text(x, y + 100, f'{y:,.0f}', ha='center', va='bottom', fontsize=9, 
                     color=colors[i % len(colors)], rotation=45)

plt.title('Annual Residential Burglary Trends', fontsize=18)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Total Burglaries', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(title='Burglary Type', fontsize=14, title_fontsize=14)
plt.xticks(annual_totals['year'].unique())
plt.tight_layout()
plt.savefig('residential_burglary_trends/annual_residential_burglary_trends.png', dpi=300)
plt.close()

# Create a stacked area chart showing total residential burglaries
annual_pivot = annual_totals.pivot_table(index='year', columns='burglary_type', values='count').fillna(0)
annual_pivot['Total'] = annual_pivot.sum(axis=1)

plt.figure(figsize=(16, 10))
annual_pivot.drop(columns=['Total']).plot.area(alpha=0.7, figsize=(16, 10), stacked=True)
plt.title('Annual Residential Burglary Trends - Stacked Area Chart', fontsize=18)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Total Burglaries', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(title='Burglary Type', fontsize=14, title_fontsize=14)
plt.tight_layout()
plt.savefig('residential_burglary_trends/stacked_annual_residential_burglary.png', dpi=300)
plt.close()

# Plot the total residential burglaries trend
plt.figure(figsize=(16, 8))
annual_totals_sum = annual_totals.groupby('year')['count'].sum().reset_index()
plt.plot(annual_totals_sum['year'], annual_totals_sum['count'], 
         linewidth=4, marker='o', markersize=10, color='darkblue')

# Add data labels
for x, y in zip(annual_totals_sum['year'], annual_totals_sum['count']):
    plt.text(x, y + 500, f'{y:,.0f}', ha='center', va='bottom', fontsize=12)

plt.title('Total Residential Burglaries by Year (All Types Combined)', fontsize=18)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Total Burglaries', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xticks(annual_totals['year'].unique())
plt.tight_layout()
plt.savefig('residential_burglary_trends/total_residential_burglary_trend.png', dpi=300)
plt.close()

print("\nAnalysis complete! Output saved to 'residential_burglary_trends' directory.") 