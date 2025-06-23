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

# Find all unique time columns across the three datasets
all_time_cols = list(set(recent_time_cols + historical_time_cols + historical_earlier_time_cols))
all_time_cols.sort()  # Sort to ensure chronological order

# Get the exact time range
min_date = min(all_time_cols)
max_date = max(all_time_cols)
min_year_month = f"{min_date[:4]}-{min_date[4:]}"
max_year_month = f"{max_date[:4]}-{max_date[4:]}"
time_range_str = f"April 2010 to April 2025 ({min_year_month} to {max_year_month})"

print(f"\nExact time range: {time_range_str}")

# Filter for TRUE residential burglary types
burglary_categories_recent = recent_data[recent_data['Major Category'] == 'BURGLARY']['Minor Category'].unique()
burglary_categories_historical = historical_data[historical_data['Major Category'] == 'BURGLARY']['Minor Category'].unique()
burglary_categories_earlier = historical_earlier_data[historical_data['Major Category'] == 'BURGLARY']['Minor Category'].unique()

all_burglary_categories = list(set(list(burglary_categories_recent) + list(burglary_categories_historical) + list(burglary_categories_earlier)))
residential_types = [cat for cat in all_burglary_categories if ('RESIDENTIAL' in cat or 'DWELLING' in cat) and 'NON' not in cat]

# Combine the datasets
common_cols = ['LSOA Code', 'LSOA Name', 'Borough', 'Major Category', 'Minor Category']

# For recent data, keep only the common columns and the time columns
recent_subset = recent_data[common_cols + recent_time_cols]
historical_subset = historical_data[common_cols + historical_time_cols]
historical_earlier_subset = historical_earlier_data[common_cols + historical_earlier_time_cols]

# Filter for residential burglary data only
def filter_residential_burglary(df):
    return df[(df['Major Category'] == 'BURGLARY') & 
              (df['Minor Category'].isin(residential_types))]

recent_residential = filter_residential_burglary(recent_subset)
historical_residential = filter_residential_burglary(historical_subset)
historical_earlier_residential = filter_residential_burglary(historical_earlier_subset)

# Aggregate the data by LSOA and burglary type
def aggregate_by_lsoa_and_type(df, time_cols):
    df_copy = df.copy()
    df_copy['total'] = df_copy[time_cols].sum(axis=1)
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

# Get LSOA data for top 10
top_lsoa_codes = top_lsoas['LSOA Code'].tolist()
top_lsoa_data = combined_by_lsoa_type[combined_by_lsoa_type['LSOA Code'].isin(top_lsoa_codes)]

# Pivot the data for better visualization
pivot_data = top_lsoa_data.pivot_table(
    index=['LSOA Name', 'Borough', 'LSOA Code'],  # Include LSOA Code in index to keep it for labels
    columns='Minor Category',
    values='total',
    aggfunc='sum'
).fillna(0)

# Prepare for plotting
plt.figure(figsize=(18, 12))
ax = pivot_data.plot(kind='bar', stacked=False, figsize=(18, 12))

# Update labels and title
plt.title('Residential Burglary Types by LSOA (Top 10 LSOAs)', fontsize=18)
plt.xlabel('LSOA', fontsize=16)
plt.ylabel(f'Total Residential Burglaries ({time_range_str})', fontsize=16)  # Updated y-axis label
plt.xticks(rotation=45, ha='right')
plt.legend(title='Burglary Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Improve x-axis labels to show LSOA code without Borough
x_labels = []
for idx in pivot_data.index:
    lsoa_name, borough, lsoa_code = idx
    x_labels.append(f"({lsoa_code[-4:]}) {lsoa_name}\n{borough}")

ax.set_xticklabels(x_labels)

# Increase figure size to accommodate the legend and labels
plt.gcf().set_size_inches(20, 12)
plt.tight_layout()

# Save the updated plot
plt.savefig('residential_burglary_trends/updated_top10_lsoas_residential_types.png', dpi=300)
plt.close()

print("\nUpdated plot saved to 'residential_burglary_trends/updated_top10_lsoas_residential_types.png'") 