import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import folium
from folium.plugins import HeatMap, MarkerCluster
import calendar
from collections import Counter

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Base directory for data storage
base_dir = Path(r'./cleaned_monthly_burglary_data')

def load_cleaned_data():
    """Load all cleaned burglary data files and combine them"""
    all_data = []
    
    # Check if directory exists
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory {base_dir} does not exist")
    
    # Print out the files we're loading
    print(f"Loading files from {base_dir}:")
    for file in base_dir.glob('*.csv'):
        print(f"  - {file.name}")
        df = pd.read_csv(file)
        all_data.append(df)
    
    if not all_data:
        raise ValueError(f"No CSV files found in {base_dir}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Successfully loaded {len(all_data)} files with {len(combined_df)} total records")
    
    return combined_df

def preprocess_data(df):
    """Preprocess the data for analysis"""
    # Convert Month to datetime
    df['Month'] = pd.to_datetime(df['Month'])
    
    # Extract additional date components
    df['Year'] = df['Month'].dt.year
    df['MonthName'] = df['Month'].dt.month_name()
    
    # Create a numerical month for sorting
    month_order = {m: i for i, m in enumerate(calendar.month_name[1:], 1)}
    df['MonthNum'] = df['MonthName'].map(month_order)
    
    # Clean up any missing values
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"Column '{col}' has {missing} missing values")
    
    return df

def analyze_basic_stats(df):
    """Perform basic statistical analysis on the data"""
    
    print("\n=== Basic Dataset Information ===")
    print(f"Total number of records: {len(df)}")
    print("\nColumns in the dataset:")
    print(df.columns.tolist())
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    print("\n=== Summary Statistics ===")
    print(df.describe())
    
    # Count of records by year
    year_counts = df['Year'].value_counts().sort_index()
    print("\n=== Records by Year ===")
    for year, count in year_counts.items():
        print(f"{year}: {count} records ({count/len(df)*100:.1f}%)")

def create_visualizations(df, viz_dir):
    """Create comprehensive visualizations for the burglary data"""
    
    # Create visualizations directory if it doesn't exist
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Time series plots
    plot_time_series(df, viz_dir)
    
    # 2. Location analysis
    plot_location_analysis(df, viz_dir)
    
    # 3. Investigation status analysis
    plot_investigation_status(df, viz_dir)
    
    # 4. Geographic analysis
    plot_geographic_analysis(df, viz_dir)
    
    # 5. Comparative analysis
    plot_comparative_analysis(df, viz_dir)
    
    # 6. Correlation analysis
    plot_correlation_analysis(df, viz_dir)

def plot_time_series(df, viz_dir):
    """Create time series visualizations"""
    
    # Monthly trend of burglaries
    plt.figure(figsize=(14, 7))
    monthly_counts = df.groupby('Month').size()
    plt.plot(monthly_counts.index, monthly_counts.values, marker='o', linestyle='-', color='#2a9d8f', linewidth=2)
    plt.title('Number of Burglaries Over Time', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Burglaries', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / 'burglaries_over_time.png', dpi=300)
    plt.close()
    
    # Seasonal patterns (by month)
    plt.figure(figsize=(14, 7))
    monthly_avg = df.groupby('MonthName')['Crime ID'].count().reindex(calendar.month_name[1:])
    month_names = list(calendar.month_name[1:])
    sns.barplot(x=month_names, y=monthly_avg.values, palette='YlOrRd')
    plt.title('Seasonal Patterns of Burglaries', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average Number of Burglaries', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / 'seasonal_patterns.png', dpi=300)
    plt.close()
    
    # Year over year comparison
    years = sorted(df['Year'].unique())
    if len(years) > 1:
        plt.figure(figsize=(16, 8))
        
        # Group by year and month, then pivot
        year_month_counts = df.groupby(['Year', 'MonthNum']).size().reset_index(name='count')
        year_month_pivot = year_month_counts.pivot(index='MonthNum', columns='Year', values='count')
        
        # Plot each year as a line
        for year in years:
            if year in year_month_pivot.columns:
                plt.plot(year_month_pivot.index, year_month_pivot[year], 
                        marker='o', linewidth=2, label=str(year))
        
        plt.title('Year-over-Year Comparison of Burglaries by Month', fontsize=16)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Number of Burglaries', fontsize=12)
        plt.xticks(range(1, 13), calendar.month_name[1:], rotation=45)
        plt.legend(title='Year')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / 'year_over_year.png', dpi=300)
        plt.close()

def plot_location_analysis(df, viz_dir):
    """Create visualizations related to location types"""
    
    # Location type distribution
    plt.figure(figsize=(14, 8))
    location_counts = df['Location'].value_counts().head(15)
    sns.barplot(x=location_counts.values, y=location_counts.index, palette='Blues_r')
    plt.title('Top 15 Locations for Burglaries', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Location', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / 'location_type_distribution.png', dpi=300)
    plt.close()
    
    # LSOA distribution (top 20)
    plt.figure(figsize=(16, 10))
    lsoa_counts = df.groupby('LSOA name')['Crime ID'].count().sort_values(ascending=False).head(20)
    sns.barplot(x=lsoa_counts.values, y=lsoa_counts.index, palette='Reds_r')
    plt.title('Top 20 LSOA Areas with Most Burglaries', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('LSOA Area', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / 'lsoa_distribution.png', dpi=300)
    plt.close()
    
    # Police force comparison
    plt.figure(figsize=(12, 8))
    police_counts = df['Reported by'].value_counts()
    explode = [0.1] * len(police_counts)
    plt.pie(police_counts.values, labels=police_counts.index, autopct='%1.1f%%', 
            shadow=True, startangle=90, explode=explode)
    plt.title('Distribution of Burglaries by Police Force', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(viz_dir / 'police_force_distribution.png', dpi=300)
    plt.close()

def plot_investigation_status(df, viz_dir):
    """Create visualizations related to investigation status"""
    
    # Investigation status distribution
    plt.figure(figsize=(12, 8))
    status_counts = df['Last outcome category'].value_counts()
    explode = [0.1] + [0] * (len(status_counts) - 1)  # Highlight the most common status
    plt.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', 
            shadow=True, startangle=90, explode=explode)
    plt.title('Investigation Status Distribution', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(viz_dir / 'investigation_status.png', dpi=300)
    plt.close()
    
    # Top 10 investigation statuses (bar chart)
    plt.figure(figsize=(14, 8))
    top_statuses = df['Last outcome category'].value_counts().head(10)
    sns.barplot(x=top_statuses.values, y=top_statuses.index, palette='viridis')
    plt.title('Top 10 Investigation Outcomes', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Outcome', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / 'top_investigation_outcomes.png', dpi=300)
    plt.close()
    
    # Investigation status by year (if multiple years)
    years = sorted(df['Year'].unique())
    if len(years) > 1:
        plt.figure(figsize=(16, 10))
        status_by_year = df.groupby(['Year', 'Last outcome category']).size().reset_index(name='count')
        top_statuses = df['Last outcome category'].value_counts().nlargest(5).index.tolist()
        
        # Filter to top 5 statuses for clarity
        status_by_year = status_by_year[status_by_year['Last outcome category'].isin(top_statuses)]
        
        sns.barplot(x='Year', y='count', hue='Last outcome category', data=status_by_year, palette='Set2')
        plt.title('Top 5 Investigation Outcomes by Year', fontsize=16)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title='Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / 'outcomes_by_year.png', dpi=300)
        plt.close()

def plot_geographic_analysis(df, viz_dir):
    """Create geographic visualizations"""
    
    # Filter rows with valid coordinates
    geo_df = df.dropna(subset=['Latitude', 'Longitude'])
    
    if len(geo_df) > 0:
        # Create a heatmap of burglaries
        map_center = [geo_df['Latitude'].mean(), geo_df['Longitude'].mean()]
        burglary_map = folium.Map(location=map_center, zoom_start=11)
        
        # Add heatmap
        heat_data = [[row['Latitude'], row['Longitude']] for index, row in geo_df.sample(min(10000, len(geo_df))).iterrows()]
        HeatMap(heat_data, radius=15).add_to(burglary_map)
        
        # Save the map
        burglary_map.save(viz_dir / 'burglary_heatmap.html')
        
        # Create a cluster map
        cluster_map = folium.Map(location=map_center, zoom_start=11)
        marker_cluster = MarkerCluster().add_to(cluster_map)
        
        # Sample points to avoid overloading the map
        for idx, row in geo_df.sample(min(1000, len(geo_df))).iterrows():
            popup_text = f"Location: {row['Location']}<br>Date: {row['Month']}<br>Outcome: {row['Last outcome category']}"
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=popup_text,
                icon=folium.Icon(color='red', icon='home')
            ).add_to(marker_cluster)
        
        # Save the cluster map
        cluster_map.save(viz_dir / 'burglary_cluster_map.html')

def plot_comparative_analysis(df, viz_dir):
    """Create comparative analysis visualizations"""
    
    # Comparison of burglary rates across different areas
    plt.figure(figsize=(16, 10))
    area_counts = df.groupby(['Falls within'])['Crime ID'].count().sort_values(ascending=False)
    sns.barplot(x=area_counts.index, y=area_counts.values, palette='YlGnBu')
    plt.title('Comparison of Burglary Counts by Police Force Area', fontsize=16)
    plt.xlabel('Police Force Area', fontsize=12)
    plt.ylabel('Number of Burglaries', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / 'burglary_by_police_area.png', dpi=300)
    plt.close()
    
    # Top 10 LSOA areas with successful prosecutions
    if 'Last outcome category' in df.columns:
        successful_outcomes = ['Suspect charged', 'Awaiting court outcome', 'Court result unavailable', 
                            'Defendant found guilty', 'Defendant sent to Crown Court']
        
        # Filter for successful outcomes
        successful_df = df[df['Last outcome category'].isin(successful_outcomes)]
        
        if len(successful_df) > 0:
            plt.figure(figsize=(16, 10))
            success_by_lsoa = successful_df.groupby('LSOA name').size().sort_values(ascending=False).head(10)
            sns.barplot(x=success_by_lsoa.values, y=success_by_lsoa.index, palette='Greens_r')
            plt.title('Top 10 Areas with Successful Prosecution Outcomes', fontsize=16)
            plt.xlabel('Count of Successful Outcomes', fontsize=12)
            plt.ylabel('LSOA Area', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(viz_dir / 'successful_prosecutions_by_area.png', dpi=300)
            plt.close()

def plot_correlation_analysis(df, viz_dir):
    """Create correlation analysis where applicable"""
    
    # Try to derive numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # If we have multiple numeric columns, create a correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        correlation = df[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Between Numeric Variables', fontsize=16)
        plt.tight_layout()
        plt.savefig(viz_dir / 'correlation_heatmap.png', dpi=300)
        plt.close()

def main():
    print("Loading cleaned burglary data...")
    try:
        df = load_cleaned_data()
        
        print("Preprocessing data...")
        df = preprocess_data(df)
        
        print("Performing basic statistical analysis...")
        analyze_basic_stats(df)
        
        print("Creating visualizations...")
        viz_dir = base_dir.parent / 'visualizations'
        create_visualizations(df, viz_dir)
        
        print("\nAnalysis complete. Visualizations have been saved to", viz_dir)
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()