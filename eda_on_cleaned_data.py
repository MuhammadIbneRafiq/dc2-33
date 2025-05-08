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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import json
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

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
    df['Quarter'] = df['Month'].dt.quarter
    df['YearMonth'] = df['Month'].dt.strftime('%Y-%m')
    
    # Create a numerical month for sorting
    month_order = {m: i for i, m in enumerate(calendar.month_name[1:], 1)}
    df['MonthNum'] = df['MonthName'].map(month_order)
    
    # Clean up any missing values
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"Column '{col}' has {missing} missing values")
    
    # Add day/night classification based on conventions
    if 'Time' in df.columns:
        # If we have time data
        df['DayNight'] = df['Time'].apply(classify_day_night)
    
    # Create bins for time of day if we have hour information
    if 'Hour' in df.columns:
        df['TimeOfDay'] = pd.cut(
            df['Hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['Night (00:00-06:00)', 'Morning (06:00-12:00)', 
                    'Afternoon (12:00-18:00)', 'Evening (18:00-24:00)'],
            include_lowest=True
        )
    
    return df

def classify_day_night(time_str):
    """Classify a time as day or night"""
    try:
        hour = int(time_str.split(':')[0])
        return 'Day' if 7 <= hour < 19 else 'Night'
    except:
        return 'Unknown'

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
    
    # Count by quarter (if spanning multiple years)
    if len(year_counts) > 1:
        print("\n=== Records by Quarter ===")
        quarter_counts = df.groupby(['Year', 'Quarter']).size().unstack()
        print(quarter_counts)

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
    
    # 7. Interactive dashboards (Plotly)
    create_interactive_dashboards(df, viz_dir)
    
    # 8. Statistical tests
    perform_statistical_tests(df, viz_dir)
    
    # 9. Clustering analysis
    perform_clustering_analysis(df, viz_dir)
    
    # 10. Time series forecasting
    perform_time_series_forecasting(df, viz_dir)
    
    # 11. Predictive modeling
    perform_predictive_modeling(df, viz_dir)

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

def create_interactive_dashboards(df, viz_dir):
    """Create interactive dashboards using Plotly"""
    
    # Create subdirectory for interactive visualizations
    interactive_dir = viz_dir / 'interactive'
    interactive_dir.mkdir(exist_ok=True)
    
    # 1. Time Series Dashboard
    create_time_series_dashboard(df, interactive_dir)
    
    # 2. Geographic Dashboard
    create_geographic_dashboard(df, interactive_dir)
    
    # 3. Investigation Outcomes Dashboard
    create_investigation_dashboard(df, interactive_dir)

def create_time_series_dashboard(df, output_dir):
    """Create an interactive time series dashboard"""
    
    # Aggregate data by month
    monthly_data = df.groupby('YearMonth').size().reset_index(name='count')
    monthly_data['Month'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
    
    # Create the time series plot
    fig = px.line(
        monthly_data, 
        x='Month', 
        y='count',
        title='Interactive Time Series of Burglaries',
        labels={'count': 'Number of Burglaries', 'Month': 'Date'},
        markers=True,
        line_shape='linear',
        template='plotly_white'
    )
    
    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        hovermode="x unified"
    )
    
    # Save the figure
    pio.write_html(fig, file=str(output_dir / 'time_series_interactive.html'), auto_open=False)
    
    # Create a heatmap calendar view
    if len(df['Year'].unique()) > 1:
        # Group by year and month
        heatmap_data = df.groupby(['Year', 'MonthNum']).size().reset_index(name='count')
        
        # Create heatmap
        fig = px.imshow(
            heatmap_data.pivot(index='Year', columns='MonthNum', values='count'),
            labels=dict(x="Month", y="Year", color="Burglaries"),
            x=[calendar.month_name[i] for i in range(1, 13)],
            y=heatmap_data['Year'].unique(),
            color_continuous_scale="Viridis",
            title="Monthly Burglary Heatmap by Year",
            template="plotly_white"
        )
        
        # Customize layout
        fig.update_layout(
            xaxis={'side': 'top'},
            coloraxis_colorbar=dict(
                title="Number of<br>Burglaries",
            )
        )
        
        # Save the figure
        pio.write_html(fig, file=str(output_dir / 'burglary_calendar_heatmap.html'), auto_open=False)

def create_geographic_dashboard(df, output_dir):
    """Create an interactive geographic dashboard"""
    
    # Filter rows with valid coordinates
    geo_df = df.dropna(subset=['Latitude', 'Longitude'])
    
    if len(geo_df) > 0:
        # Sample data to avoid browser performance issues
        sample_size = min(5000, len(geo_df))
        geo_sample = geo_df.sample(sample_size)
        
        # Create a scatter map
        fig = px.scatter_mapbox(
            geo_sample,
            lat='Latitude',
            lon='Longitude',
            hover_name='LSOA name',
            hover_data=['Month', 'Last outcome category', 'Location'],
            color='Last outcome category',
            opacity=0.7,
            zoom=9,
            height=800,
            title='Interactive Map of Burglary Locations'
        )
        
        # Update to use open-street-map
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":30,"l":0,"b":0},
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            )
        )
        
        # Save the figure
        pio.write_html(fig, file=str(output_dir / 'interactive_map.html'), auto_open=False)
        
        # Create a density heatmap
        fig = px.density_mapbox(
            geo_sample, 
            lat='Latitude', 
            lon='Longitude', 
            z='Crime ID',  # Using 'Crime ID' to count occurrences
            radius=10,
            center=dict(lat=geo_sample['Latitude'].mean(), lon=geo_sample['Longitude'].mean()),
            zoom=9,
            mapbox_style="open-street-map",
            title='Burglary Hotspot Map',
            opacity=0.8,
            height=800
        )
        
        # Save the figure
        pio.write_html(fig, file=str(output_dir / 'density_heatmap.html'), auto_open=False)

def create_investigation_dashboard(df, output_dir):
    """Create an interactive dashboard for investigation outcomes"""
    
    # Create subplots layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Investigation Outcomes', 
            'Outcomes by Month',
            'Top Areas by Outcome Type',
            'Outcomes Trend Over Time'
        ),
        specs=[
            [{"type": "pie"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ]
    )
    
    # 1. Pie chart of outcomes
    outcome_counts = df['Last outcome category'].value_counts().head(6)
    total = outcome_counts.sum()
    other_count = len(df) - total
    if other_count > 0:
        outcome_counts = outcome_counts.append(pd.Series([other_count], index=['Other']))
    
    fig.add_trace(
        go.Pie(
            labels=outcome_counts.index,
            values=outcome_counts.values,
            textinfo='percent+label',
            hole=0.4,
            marker=dict(line=dict(color='#FFFFFF', width=2))
        ),
        row=1, col=1
    )
    
    # 2. Monthly outcomes (top 5)
    top_outcomes = df['Last outcome category'].value_counts().nlargest(5).index.tolist()
    monthly_outcome = df[df['Last outcome category'].isin(top_outcomes)]
    monthly_outcome = monthly_outcome.groupby(['YearMonth', 'Last outcome category']).size().reset_index(name='count')
    monthly_pivot = monthly_outcome.pivot(index='YearMonth', columns='Last outcome category', values='count').fillna(0)
    
    for outcome in monthly_pivot.columns:
        fig.add_trace(
            go.Bar(
                x=monthly_pivot.index[-12:],  # Last 12 months
                y=monthly_pivot[outcome][-12:],
                name=outcome
            ),
            row=1, col=2
        )
    
    # 3. Top areas by outcome
    top_lsoas = df['LSOA name'].value_counts().nlargest(8).index.tolist()
    lsoa_outcomes = df[df['LSOA name'].isin(top_lsoas)]
    
    # Get the two most common outcomes
    top_2_outcomes = df['Last outcome category'].value_counts().nlargest(2).index.tolist()
    
    # Filter for these outcomes
    lsoa_filtered = lsoa_outcomes[lsoa_outcomes['Last outcome category'].isin(top_2_outcomes)]
    
    # Group and pivot
    lsoa_outcome_counts = lsoa_filtered.groupby(['LSOA name', 'Last outcome category']).size().reset_index(name='count')
    lsoa_pivot = lsoa_outcome_counts.pivot(index='LSOA name', columns='Last outcome category', values='count').fillna(0)
    
    # Plot each outcome as a grouped bar
    for outcome in lsoa_pivot.columns:
        fig.add_trace(
            go.Bar(
                y=lsoa_pivot.index,
                x=lsoa_pivot[outcome],
                name=outcome,
                orientation='h'
            ),
            row=2, col=1
        )
    
    # 4. Time series of outcomes
    outcome_time = df.groupby(['YearMonth', 'Last outcome category']).size().reset_index(name='count')
    
    # Filter for top outcomes
    outcome_time = outcome_time[outcome_time['Last outcome category'].isin(top_outcomes)]
    
    for outcome in top_outcomes:
        outcome_data = outcome_time[outcome_time['Last outcome category'] == outcome]
        fig.add_trace(
            go.Scatter(
                x=outcome_data['YearMonth'],
                y=outcome_data['count'],
                mode='lines+markers',
                name=outcome
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=1000,
        width=1200,
        title_text="Interactive Investigation Outcomes Dashboard",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )
    
    # Update axes
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="LSOA Area", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    fig.update_xaxes(title_text="Month", row=1, col=2)
    fig.update_xaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=2)
    
    # Save the figure
    pio.write_html(fig, file=str(output_dir / 'investigation_dashboard.html'), auto_open=False)

def perform_statistical_tests(df, viz_dir):
    """Perform statistical tests on the data"""
    
    # Create a directory for statistical analysis
    stats_dir = viz_dir / 'statistics'
    stats_dir.mkdir(exist_ok=True)
    
    # Open a file to write statistical results
    with open(stats_dir / 'statistical_analysis.txt', 'w') as f:
        f.write("=== Statistical Analysis of Burglary Data ===\n\n")
        
        # 1. Chi-Square Test for Independence: Location Type vs Investigation Outcome
        if all(col in df.columns for col in ['Location', 'Last outcome category']):
            f.write("1. Chi-Square Test: Location vs Investigation Outcome\n")
            
            # Create a contingency table
            contingency = pd.crosstab(
                df['Location'].apply(lambda x: x if x in df['Location'].value_counts().nlargest(5).index else 'Other'),
                df['Last outcome category'].apply(lambda x: x if x in df['Last outcome category'].value_counts().nlargest(5).index else 'Other')
            )
            
            # Perform chi-square test
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            
            f.write(f"Chi2 value: {chi2:.2f}\n")
            f.write(f"p-value: {p:.4f}\n")
            f.write(f"Degrees of freedom: {dof}\n")
            f.write(f"Result: {'Dependent (reject H0)' if p < 0.05 else 'Independent (fail to reject H0)'}\n\n")
            
            # Create a heatmap of the contingency table
            plt.figure(figsize=(14, 10))
            sns.heatmap(contingency, annot=True, cmap='viridis', fmt='d')
            plt.title('Contingency Table: Location vs Investigation Outcome')
            plt.tight_layout()
            plt.savefig(stats_dir / 'chi_square_heatmap.png', dpi=300)
            plt.close()
        
        # 2. ANOVA: Burglary counts between different years
        if 'Year' in df.columns and len(df['Year'].unique()) > 1:
            f.write("2. ANOVA Test: Burglary Counts Between Years\n")
            
            # Prepare data for ANOVA
            year_month_counts = df.groupby(['Year', 'MonthNum']).size().reset_index(name='count')
            
            # Group data by year for ANOVA
            groups = [year_month_counts[year_month_counts['Year'] == year]['count'] for year in df['Year'].unique()]
            
            # Perform ANOVA
            f_val, p_val = stats.f_oneway(*groups)
            
            f.write(f"F-value: {f_val:.2f}\n")
            f.write(f"p-value: {p_val:.4f}\n")
            f.write(f"Result: {'Significant difference between years (reject H0)' if p_val < 0.05 else 'No significant difference (fail to reject H0)'}\n\n")
            
            # Create a boxplot to visualize the distribution of counts by year
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='Year', y='count', data=year_month_counts)
            plt.title('Distribution of Monthly Burglary Counts by Year')
            plt.xlabel('Year')
            plt.ylabel('Number of Burglaries per Month')
            plt.grid(True, alpha=0.3)
            plt.savefig(stats_dir / 'anova_boxplot.png', dpi=300)
            plt.close()
        
        # 3. Seasonal Decomposition (if time permits)
        f.write("3. Time Series Seasonality Analysis\n")
        # This would be implemented with statsmodels, but requires more complex code
        f.write("Seasonal decomposition results would be shown here in a more complete implementation.\n\n")
        
        # 4. Trend Analysis using Mann-Kendall Test
        f.write("4. Mann-Kendall Trend Test\n")
        f.write("Trend analysis would be implemented in a more complete version.\n\n")

def perform_clustering_analysis(df, viz_dir):
    """Perform clustering analysis using K-means"""
    
    # Only proceed if we have geographic data
    geo_df = df.dropna(subset=['Latitude', 'Longitude'])
    
    if len(geo_df) > 0:
        # Create a directory for clustering results
        cluster_dir = viz_dir / 'clustering'
        cluster_dir.mkdir(exist_ok=True)
        
        # Prepare data for clustering
        features = geo_df[['Latitude', 'Longitude']].copy()
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Determine optimal number of clusters using elbow method
        inertia = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features_scaled)
            inertia.append(kmeans.inertia_)
        
        # Plot elbow method
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertia, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.grid(True, alpha=0.3)
        plt.savefig(cluster_dir / 'elbow_method.png', dpi=300)
        plt.close()
        
        # Choose k=5 for demonstration (in practice, would choose based on elbow plot)
        k = 5
        kmeans = KMeans(n_clusters=k, random_state=42)
        geo_df['Cluster'] = kmeans.fit_predict(features_scaled)
        
        # Get cluster centers
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        centers_df = pd.DataFrame(centers, columns=['Latitude', 'Longitude'])
        centers_df['Cluster'] = range(k)
        
        # Create an interactive cluster map
        fig = px.scatter_mapbox(
            geo_df.sample(min(5000, len(geo_df))),
            lat='Latitude',
            lon='Longitude',
            color='Cluster',
            color_continuous_scale=px.colors.qualitative.Safe,
            hover_data=['Location', 'Last outcome category'],
            opacity=0.7,
            zoom=9,
            height=800,
            title=f'K-means Clustering of Burglary Locations (k={k})'
        )
        
        # Add cluster centers
        for i, row in centers_df.iterrows():
            fig.add_trace(
                go.Scattermapbox(
                    lat=[row['Latitude']],
                    lon=[row['Longitude']],
                    mode='markers',
                    marker=dict(size=15, color='black'),
                    text=f'Cluster {i} Center',
                    name=f'Cluster {i} Center'
                )
            )
        
        # Update layout
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":30,"l":0,"b":0}
        )
        
        # Save the figure
        pio.write_html(fig, file=str(cluster_dir / 'kmeans_clusters.html'), auto_open=False)
        
        # Create a summary of clusters
        cluster_summary = geo_df.groupby('Cluster').agg({
            'Crime ID': 'count',
            'Latitude': 'mean',
            'Longitude': 'mean',
            'Last outcome category': lambda x: x.value_counts().index[0],
            'LSOA name': lambda x: x.value_counts().index[0]
        }).reset_index()
        
        cluster_summary.columns = ['Cluster', 'Count', 'Avg Latitude', 'Avg Longitude', 
                                  'Most Common Outcome', 'Most Common LSOA']
        
        # Save the cluster summary
        cluster_summary.to_csv(cluster_dir / 'cluster_summary.csv', index=False)

def perform_time_series_forecasting(df, viz_dir):
    """Perform time series forecasting on burglary data"""
    try:
        # Import time series forecasting libraries
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.arima.model import ARIMA
        import pmdarima as pm
        
        # Create a directory for forecasting
        forecast_dir = viz_dir / 'forecasting'
        forecast_dir.mkdir(exist_ok=True)
        
        # Get monthly time series data
        monthly_series = df.groupby('Month').size()
        
        # Decompose the time series
        if len(monthly_series) >= 24:  # Need at least 2 years of data for meaningful decomposition
            decomposition = seasonal_decompose(monthly_series, model='multiplicative', period=12)
            
            # Plot the decomposition
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16))
            
            decomposition.observed.plot(ax=ax1)
            ax1.set_title('Observed')
            ax1.set_ylabel('Count')
            
            decomposition.trend.plot(ax=ax2)
            ax2.set_title('Trend')
            ax2.set_ylabel('Trend')
            
            decomposition.seasonal.plot(ax=ax3)
            ax3.set_title('Seasonality')
            ax3.set_ylabel('Factor')
            
            decomposition.resid.plot(ax=ax4)
            ax4.set_title('Residuals')
            ax4.set_ylabel('Residuals')
            
            plt.tight_layout()
            plt.savefig(forecast_dir / 'time_series_decomposition.png', dpi=300)
            plt.close()
        
        # Auto ARIMA model fitting
        if len(monthly_series) >= 18:  # Need sufficient data for meaningful ARIMA
            # Fit auto ARIMA model
            model = pm.auto_arima(
                monthly_series,
                start_p=1, start_q=1,
                max_p=3, max_q=3,
                m=12,  # Monthly seasonality
                seasonal=True,
                d=None, D=1,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            
            # Forecast next 12 months
            n_forecast = 12
            forecast, conf_int = model.predict(n_periods=n_forecast, return_conf_int=True)
            
            # Create date range for forecast
            last_date = monthly_series.index[-1]
            future_dates = pd.date_range(start=last_date, periods=n_forecast + 1, freq='MS')[1:]
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'forecast': forecast,
                'lower_ci': conf_int[:, 0],
                'upper_ci': conf_int[:, 1]
            }, index=future_dates)
            
            # Plot the forecast
            plt.figure(figsize=(14, 8))
            
            # Plot historical data
            plt.plot(monthly_series.index, monthly_series.values, color='#2a9d8f', label='Historical Data')
            
            # Plot forecast
            plt.plot(forecast_df.index, forecast_df['forecast'], color='#e76f51', label='Forecast')
            
            # Plot confidence interval
            plt.fill_between(
                forecast_df.index,
                forecast_df['lower_ci'],
                forecast_df['upper_ci'],
                color='#e76f51', alpha=0.2,
                label='95% Confidence Interval'
            )
            
            # Add labels and title
            plt.title('12-Month Forecast of Burglary Cases', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Number of Burglaries', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(forecast_dir / 'burglary_forecast.png', dpi=300)
            plt.close()
            
            # Save the forecast data
            forecast_df.to_csv(forecast_dir / 'burglary_forecast.csv')
            
            # Create interactive forecast plot with Plotly
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(
                go.Scatter(
                    x=monthly_series.index,
                    y=monthly_series.values,
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='#2a9d8f')
                )
            )
            
            # Add forecast
            fig.add_trace(
                go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['forecast'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#e76f51')
                )
            )
            
            # Add confidence interval
            fig.add_trace(
                go.Scatter(
                    x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
                    y=forecast_df['upper_ci'].tolist() + forecast_df['lower_ci'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(231, 111, 81, 0.2)',
                    line=dict(color='rgba(255, 255, 255, 0)'),
                    name='95% Confidence Interval'
                )
            )
            
            # Update layout
            fig.update_layout(
                title='Interactive 12-Month Forecast of Burglary Cases',
                xaxis_title='Date',
                yaxis_title='Number of Burglaries',
                hovermode="x unified",
                template="plotly_white"
            )
            
            # Save the interactive plot
            pio.write_html(fig, file=str(forecast_dir / 'interactive_forecast.html'), auto_open=False)
    
    except ImportError:
        print("Forecasting libraries not available. Install statsmodels and pmdarima for time series forecasting.")
    except Exception as e:
        print(f"Error in time series forecasting: {e}")

def perform_predictive_modeling(df, viz_dir):
    """Perform predictive modeling to identify factors associated with successful prosecutions"""
    try:
        # Import modeling libraries
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        import joblib
        
        # Create a directory for predictive modeling
        model_dir = viz_dir / 'predictive_models'
        model_dir.mkdir(exist_ok=True)
        
        # Define successful outcomes
        successful_outcomes = ['Suspect charged', 'Awaiting court outcome', 'Court result unavailable', 
                               'Defendant found guilty', 'Defendant sent to Crown Court']
        
        # Create binary target variable: successful prosecution or not
        df['Successful'] = df['Last outcome category'].apply(lambda x: 1 if x in successful_outcomes else 0)
        
        # Select features for the model
        categorical_features = ['LSOA name', 'Location', 'Reported by', 'Falls within']
        temporal_features = ['MonthNum', 'Year']
        
        # Check if all features exist in the dataframe
        available_cat_features = [f for f in categorical_features if f in df.columns]
        available_temp_features = [f for f in temporal_features if f in df.columns]
        
        # Only proceed if we have features and target
        if len(available_cat_features + available_temp_features) > 0 and 'Successful' in df.columns:
            # Prepare features and target
            X = df[available_cat_features + available_temp_features].copy()
            y = df['Successful']
            
            # Handle missing values in features
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = X[col].fillna('Unknown')
                else:
                    X[col] = X[col].fillna(-1)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Define preprocessing for categorical features
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            # Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, available_cat_features)
                ],
                remainder='passthrough'
            )
            
            # Create a pipeline with preprocessing and model
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Save model results
            with open(model_dir / 'model_performance.txt', 'w') as f:
                f.write("=== Random Forest Model for Predicting Successful Prosecutions ===\n\n")
                f.write(f"Accuracy: {accuracy:.4f}\n\n")
                f.write("Confusion Matrix:\n")
                f.write(f"{conf_matrix}\n\n")
                f.write("Classification Report:\n")
                for label, metrics in class_report.items():
                    if isinstance(metrics, dict):
                        f.write(f"{label}:\n")
                        for metric_name, value in metrics.items():
                            f.write(f"  {metric_name}: {value:.4f}\n")
                    else:
                        f.write(f"{label}: {metrics:.4f}\n")
            
            # Create confusion matrix visualization
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Unsuccessful', 'Successful'],
                        yticklabels=['Unsuccessful', 'Successful'])
            plt.title('Confusion Matrix', fontsize=16)
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('Actual', fontsize=12)
            plt.tight_layout()
            plt.savefig(model_dir / 'confusion_matrix.png', dpi=300)
            plt.close()
            
            # Extract feature importances
            if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
                # Get feature names after one-hot encoding
                feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                
                # Get feature importances
                importances = pipeline.named_steps['classifier'].feature_importances_
                
                # Create a dataframe of feature importances
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Save feature importances
                feature_importance_df.to_csv(model_dir / 'feature_importances.csv', index=False)
                
                # Plot top 20 feature importances
                plt.figure(figsize=(12, 10))
                sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
                plt.title('Top 20 Feature Importances', fontsize=16)
                plt.xlabel('Importance', fontsize=12)
                plt.ylabel('Feature', fontsize=12)
                plt.tight_layout()
                plt.savefig(model_dir / 'feature_importances.png', dpi=300)
                plt.close()
                
                # Create interactive feature importance plot
                fig = px.bar(
                    feature_importance_df.head(20),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Top 20 Features for Predicting Successful Prosecutions',
                    template='plotly_white'
                )
                
                # Save the interactive plot
                pio.write_html(fig, file=str(model_dir / 'interactive_feature_importance.html'), auto_open=False)
            
            # Save the model
            joblib.dump(pipeline, model_dir / 'random_forest_model.pkl')
            
    except ImportError:
        print("Modeling libraries not available. Install scikit-learn for predictive modeling.")
    except Exception as e:
        print(f"Error in predictive modeling: {e}")

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
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()