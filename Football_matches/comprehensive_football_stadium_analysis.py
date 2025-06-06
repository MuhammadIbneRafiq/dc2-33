import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy import stats
from math import radians, cos, sin, asin, sqrt
warnings.filterwarnings('ignore')

def load_spatial_monthly_data(months_to_load=12*12):  # Last 8 years for comprehensive analysis
    
    data_dir = "../data/cleaned_spatial_monthly_data/cleaned_spatial_monthly_burglary_data"
    spatial_files = []
    for file in os.listdir(data_dir):
        if file.endswith('_burglary_cleaned_spatial.csv'):
            spatial_files.append(file)
    
    spatial_files.sort(reverse=True)  # Most recent first
    spatial_files = spatial_files[:months_to_load]  # Take last N months
        
    sample_file = os.path.join(data_dir, spatial_files[0])
    sample = pd.read_csv(sample_file)
    print(f"Sample file columns: {list(sample.columns)}")
    
    all_data = []
    for file in spatial_files:
        filepath = os.path.join(data_dir, file)
        df = pd.read_csv(filepath)
        
        # Extract date from filename (format: YYYY-MM_burglary_cleaned_spatial.csv)
        date_part = file.replace('_burglary_cleaned_spatial.csv', '')
        date = pd.to_datetime(date_part, format='%Y-%m')
        df['month_date'] = date
        df['source_file'] = file
        
        # Clean coordinate data
        df = df.dropna(subset=['Latitude', 'Longitude'])
        df = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)]
        
        all_data.append(df)
        print(f"Loaded {file}: {len(df)} valid records for {date.strftime('%Y-%m')}")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"Combined dataset: {len(combined)} total burglary records")
    print(f"Date range: {combined['month_date'].min().strftime('%Y-%m')} to {combined['month_date'].max().strftime('%Y-%m')}")
    print(f"Unique LSOAs: {combined['LSOA code'].nunique()}")
    return combined


def define_london_stadiums():    
    return {
        'Arsenal': {
            'venue': 'Emirates Stadium',
            'lat': 51.5549, 'lon': -0.1084,
            'borough': 'Islington',
            'address': 'Highbury, London N5 1BU'
        },
        'Chelsea': {
            'venue': 'Stamford Bridge',
            'lat': 51.4816, 'lon': -0.1909,
            'borough': 'Hammersmith and Fulham', 
            'address': 'Fulham Road, London SW6 1HS'
        },
        'Tottenham': {
            'venue': 'Wembley Stadium',
            'lat': 51.5560, 'lon': -0.2796,
            'borough': 'Haringey',
            'address': 'High Road, London N17 0AP'
        },
        'West Ham': {
            'venue': 'London Stadium',
            'lat': 51.5386, 'lon': -0.0164,
            'borough': 'Newham',
            'address': 'Queen Elizabeth Olympic Park, London E20 2ST'
        },
        'Crystal Palace': {
            'venue': 'Selhurst Park',
            'lat': 51.3983, 'lon': -0.0855,
            'borough': 'Croydon',
            'address': 'Holmesdale Road, London SE25 6PU'
        },
        'Fulham': {
            'venue': 'Craven Cottage',
            'lat': 51.4749, 'lon': -0.2217,
            'borough': 'Hammersmith and Fulham',
            'address': 'Stevenage Road, London SW6 6HH'
        }
    }
    
def calculate_distance(lat1, lon1, lat2, lon2):    
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    return c * r

def identify_stadium_nearby_crimes_enhanced(crime_data, stadiums, radius_km=25.0):
    for team, stadium in stadiums.items():
        distance_col = f'distance_to_{team.lower()}'
        near_col = f'near_{team.lower()}'
        
        # Calculate distances for all crimes
        crime_data[distance_col] = crime_data.apply(
            lambda row: calculate_distance(
                row['Latitude'], row['Longitude'],
                stadium['lat'], stadium['lon']
            ) if pd.notna(row['Latitude']) and pd.notna(row['Longitude']) else np.inf,
            axis=1
        )
        
        # Mark crimes within radius
        crime_data[near_col] = crime_data[distance_col] <= radius_km
        
        nearby_count = crime_data[near_col].sum()
        print(f"  {team}: {nearby_count} burglaries within {radius_km}km")
        
        # Get unique LSOAs near this stadium
        nearby_crimes = crime_data[crime_data[near_col]]
        if not nearby_crimes.empty:
            unique_lsoas = nearby_crimes['LSOA code'].nunique()
            print(f"    - Affecting {unique_lsoas} unique LSOAs")
            
            # Find the closest burglaries
            closest_distances = nearby_crimes[distance_col].nsmallest(10)
            print(f"    - Closest burglary: {closest_distances.iloc[0]:.3f}km")
            print(f"    - Average distance of nearest 10: {closest_distances.mean():.3f}km")
    
    # Identify which stadium each crime is closest to
    distance_columns = [f'distance_to_{team.lower()}' for team in stadiums.keys()]
    crime_data['closest_stadium'] = crime_data[distance_columns].idxmin(axis=1)
    crime_data['closest_stadium'] = crime_data['closest_stadium'].str.replace('distance_to_', '').str.title()
    crime_data['min_stadium_distance'] = crime_data[distance_columns].min(axis=1)
    
    return crime_data

def load_premier_league_data():
    """Load Premier League match data with enhanced date handling"""
    print("Loading Premier League match data...")
    
    try:
        matches = pd.read_csv('prem_matches_cleaned.csv')
        matches['Date'] = pd.to_datetime(matches['Date'])
        matches['year_month'] = matches['Date'].dt.to_period('M')
        matches['match_day'] = matches['Date'].dt.date
        
        # Filter for London teams
        london_teams = ['Arsenal', 'Chelsea', 'Tottenham', 'West Ham', 'Crystal Palace', 'Fulham']
        london_matches = matches[matches['Home'].isin(london_teams)].copy()
        
        print(f"Loaded {len(london_matches)} London home matches")
        print(f"Date range: {london_matches['Date'].min()} to {london_matches['Date'].max()}")
        
        # Create match date lookup for quick checking
        match_dates_by_team = {}
        for team in london_teams:
            team_matches = london_matches[london_matches['Home'] == team]
            match_dates_by_team[team] = set(team_matches['match_day'].tolist())
            print(f"  {team}: {len(team_matches)} home matches")
        
        # Count matches by team and month with additional stats
        match_summary = london_matches.groupby(['Home', 'year_month']).agg({
            'Date': 'count',
            'Home Goals': 'mean',
            'Away Goals': 'mean',
            'Attendance': 'mean'
        }).rename(columns={'Date': 'match_count'}).reset_index()
        
        print(f"Match summary: {len(match_summary)} team-month combinations")
        
        return london_matches, match_summary, match_dates_by_team
        
    except Exception as e:
        print(f"Error loading match data: {e}")
        return None, None, None

def enhanced_burglary_match_analysis(crime_data, matches, match_summary, match_dates_by_team, stadiums):
    """Enhanced analysis of burglary patterns around match days"""
    # Ensure we have the month_date as date for comparison
    crime_data['crime_date'] = crime_data['month_date'].dt.date
    
    results = {}
    detailed_results = []
    
    for team in stadiums.keys():
        print(f"\nAnalyzing {team}...")
        near_col = f'near_{team.lower()}'
        distance_col = f'distance_to_{team.lower()}'
        
        # Get burglaries near this stadium
        team_burglary = crime_data[crime_data[near_col] == True].copy()
        
        # EXCLUDE JUNE AND JULY (months 6 and 7) - summer break
        team_burglary = team_burglary[~team_burglary['month_date'].dt.month.isin([6, 7])]
        
        print(f"  Found {len(team_burglary)} burglaries near {team} stadium (excluding Jun/Jul)")
        
        # Enhanced monthly analysis
        team_burglary['year_month'] = team_burglary['month_date'].dt.to_period('M')
        monthly_burglary = team_burglary.groupby('year_month').agg({
            'Crime ID': 'count',
            distance_col: ['mean', 'min', 'max'],
            'LSOA code': 'nunique'
        }).reset_index()
        
        # Flatten column names
        monthly_burglary.columns = ['year_month', 'burglary_count', 'avg_distance', 'min_distance', 'max_distance', 'unique_lsoas']
        
        # Get team's match data - also exclude June and July
        team_matches = match_summary[match_summary['Home'] == team].copy()
        team_matches = team_matches[~team_matches['year_month'].dt.month.isin([6, 7])]
        
        # Enhanced merge with attendance and goals data
        combined = pd.merge(monthly_burglary, team_matches, on='year_month', how='left')
        combined['match_count'] = combined['match_count'].fillna(0)
        combined['has_matches'] = combined['match_count'] > 0
        combined['avg_attendance'] = combined['Attendance'].fillna(0)
        combined['avg_home_goals'] = combined['Home Goals'].fillna(0)

        # Statistical analysis
        match_months = combined[combined['has_matches']]
        no_match_months = combined[~combined['has_matches']]
        
        if len(match_months) > 0 and len(no_match_months) > 0:
            # Basic statistics
            avg_burglary_match = match_months['burglary_count'].mean()
            avg_burglary_no_match = no_match_months['burglary_count'].mean()
            std_burglary_match = match_months['burglary_count'].std()
            std_burglary_no_match = no_match_months['burglary_count'].std()
            
            # Distance analysis
            avg_distance_match = match_months['avg_distance'].mean() if not match_months['avg_distance'].isna().all() else 0
            avg_distance_no_match = no_match_months['avg_distance'].mean() if not no_match_months['avg_distance'].isna().all() else 0
            
            # LSOA analysis
            avg_lsoas_match = match_months['unique_lsoas'].mean()
            avg_lsoas_no_match = no_match_months['unique_lsoas'].mean()
                        
            # T-test for burglary counts
            t_stat, p_value_ttest = stats.ttest_ind(
                match_months['burglary_count'], 
                no_match_months['burglary_count']
            )
            
            # Mann-Whitney U test (non-parametric)
            u_stat, p_value_mann = stats.mannwhitneyu(
                match_months['burglary_count'], 
                no_match_months['burglary_count'],
                alternative='two-sided'
            )
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(match_months) - 1) * std_burglary_match**2 + 
                                (len(no_match_months) - 1) * std_burglary_no_match**2) / 
                                (len(match_months) + len(no_match_months) - 2))
            cohens_d = (avg_burglary_match - avg_burglary_no_match) / pooled_std if pooled_std > 0 else 0
                        
            # Correlation analysis
            if len(combined) > 5:
                corr_matches_burglary = combined['match_count'].corr(combined['burglary_count'])
                corr_attendance_burglary = combined['avg_attendance'].corr(combined['burglary_count']) if combined['avg_attendance'].sum() > 0 else 0
                corr_goals_burglary = combined['avg_home_goals'].corr(combined['burglary_count']) if combined['avg_home_goals'].sum() > 0 else 0

            # LSOA-level analysis
            team_lsoa_analysis = team_burglary.groupby('LSOA code').agg({
                'Crime ID': 'count',
                distance_col: 'mean',
                'LSOA name': 'first'
            }).reset_index()
            team_lsoa_analysis.columns = ['LSOA_code', 'crime_count', 'avg_distance_to_stadium', 'LSOA_name']
            team_lsoa_analysis = team_lsoa_analysis.sort_values('crime_count', ascending=False)
            
            results[team] = {
                'venue': stadiums[team]['venue'],
                'borough': stadiums[team]['borough'],
                'total_burglaries': len(team_burglary),
                'unique_lsoas_affected': team_burglary['LSOA code'].nunique(),
                'monthly_data': combined,
                'lsoa_analysis': team_lsoa_analysis,
                
                # Basic statistics
                'avg_burglary_match_months': avg_burglary_match,
                'avg_burglary_no_match_months': avg_burglary_no_match,
                'std_burglary_match_months': std_burglary_match,
                'std_burglary_no_match_months': std_burglary_no_match,
                'difference': avg_burglary_match - avg_burglary_no_match,
                'percentage_change': ((avg_burglary_match - avg_burglary_no_match) / avg_burglary_no_match * 100) if avg_burglary_no_match > 0 else 0,
                
                # Distance analysis
                'avg_distance_match_months': avg_distance_match,
                'avg_distance_no_match_months': avg_distance_no_match,
                'distance_difference': avg_distance_match - avg_distance_no_match,
                
                # LSOA analysis
                'avg_lsoas_match_months': avg_lsoas_match,
                'avg_lsoas_no_match_months': avg_lsoas_no_match,
                'lsoa_difference': avg_lsoas_match - avg_lsoas_no_match,
                
                # Statistical tests
                't_statistic': t_stat,
                'p_value_ttest': p_value_ttest,
                'u_statistic': u_stat,
                'p_value_mann_whitney': p_value_mann,
                'cohens_d': cohens_d,
                'effect_size_interpretation': 'Large' if abs(cohens_d) >= 0.8 else 'Medium' if abs(cohens_d) >= 0.5 else 'Small' if abs(cohens_d) >= 0.2 else 'Negligible',
                
                # Significance
                'significant_ttest': p_value_ttest < 0.05,
                'significant_mann_whitney': p_value_mann < 0.05,
                
                # Correlations
                'correlation_matches_burglary': corr_matches_burglary,
                'correlation_attendance_burglary': corr_attendance_burglary,
                'correlation_goals_burglary': corr_goals_burglary,
                
                # Data coverage
                'match_months_count': len(match_months),
                'no_match_months_count': len(no_match_months),
                'data_months': len(combined),
                'data_period': f"{combined['year_month'].min()} to {combined['year_month'].max()}"
            }
            
            # Store detailed results for CSV
            detailed_results.append({
                'Team': team,
                'Venue': stadiums[team]['venue'],
                'Borough': stadiums[team]['borough'],
                'Total_Burglaries': len(team_burglary),
                'Unique_LSOAs': team_burglary['LSOA code'].nunique(),
                'Most_Affected_LSOA': team_lsoa_analysis.iloc[0]['LSOA_code'] if not team_lsoa_analysis.empty else 'N/A',
                'Most_Affected_LSOA_Name': team_lsoa_analysis.iloc[0]['LSOA_name'] if not team_lsoa_analysis.empty else 'N/A',
                'Crimes_in_Most_Affected_LSOA': team_lsoa_analysis.iloc[0]['crime_count'] if not team_lsoa_analysis.empty else 0,
                'Avg_Distance_Most_Affected': team_lsoa_analysis.iloc[0]['avg_distance_to_stadium'] if not team_lsoa_analysis.empty else 0,
                'Data_Months': len(combined),
                'Match_Months': len(match_months),
                'No_Match_Months': len(no_match_months),
                'Avg_Burglary_Match': avg_burglary_match,
                'Avg_Burglary_No_Match': avg_burglary_no_match,
                'Difference': avg_burglary_match - avg_burglary_no_match,
                'Percentage_Change': ((avg_burglary_match - avg_burglary_no_match) / avg_burglary_no_match * 100) if avg_burglary_no_match > 0 else 0,
                'Std_Match': std_burglary_match,
                'Std_No_Match': std_burglary_no_match,
                'Avg_Distance_Match': avg_distance_match,
                'Avg_Distance_No_Match': avg_distance_no_match,
                'Distance_Difference': avg_distance_match - avg_distance_no_match,
                'Avg_LSOAs_Match': avg_lsoas_match,
                'Avg_LSOAs_No_Match': avg_lsoas_no_match,
                'LSOA_Difference': avg_lsoas_match - avg_lsoas_no_match,
                'T_Statistic': t_stat,
                'P_Value_TTest': p_value_ttest,
                'U_Statistic': u_stat,
                'P_Value_Mann_Whitney': p_value_mann,
                'Cohens_D': cohens_d,
                'Effect_Size': 'Large' if abs(cohens_d) >= 0.8 else 'Medium' if abs(cohens_d) >= 0.5 else 'Small' if abs(cohens_d) >= 0.2 else 'Negligible',
                'Significant_TTest': p_value_ttest < 0.05,
                'Significant_Mann_Whitney': p_value_mann < 0.05,
                'Correlation_Matches_Burglary': corr_matches_burglary,
                'Correlation_Attendance_Burglary': corr_attendance_burglary,
                'Correlation_Goals_Burglary': corr_goals_burglary,
                'Data_Period': f"{combined['year_month'].min()} to {combined['year_month'].max()}"
            })
            
            print(f"  Match months avg: {avg_burglary_match:.2f} ± {std_burglary_match:.2f} burglaries")
            print(f"  No-match months avg: {avg_burglary_no_match:.2f} ± {std_burglary_no_match:.2f} burglaries")
            print(f"  Difference: {avg_burglary_match - avg_burglary_no_match:.2f} ({((avg_burglary_match - avg_burglary_no_match) / avg_burglary_no_match * 100):+.1f}%)")
            print(f"  Effect size (Cohen's d): {cohens_d:.3f} ({results[team]['effect_size_interpretation']})")
            print(f"  T-test p-value: {p_value_ttest:.4f}, Mann-Whitney p-value: {p_value_mann:.4f}")
            print(f"  Match-burglary correlation: {corr_matches_burglary:.3f}")
            print(f"  Unique LSOAs affected: {team_burglary['LSOA code'].nunique()}")
            if not team_lsoa_analysis.empty:
                print(f"  Most affected LSOA: {team_lsoa_analysis.iloc[0]['LSOA_name']} ({team_lsoa_analysis.iloc[0]['crime_count']} crimes)")
        else:
            print(f"  Insufficient comparison data for {team}")
    
    return results, detailed_results

def create_comprehensive_visualizations(results, stadiums):
    """Create comprehensive visualizations"""
    print("Creating comprehensive visualizations...")
    
    if not results:
        print("No results to visualize")
        return
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig = plt.figure(figsize=(20, 16))
    
    teams = list(results.keys())
    
    # Plot 1: Time series for each stadium area
    ax1 = plt.subplot(3, 3, 1)
    for team in teams:
        data = results[team]['monthly_data']
        data_sorted = data.sort_values('year_month')
        dates = [pd.to_datetime(str(ym)) for ym in data_sorted['year_month']]
        
        ax1.plot(dates, data_sorted['burglary_count'], marker='o', label=team, linewidth=2, alpha=0.8)
        
        # Highlight match months
        match_data = data_sorted[data_sorted['has_matches']]
        if not match_data.empty:
            match_dates = [pd.to_datetime(str(ym)) for ym in match_data['year_month']]
            ax1.scatter(match_dates, match_data['burglary_count'], 
                       s=60, alpha=0.7, edgecolors='red', facecolors='none', linewidths=2)
    
    ax1.set_title('Monthly Burglary Counts Near Stadiums\n(Red circles = months with home matches)', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Burglary Count')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Match vs No-Match comparison
    ax2 = plt.subplot(3, 3, 2)
    match_avgs = [results[team]['avg_burglary_match_months'] for team in teams]
    no_match_avgs = [results[team]['avg_burglary_no_match_months'] for team in teams]
    
    x = np.arange(len(teams))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, match_avgs, width, label='Match Months', alpha=0.8, color='orange')
    bars2 = ax2.bar(x + width/2, no_match_avgs, width, label='No-Match Months', alpha=0.8, color='skyblue')
    
    ax2.set_title('Average Monthly Burglary\nby Match Status', fontweight='bold')
    ax2.set_xlabel('Team')
    ax2.set_ylabel('Avg Burglary Count')
    ax2.set_xticks(x)
    ax2.set_xticklabels(teams, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax2.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.05, 
                f'{match_avgs[i]:.1f}', ha='center', va='bottom', fontsize=9)
        ax2.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.05, 
                f'{no_match_avgs[i]:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Percentage change
    ax3 = plt.subplot(3, 3, 3)
    pct_changes = [results[team]['percentage_change'] for team in teams]
    p_values = [results[team]['p_value_ttest'] for team in teams]
    colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'gray' for p in p_values]
    
    bars = ax3.bar(teams, pct_changes, color=colors, alpha=0.7)
    ax3.set_title('Percentage Change in Burglary\n(Red=p<0.05, Orange=p<0.1)', fontweight='bold')
    ax3.set_xlabel('Team')
    ax3.set_ylabel('Change (%)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    for bar, pct in zip(bars, pct_changes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.5 if pct > 0 else -1), 
                f'{pct:+.1f}%', ha='center', va='bottom' if pct > 0 else 'top', fontsize=9, fontweight='bold')
    
    # Plot 4: Statistical significance
    ax4 = plt.subplot(3, 3, 4)
    significant = [results[team]['significant_ttest'] for team in teams]
    sig_colors = ['red' if sig else 'gray' for sig in significant]
    
    bars = ax4.bar(teams, [-np.log10(p) for p in p_values], color=sig_colors, alpha=0.7)
    ax4.set_title('Statistical Significance\n(Red = significant)', fontweight='bold')
    ax4.set_xlabel('Team')
    ax4.set_ylabel('-log10(p-value)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
    ax4.legend()
    
    # Plot 5: Correlation analysis
    ax5 = plt.subplot(3, 3, 5)
    correlations = [results[team]['correlation_matches_burglary'] for team in teams]
    corr_colors = ['green' if c > 0.3 else 'red' if c < -0.3 else 'gray' for c in correlations]
    
    bars = ax5.bar(teams, correlations, color=corr_colors, alpha=0.7)
    ax5.set_title('Match Count vs Burglary\nCorrelation', fontweight='bold')
    ax5.set_xlabel('Team')
    ax5.set_ylabel('Correlation Coefficient')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    for bar, corr in zip(bars, correlations):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.02 if corr > 0 else -0.04), 
                f'{corr:.3f}', ha='center', va='bottom' if corr > 0 else 'top', fontsize=9, fontweight='bold')
    
    # Plot 6: Total burglaries by stadium
    ax6 = plt.subplot(3, 3, 6)
    total_burglaries = [results[team]['total_burglaries'] for team in teams]
    
    bars = ax6.bar(teams, total_burglaries, alpha=0.7, color='purple')
    ax6.set_title('Total Burglaries Near Stadium\n(Within 2km radius)', fontweight='bold')
    ax6.set_xlabel('Team')
    ax6.set_ylabel('Total Burglary Count')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3, axis='y')
    
    for bar, total in zip(bars, total_burglaries):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{total}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 7: Data coverage
    ax7 = plt.subplot(3, 3, 7)
    data_months = [results[team]['data_months'] for team in teams]
    match_months = [results[team]['match_months_count'] for team in teams]
    
    x = np.arange(len(teams))
    width = 0.35
    
    bars1 = ax7.bar(x - width/2, data_months, width, label='Total Months', alpha=0.8, color='lightblue')
    bars2 = ax7.bar(x + width/2, match_months, width, label='Match Months', alpha=0.8, color='darkblue')
    
    ax7.set_title('Data Coverage by Team', fontweight='bold')
    ax7.set_xlabel('Team')
    ax7.set_ylabel('Number of Months')
    ax7.set_xticks(x)
    ax7.set_xticklabels(teams, rotation=45)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Plot 8: Summary statistics table
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    summary_data = []
    for team in teams:
        r = results[team]
        summary_data.append([
            team,
            f"{r['avg_burglary_match_months']:.1f}",
            f"{r['avg_burglary_no_match_months']:.1f}",
            f"{r['percentage_change']:+.1f}%",
            f"{r['correlation_matches_burglary']:.3f}",
            "✓" if r['significant_ttest'] else "✗"
        ])
    
    table = ax8.table(cellText=summary_data,
                     colLabels=['Team', 'Match\nAvg', 'No-Match\nAvg', 'Change', 'Corr', 'Sig'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    ax8.set_title('Summary Statistics', fontweight='bold')
    
    # Plot 9: Stadium locations info
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    stadium_info = []
    for team in teams:
        stadium_info.append([
            team,
            stadiums[team]['venue'],
            stadiums[team]['borough'],
            f"{results[team]['total_burglaries']} crimes"
        ])
    
    info_table = ax9.table(cellText=stadium_info,
                          colLabels=['Team', 'Venue', 'Borough', 'Total\nCrimes'],
                          cellLoc='center',
                          loc='center')
    info_table.auto_set_font_size(False)
    info_table.set_fontsize(8)
    info_table.scale(1, 2)
    ax9.set_title('Stadium Information', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('comprehensive_football_stadium_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: comprehensive_football_stadium_analysis.png")
    plt.show()
    
    return fig

def generate_enhanced_comprehensive_report(results, detailed_results, stadiums):
    """Generate comprehensive analysis report with detailed explanations"""
    print("\n" + "="*100)
    print("ENHANCED COMPREHENSIVE FOOTBALL STADIUM BURGLARY ANALYSIS")
    print("8+ Years of London Crime Data with LSOA and Statistical Analysis (2017-2025)")
    print("Excluding June/July (summer break months) from match-burglary analysis")
    print("="*100)
    
    if not results:
        print("No results to report")
        return
    
    print("\nSTADIUM PROXIMITY BURGLARY ANALYSIS")
    print("Analyzing burglary patterns within 2km radius of football stadiums")
    print("Comparing months with home matches vs months without home matches")
    print("Including LSOA-level analysis and enhanced statistical testing\n")
    
    for team, r in results.items():
        print(f"{team.upper()} - {r['venue']} ({r['borough']}):")
        print(f"  Analysis period: {r['data_period']}")
        print(f"  Total months analyzed: {r['data_months']}")
        print(f"  Months with home matches: {r['match_months_count']}")
        print(f"  Months without home matches: {r['no_match_months_count']}")
        print(f"  Total burglaries in area: {r['total_burglaries']}")
        print(f"  Unique LSOAs affected: {r['unique_lsoas_affected']}")
        
        print(f"\n  BURGLARY STATISTICS:")
        print(f"    Match months: {r['avg_burglary_match_months']:.2f} ± {r['std_burglary_match_months']:.2f} burglaries")
        print(f"    No-match months: {r['avg_burglary_no_match_months']:.2f} ± {r['std_burglary_no_match_months']:.2f} burglaries")
        print(f"    Absolute difference: {r['difference']:.2f}")
        print(f"    Percentage change: {r['percentage_change']:+.1f}%")
        
        print(f"\n  DISTANCE ANALYSIS:")
        print(f"    Average distance (match months): {r['avg_distance_match_months']:.3f}km")
        print(f"    Average distance (no-match months): {r['avg_distance_no_match_months']:.3f}km")
        print(f"    Distance difference: {r['distance_difference']:+.3f}km")
        
        print(f"\n  LSOA ANALYSIS:")
        print(f"    Average LSOAs affected (match months): {r['avg_lsoas_match_months']:.2f}")
        print(f"    Average LSOAs affected (no-match months): {r['avg_lsoas_no_match_months']:.2f}")
        print(f"    LSOA difference: {r['lsoa_difference']:+.2f}")
        
        print(f"\n  STATISTICAL TESTS:")
        print(f"    T-test: t={r['t_statistic']:.3f}, p={r['p_value_ttest']:.4f} ({'Significant' if r['significant_ttest'] else 'Not significant'})")
        print(f"    Mann-Whitney U: U={r['u_statistic']:.1f}, p={r['p_value_mann_whitney']:.4f} ({'Significant' if r['significant_mann_whitney'] else 'Not significant'})")
        print(f"    Effect size (Cohen's d): {r['cohens_d']:.3f} ({r['effect_size_interpretation']})")
        
        print(f"\n  CORRELATIONS:")
        print(f"    Match count vs burglary: {r['correlation_matches_burglary']:.3f}")
        print(f"    Attendance vs burglary: {r['correlation_attendance_burglary']:.3f}")
        print(f"    Home goals vs burglary: {r['correlation_goals_burglary']:.3f}")
        
        if not r['lsoa_analysis'].empty:
            top_lsoa = r['lsoa_analysis'].iloc[0]
            print(f"\n  MOST AFFECTED LSOA:")
            print(f"    {top_lsoa['LSOA_name']} ({top_lsoa['LSOA_code']})")
            print(f"    Crimes: {top_lsoa['crime_count']}, Avg distance: {top_lsoa['avg_distance_to_stadium']:.3f}km")
        
        print("\n" + "-"*80)
    
    print("=" * 100)
    print("KEY FINDINGS AND INTERPRETATIONS:")
    print("=" * 100)
    
    # Overall statistics
    avg_change = np.mean([r['percentage_change'] for r in results.values()])
    significant_teams_ttest = [team for team, r in results.items() if r['significant_ttest']]
    significant_teams_mann = [team for team, r in results.items() if r['significant_mann_whitney']]
    positive_correlation_teams = [team for team, r in results.items() if r['correlation_matches_burglary'] > 0.2]
    increase_teams = [team for team, r in results.items() if r['difference'] > 0]
    large_effect_teams = [team for team, r in results.items() if abs(r['cohens_d']) >= 0.5]
    
    print(f"1. OVERALL IMPACT:")
    print(f"   • Average change across all stadiums: {avg_change:+.1f}%")
    print(f"   • Teams with statistically significant differences (T-test): {len(significant_teams_ttest)}/{len(results)}")
    print(f"   • Teams with statistically significant differences (Mann-Whitney): {len(significant_teams_mann)}/{len(results)}")
    if significant_teams_ttest:
        print(f"     - T-test significant: {', '.join(significant_teams_ttest)}")
    if significant_teams_mann:
        print(f"     - Mann-Whitney significant: {', '.join(significant_teams_mann)}")
    
    print(f"\n2. EFFECT SIZES:")
    print(f"   • Teams with large effect sizes (|Cohen's d| ≥ 0.5): {len(large_effect_teams)}")
    if large_effect_teams:
        print(f"     - Large effects: {', '.join(large_effect_teams)}")
    
    print(f"\n3. DIRECTIONAL EFFECTS:")
    if increase_teams:
        print(f"   • Teams with increased burglary during match months: {', '.join(increase_teams)}")
    decrease_teams = [team for team, r in results.items() if r['difference'] < 0]
    if decrease_teams:
        print(f"   • Teams with decreased burglary during match months: {', '.join(decrease_teams)}")
    
    print(f"\n4. CORRELATION ANALYSIS:")
    if positive_correlation_teams:
        print(f"   • Teams with positive match-burglary correlation (>0.2): {', '.join(positive_correlation_teams)}")
    
    # Find strongest effects
    strongest_effect = max(results.items(), key=lambda x: abs(x[1]['percentage_change']))
    largest_cohens_d = max(results.items(), key=lambda x: abs(x[1]['cohens_d']))
    most_burglaries = max(results.items(), key=lambda x: x[1]['total_burglaries'])
    most_lsoas = max(results.items(), key=lambda x: x[1]['unique_lsoas_affected'])
    
    print(f"\n5. EXTREME VALUES:")
    print(f"   • Strongest percentage effect: {strongest_effect[0]} ({strongest_effect[1]['percentage_change']:+.1f}%)")
    print(f"   • Largest effect size: {largest_cohens_d[0]} (Cohen's d = {largest_cohens_d[1]['cohens_d']:.3f})")
    print(f"   • Highest crime area: {most_burglaries[0]} ({most_burglaries[1]['total_burglaries']} total burglaries)")
    print(f"   • Most LSOAs affected: {most_lsoas[0]} ({most_lsoas[1]['unique_lsoas_affected']} unique LSOAs)")
    
    print(f"\n6. STATISTICAL INTERPRETATION:")
    print(f"   • P-value < 0.05: Statistically significant difference")
    print(f"   • Cohen's d interpretation: Negligible (<0.2), Small (0.2-0.5), Medium (0.5-0.8), Large (≥0.8)")
    print(f"   • Positive correlation: More matches associated with more burglaries")
    print(f"   • Negative correlation: More matches associated with fewer burglaries")
    
    print(f"\n7. METHODOLOGY:")
    print(f"   • Radius: 2km around each stadium")
    print(f"   • Time period: Monthly aggregation (excluding June/July summer break)")
    print(f"   • Statistical tests: Independent t-test and Mann-Whitney U test")
    print(f"   • Effect size: Cohen's d for practical significance")
    print(f"   • LSOA analysis: Lower Layer Super Output Area crime distribution")
    
    # Save comprehensive results
    summary_df = pd.DataFrame([
        {
            'Team': team,
            'Venue': r['venue'],
            'Borough': r['borough'],
            'Total_Burglaries': r['total_burglaries'],
            'Unique_LSOAs': r['unique_lsoas_affected'],
            'Data_Months': r['data_months'],
            'Match_Months': r['match_months_count'],
            'No_Match_Months': r['no_match_months_count'],
            'Avg_Burglary_Match': r['avg_burglary_match_months'],
            'Avg_Burglary_No_Match': r['avg_burglary_no_match_months'],
            'Std_Match': r['std_burglary_match_months'],
            'Std_No_Match': r['std_burglary_no_match_months'],
            'Difference': r['difference'],
            'Percentage_Change': r['percentage_change'],
            'Avg_Distance_Match': r['avg_distance_match_months'],
            'Avg_Distance_No_Match': r['avg_distance_no_match_months'],
            'Distance_Difference': r['distance_difference'],
            'Avg_LSOAs_Match': r['avg_lsoas_match_months'],
            'Avg_LSOAs_No_Match': r['avg_lsoas_no_match_months'],
            'LSOA_Difference': r['lsoa_difference'],
            'T_Statistic': r['t_statistic'],
            'P_Value_TTest': r['p_value_ttest'],
            'U_Statistic': r['u_statistic'],
            'P_Value_Mann_Whitney': r['p_value_mann_whitney'],
            'Cohens_D': r['cohens_d'],
            'Effect_Size': r['effect_size_interpretation'],
            'Significant_TTest': r['significant_ttest'],
            'Significant_Mann_Whitney': r['significant_mann_whitney'],
            'Correlation_Matches_Burglary': r['correlation_matches_burglary'],
            'Correlation_Attendance_Burglary': r['correlation_attendance_burglary'],
            'Correlation_Goals_Burglary': r['correlation_goals_burglary'],
            'Data_Period': r['data_period']
        }
        for team, r in results.items()
    ])
    
    # Save detailed results
    detailed_df = pd.DataFrame(detailed_results)
    
    summary_df.to_csv('enhanced_stadium_analysis_summary.csv', index=False)
    detailed_df.to_csv('enhanced_stadium_analysis_detailed.csv', index=False)
    
    print(f"\n• Summary results saved to: Football_matches/enhanced_stadium_analysis_summary.csv")
    print(f"• Detailed results saved to: Football_matches/enhanced_stadium_analysis_detailed.csv")
    
    # Save LSOA analysis for each team
    for team, r in results.items():
        if not r['lsoa_analysis'].empty:
            lsoa_filename = f"{team.lower()}_lsoa_analysis.csv"
            r['lsoa_analysis'].to_csv(lsoa_filename, index=False)
            print(f"• {team} LSOA analysis saved to: {lsoa_filename}")
    
    return summary_df, detailed_df

def run_multi_parameter_analysis():
    """Run analysis with multiple parameter combinations"""
    from datetime import datetime
    import itertools
    
    print("=" * 80)
    print("COMPREHENSIVE MULTI-PARAMETER FOOTBALL STADIUM BURGLARY ANALYSIS")
    print("=" * 80)
    
    # Define parameter combinations to test
    years_options = [5, 8, 10, 15]
    radius_options = [2, 5, 10]  # km
    
    stadiums = define_london_stadiums()
    all_results = {}
    all_detailed_results = {}
    
    # Load match data once (it's the same for all iterations)
    matches, match_summary, match_dates_by_team = load_premier_league_data()
    if matches is None:
        return
    
    print(f"\nRunning analysis for {len(years_options)} year periods × {len(radius_options)} radii = {len(years_options) * len(radius_options)} configurations")
    print(f"Years: {years_options}")
    print(f"Radii: {radius_options} km")
    
    for years in years_options:
        for radius in radius_options:
            config = (years, radius)
            print(f"\n{'='*60}")
            print(f"CONFIGURATION: {years} years of data, {radius}km radius")
            print(f"{'='*60}")
            
            # Load crime data for this time period
            print(f"\nLoading {years} years of burglary data...")
            crime_data = load_spatial_monthly_data(months_to_load=years*12)
            
            if crime_data.empty:
                print(f"No data available for {years} years")
                continue
            
            # Identify crimes near stadiums with this radius
            print(f"\nIdentifying crimes within {radius}km of stadiums...")
            crime_data_enhanced = identify_stadium_nearby_crimes_enhanced(crime_data, stadiums, radius_km=radius)
            
            # Run analysis for this configuration
            results, detailed_results = enhanced_burglary_match_analysis(
                crime_data_enhanced, matches, match_summary, match_dates_by_team, stadiums)
            
            all_results[config] = results
            all_detailed_results[config] = detailed_results
            
            # Create visualizations for this configuration
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig = create_comprehensive_visualizations(results, stadiums)
            
            # Save this configuration's plot
            config_filename = f'analysis_{years}years_{radius}km_{timestamp}.png'
            fig.savefig(f'{config_filename}', dpi=300, bbox_inches='tight')
            print(f"Visualization saved as {config_filename}")
            
            # Print summary for this configuration
            print(f"\nSUMMARY for {years} years, {radius}km radius:")
            for team, stats in results.items():
                sig_marker = "***" if stats['p_value_ttest'] < 0.001 else "**" if stats['p_value_ttest'] < 0.01 else "*" if stats['p_value_ttest'] < 0.05 else ""
                print(f"  {team}: {stats['total_burglaries']} crimes, {stats['percentage_change']:+.1f}% change, p={stats['p_value_ttest']:.3f}{sig_marker}")
    
    # Create comprehensive comparison visualization
    print(f"\n{'='*60}")
    print("CREATING COMPREHENSIVE COMPARISON VISUALIZATION")
    print(f"{'='*60}")
    
    comparison_data = create_multi_parameter_comparison(all_results)
    
    # Save detailed results for all configurations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f'multi_parameter_analysis_results_{timestamp}.csv'
    comparison_data.to_csv(f'{results_filename}', index=False)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nConfigurations with significant results (p < 0.05):")
    significant_results = comparison_data[comparison_data['significant_ttest'] == True]
    if len(significant_results) > 0:
        for _, row in significant_results.iterrows():
            print(f"  {row['team']} - {row['years_of_data']} years, {row['radius_km']}km: {row['percentage_change']:+.1f}% (p={row['p_value_ttest']:.3f})")
    else:
        print("  No statistically significant results found in any configuration")
    
    print(f"\nLargest effects by configuration:")
    top_effects = comparison_data.nlargest(5, 'percentage_change')
    for _, row in top_effects.iterrows():
        sig_marker = "*" if row['significant_ttest'] else ""
        print(f"  {row['team']} - {row['years_of_data']} years, {row['radius_km']}km: {row['percentage_change']:+.1f}% (p={row['p_value_ttest']:.3f}){sig_marker}")
    
    print(f"\nTotal configurations analyzed: {len(all_results)}")
    print(f"Results saved to: {results_filename}")
    
    return all_results, all_detailed_results, comparison_data

def create_multi_parameter_comparison(all_results):
    """Create comparison data from all parameter combinations"""
    comparison_data = []
    
    for config, results in all_results.items():
        years, radius = config
        for team, stats in results.items():
            row = {
                'years_of_data': years,
                'radius_km': radius,
                'team': team,
                'venue': stats['venue'],
                'borough': stats['borough'],
                'total_burglaries': stats['total_burglaries'],
                'unique_lsoas': stats['unique_lsoas_affected'],
                'data_months': stats['data_months'],
                'match_months': stats['match_months_count'],
                'no_match_months': stats['no_match_months_count'],
                'avg_burglary_match': stats['avg_burglary_match_months'],
                'avg_burglary_no_match': stats['avg_burglary_no_match_months'],
                'difference': stats['difference'],
                'percentage_change': stats['percentage_change'],
                't_statistic': stats['t_statistic'],
                'p_value_ttest': stats['p_value_ttest'],
                'p_value_mann_whitney': stats['p_value_mann_whitney'],
                'cohens_d': stats['cohens_d'],
                'effect_size': stats['effect_size_interpretation'],
                'significant_ttest': stats['significant_ttest'],
                'significant_mann_whitney': stats['significant_mann_whitney'],
                'correlation_matches_burglary': stats['correlation_matches_burglary'],
                'correlation_attendance_burglary': stats['correlation_attendance_burglary'],
                'correlation_goals_burglary': stats['correlation_goals_burglary']
            }
            comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def create_comparison_visualization(comparison_df):
    """Create comprehensive comparison visualization"""
    plt.style.use('default')
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('Multi-Parameter Football Stadium Burglary Analysis Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Total crimes by radius and years
    ax1 = axes[0, 0]
    pivot_crimes = comparison_df.pivot_table(values='total_burglaries', index='team', columns=['years_of_data', 'radius_km'], aggfunc='sum')
    sns.heatmap(pivot_crimes, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax1)
    ax1.set_title('Total Crimes by Configuration')
    ax1.set_xlabel('(Years, Radius km)')
    
    # Plot 2: Percent increase by radius
    ax2 = axes[0, 1]
    for years in sorted(comparison_df['years_of_data'].unique()):
        year_data = comparison_df[comparison_df['years_of_data'] == years]
        for team in year_data['team'].unique():
            team_data = year_data[year_data['team'] == team]
            ax2.plot(team_data['radius_km'], team_data['percentage_change'], 
                    marker='o', label=f'{team} ({years}yr)', alpha=0.7)
    ax2.set_xlabel('Radius (km)')
    ax2.set_ylabel('Percent Increase (%)')
    ax2.set_title('Percent Increase vs Radius')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Statistical significance heatmap
    ax3 = axes[0, 2]
    sig_pivot = comparison_df.pivot_table(values='significant_ttest', index='team', columns=['years_of_data', 'radius_km'], aggfunc='first')
    sns.heatmap(sig_pivot.astype(int), annot=True, cmap='RdYlGn', cbar_kws={'label': 'Significant'}, ax=ax3)
    ax3.set_title('Statistical Significance (p < 0.05)')
    ax3.set_xlabel('(Years, Radius km)')
    
    # Plot 4: Effect sizes (Cohen's d)
    ax4 = axes[1, 0]
    effect_pivot = comparison_df.pivot_table(values='cohens_d', index='team', columns=['years_of_data', 'radius_km'], aggfunc='first')
    sns.heatmap(effect_pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax4)
    ax4.set_title("Effect Sizes (Cohen's d)")
    ax4.set_xlabel('(Years, Radius km)')
    
    # Plot 5: P-values heatmap
    ax5 = axes[1, 1]
    p_pivot = comparison_df.pivot_table(values='p_value_ttest', index='team', columns=['years_of_data', 'radius_km'], aggfunc='first')
    sns.heatmap(p_pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax5)
    ax5.set_title('P-values (lower = more significant)')
    ax5.set_xlabel('(Years, Radius km)')
    
    # Plot 6: Crimes vs Years for each radius
    ax6 = axes[1, 2]
    for radius in sorted(comparison_df['radius_km'].unique()):
        radius_data = comparison_df[comparison_df['radius_km'] == radius]
        for team in radius_data['team'].unique():
            team_data = radius_data[radius_data['team'] == team]
            ax6.plot(team_data['years_of_data'], team_data['total_burglaries'], 
                    marker='o', label=f'{team} ({radius}km)', alpha=0.7)
    ax6.set_xlabel('Years of Data')
    ax6.set_ylabel('Total Crimes')
    ax6.set_title('Total Crimes vs Data Period')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Box plot of percent increases by radius
    ax7 = axes[2, 0]
    sns.boxplot(data=comparison_df, x='radius_km', y='percentage_change', ax=ax7)
    ax7.set_title('Distribution of Percent Increases by Radius')
    ax7.set_xlabel('Radius (km)')
    ax7.set_ylabel('Percent Increase (%)')
    
    # Plot 8: Box plot of percent increases by years
    ax8 = axes[2, 1]
    sns.boxplot(data=comparison_df, x='years_of_data', y='percentage_change', ax=ax8)
    ax8.set_title('Distribution of Percent Increases by Years')
    ax8.set_xlabel('Years of Data')
    ax8.set_ylabel('Percent Increase (%)')
    
    # Plot 9: Summary statistics table
    ax9 = axes[2, 2]
    ax9.axis('off')
    
    # Create summary statistics table
    summary_stats = comparison_df.groupby(['years_of_data', 'radius_km']).agg({
        'total_burglaries': 'sum',
        'percentage_change': 'mean',
        'significant_ttest': 'sum',
        'p_value_ttest': 'mean'
    }).round(3)
    
    # Create table text
    table_text = "Summary Statistics by Configuration\n\n"
    table_text += f"{'Years':<6} {'Radius':<8} {'Total':<8} {'Avg %':<8} {'Sig':<4} {'Avg p':<8}\n"
    table_text += "-" * 50 + "\n"
    
    for (years, radius), row in summary_stats.iterrows():
        table_text += f"{years:<6} {radius:<8} {int(row['total_burglaries']):<8} {row['percentage_change']:+.1f}%{'':<3} {int(row['significant_ttest']):<4} {row['p_value_ttest']:.3f}\n"
    
    ax9.text(0.1, 0.9, table_text, transform=ax9.transAxes, fontfamily='monospace',
             verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_filename = f'multi_parameter_comparison_{timestamp}.png'
    plt.savefig(f'{comparison_filename}', dpi=300, bbox_inches='tight')
    print(f"Comprehensive comparison visualization saved as {comparison_filename}")
    
    return fig

def main():
    print("ENHANCED COMPREHENSIVE FOOTBALL STADIUM BURGLARY ANALYSIS")
    print("Including statistical testing, effect sizes, and correlation analysis")
    print("Excluding June/July (summer break) from match-burglary correlations")
    print("="*100)
    
    # Ask user for analysis type
    print("\nChoose analysis type:")
    print("1. Single configuration (original analysis)")
    print("2. Multi-parameter analysis (different radii and time periods)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        all_results, all_detailed_results, comparison_df = run_multi_parameter_analysis()
        
        # Create final comparison visualization
        create_comparison_visualization(comparison_df)
        
        return all_results, all_detailed_results, comparison_df
    else:
        # Original single configuration analysis
        crime_data = load_spatial_monthly_data()  # Last 12 years
        stadiums = define_london_stadiums()
        
        crime_data = identify_stadium_nearby_crimes_enhanced(crime_data, stadiums, radius_km=25.0)
        
        matches, match_summary, match_dates_by_team = load_premier_league_data()
        
        results, detailed_results = enhanced_burglary_match_analysis(
            crime_data, matches, match_summary, match_dates_by_team, stadiums)

        create_comprehensive_visualizations(results, stadiums)
        summary_df, detailed_df = generate_enhanced_comprehensive_report(
            results, detailed_results, stadiums)
        
        print("Generated files:")
        print("- comprehensive_football_stadium_analysis.png")
        print("- enhanced_stadium_analysis_summary.csv")
        print("- enhanced_stadium_analysis_detailed.csv")
        print("- Individual LSOA analysis files for each team")
        
        return results, detailed_results, summary_df, detailed_df

if __name__ == "__main__":
    main() 