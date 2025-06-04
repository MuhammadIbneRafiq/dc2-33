import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_cleaned_monthly_data():
    """Load existing cleaned monthly data"""
    print("Loading cleaned monthly crime data...")
    
    data_dir = "./data/cleaned_spatial_monthly_data"
    files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                files.append(file)
    
    if not files:
        print("No cleaned monthly data files found")
        return None
    
    print(f"Found {len(files)} monthly data files")
    
    # Load a sample file to understand structure
    sample_file = os.path.join(data_dir, files[0])
    sample = pd.read_csv(sample_file)
    print(f"Sample file columns: {list(sample.columns)}")
    
    # Load multiple months for analysis
    all_data = []
    for file in sorted(files)[-12:]:  # Last 12 months
        filepath = os.path.join(data_dir, file)
        df = pd.read_csv(filepath)
        # Extract date from filename
        if 'cleaned_' in file:
            date_part = file.replace('cleaned_', '').replace('.csv', '')
            try:
                date = pd.to_datetime(date_part, format='%Y_%m')
                df['date'] = date
                df['file'] = file
                all_data.append(df)
                print(f"Loaded {file}: {len(df)} records")
            except:
                continue
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"Combined dataset: {len(combined)} records")
        return combined
    else:
        print("Could not load any monthly data")
        return None

def define_stadium_areas():
    """Define London football stadiums and approximate nearby areas"""
    
    # Using known LSOA prefixes that are likely near stadiums
    stadium_areas = {
        'Arsenal': {
            'venue': 'Emirates Stadium',
            'lsoa_prefixes': ['E01002760', 'E01002761', 'E01002762', 'E01002763'],
            'borough': 'Islington'
        },
        'Chelsea': {
            'venue': 'Stamford Bridge',
            'lsoa_prefixes': ['E01002943', 'E01002944', 'E01002945', 'E01002946'],
            'borough': 'Hammersmith and Fulham'
        },
        'Tottenham': {
            'venue': 'Tottenham Hotspur Stadium', 
            'lsoa_prefixes': ['E01006726', 'E01006727', 'E01006728', 'E01006729'],
            'borough': 'Haringey'
        },
        'West Ham': {
            'venue': 'London Stadium',
            'lsoa_prefixes': ['E01004321', 'E01004322', 'E01004323', 'E01004324'],
            'borough': 'Newham'
        }
    }
    
    return stadium_areas

def load_premier_league_matches():
    """Load Premier League match data"""
    print("Loading Premier League matches...")
    
    try:
        matches = pd.read_csv('prem_matches_cleaned.csv')
        matches['Date'] = pd.to_datetime(matches['Date'])
        
        # Filter for London teams
        london_teams = ['Arsenal', 'Chelsea', 'Tottenham', 'West Ham']
        london_matches = matches[matches['Home'].isin(london_teams)].copy()
        
        print(f"Loaded {len(london_matches)} London home matches")
        print(f"Date range: {london_matches['Date'].min()} to {london_matches['Date'].max()}")
        
        return london_matches
        
    except Exception as e:
        print(f"Error loading matches: {e}")
        return None

def analyze_stadium_proximity_burglary(crime_data, stadium_areas, matches):
    """Analyze burglary patterns near stadiums vs match schedule"""
    print("\nAnalyzing stadium proximity burglary patterns...")
    
    results = {}
    
    for team, info in stadium_areas.items():
        print(f"\nAnalyzing {team} ({info['venue']})...")
        
        # Filter crime data for this team's area
        team_crime = crime_data[
            crime_data['LSOA_Code'].str.startswith(tuple(info['lsoa_prefixes'])) |
            crime_data['Borough'].str.contains(info['borough'], case=False, na=False)
        ].copy()
        
        # Filter for burglary crimes
        team_burglary = team_crime[
            team_crime['Major_Category'].str.contains('Burglary', case=False, na=False)
        ].copy()
        
        if team_burglary.empty:
            print(f"  No burglary data found for {team}")
            continue
        
        print(f"  Found {len(team_burglary)} burglary records")
        
        # Get team matches
        team_matches = matches[matches['Home'] == team].copy()
        
        if team_matches.empty:
            print(f"  No match data found for {team}")
            continue
        
        # Group by month
        monthly_burglary = team_burglary.groupby('date')['Crime_Count'].sum().reset_index()
        monthly_burglary['year_month'] = monthly_burglary['date'].dt.to_period('M')
        
        # Mark months with matches
        team_matches['year_month'] = team_matches['Date'].dt.to_period('M')
        match_months = set(team_matches['year_month'])
        
        monthly_burglary['has_matches'] = monthly_burglary['year_month'].isin(match_months)
        
        # Calculate statistics
        match_data = monthly_burglary[monthly_burglary['has_matches']]
        no_match_data = monthly_burglary[~monthly_burglary['has_matches']]
        
        if len(match_data) > 0 and len(no_match_data) > 0:
            avg_match = match_data['Crime_Count'].mean()
            avg_no_match = no_match_data['Crime_Count'].mean()
            
            # Statistical test
            try:
                from scipy import stats
                stat, p_value = stats.ttest_ind(match_data['Crime_Count'], no_match_data['Crime_Count'])
            except:
                p_value = 1.0
            
            results[team] = {
                'monthly_data': monthly_burglary,
                'avg_burglary_match': avg_match,
                'avg_burglary_no_match': avg_no_match,
                'difference': avg_match - avg_no_match,
                'pct_change': ((avg_match - avg_no_match) / avg_no_match * 100) if avg_no_match > 0 else 0,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'match_months': len(match_data),
                'no_match_months': len(no_match_data),
                'total_records': len(team_burglary),
                'venue': info['venue']
            }
            
            print(f"  Match months average: {avg_match:.1f} burglaries")
            print(f"  No-match months average: {avg_no_match:.1f} burglaries")
            print(f"  Difference: {avg_match - avg_no_match:.1f} ({((avg_match - avg_no_match) / avg_no_match * 100):+.1f}%)")
            print(f"  P-value: {p_value:.4f}")
        else:
            print(f"  Insufficient data for statistical analysis")
    
    return results

def create_stadium_burglary_visualization(results):
    """Create comprehensive visualizations"""
    print("\nCreating visualizations...")
    
    if not results:
        print("No results to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Football Stadium Proximity Burglary Analysis\nLSOA-Level Analysis of Match Day Effects', 
                 fontsize=16, fontweight='bold')
    
    teams = list(results.keys())
    
    # Plot 1: Time series comparison
    ax = axes[0, 0]
    for team in teams:
        data = results[team]['monthly_data']
        ax.plot(data['date'], data['Crime_Count'], marker='o', label=f"{team}", linewidth=2, alpha=0.8)
        
        # Highlight match months
        match_data = data[data['has_matches']]
        if not match_data.empty:
            ax.scatter(match_data['date'], match_data['Crime_Count'], 
                      s=80, alpha=0.7, edgecolors='red', facecolors='none', linewidths=3)
    
    ax.set_title('Monthly Burglary Counts by Stadium Area\n(Red circles = months with home matches)', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Monthly Burglary Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Match vs No-Match comparison
    ax = axes[0, 1]
    match_avgs = [results[team]['avg_burglary_match'] for team in teams]
    no_match_avgs = [results[team]['avg_burglary_no_match'] for team in teams]
    
    x = np.arange(len(teams))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, match_avgs, width, label='Months with Home Matches', 
                   alpha=0.8, color='orange')
    bars2 = ax.bar(x + width/2, no_match_avgs, width, label='Months without Home Matches', 
                   alpha=0.8, color='skyblue')
    
    ax.set_title('Average Monthly Burglary by Match Status', fontweight='bold')
    ax.set_xlabel('Team')
    ax.set_ylabel('Average Burglary Count')
    ax.set_xticks(x)
    ax.set_xticklabels(teams, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.1, 
               f'{match_avgs[i]:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.1, 
               f'{no_match_avgs[i]:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Percentage change and significance
    ax = axes[1, 0]
    pct_changes = [results[team]['pct_change'] for team in teams]
    p_values = [results[team]['p_value'] for team in teams]
    colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'gray' for p in p_values]
    
    bars = ax.bar(teams, pct_changes, color=colors, alpha=0.7)
    ax.set_title('Percentage Change in Burglary\n(Red=p<0.05, Orange=p<0.1, Gray=n.s.)', fontweight='bold')
    ax.set_xlabel('Team')
    ax.set_ylabel('Percentage Change (%)')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value and p-value labels
    for i, (bar, pct, p) in enumerate(zip(bars, pct_changes, p_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.5 if pct > 0 else -1.5), 
               f'{pct:+.1f}%\np={p:.3f}', ha='center', va='bottom' if pct > 0 else 'top', 
               fontsize=9, fontweight='bold')
    
    # Plot 4: Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_data = []
    for team in teams:
        r = results[team]
        summary_data.append([
            team,
            f"{r['avg_burglary_match']:.1f}",
            f"{r['avg_burglary_no_match']:.1f}",
            f"{r['pct_change']:+.1f}%",
            f"{r['p_value']:.3f}",
            "✓" if r['significant'] else "✗"
        ])
    
    table = ax.table(cellText=summary_data,
                    colLabels=['Team', 'Match\nAvg', 'No-Match\nAvg', 'Change', 'P-Value', 'Sig.'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color significant results
    for i, team in enumerate(teams):
        if results[team]['significant']:
            for j in range(6):
                table[(i+1, j)].set_facecolor('#ffcccc')
    
    ax.set_title('Statistical Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('simplified_football_burglary_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_detailed_report(results):
    """Generate detailed analysis report"""
    print("\n" + "="*80)
    print("SIMPLIFIED FOOTBALL STADIUM BURGLARY ANALYSIS REPORT")
    print("="*80)
    
    if not results:
        print("No results to report")
        return
    
    print(f"\nAnalyzed {len(results)} stadium areas using LSOA-level crime data")
    print("Comparing burglary rates in months with vs without home football matches\n")
    
    for team, r in results.items():
        print(f"{team.upper()} ({r['venue']}):")
        print(f"  Total burglary records analyzed: {r['total_records']}")
        print(f"  Months with home matches: {r['match_months']}")
        print(f"  Months without home matches: {r['no_match_months']}")
        print(f"  Average burglary (match months): {r['avg_burglary_match']:.2f}")
        print(f"  Average burglary (no-match months): {r['avg_burglary_no_match']:.2f}")
        print(f"  Absolute difference: {r['difference']:.2f}")
        print(f"  Percentage change: {r['pct_change']:+.1f}%")
        print(f"  Statistical significance: p={r['p_value']:.4f} ({'Significant' if r['significant'] else 'Not significant'})")
        print()
    
    print("KEY FINDINGS:")
    print("-" * 40)
    
    # Overall statistics
    avg_change = np.mean([r['pct_change'] for r in results.values()])
    significant_teams = [team for team, r in results.items() if r['significant']]
    increase_teams = [team for team, r in results.items() if r['difference'] > 0]
    
    print(f"• Average change across all stadiums: {avg_change:+.1f}%")
    print(f"• Teams with statistically significant differences: {len(significant_teams)}/{len(results)}")
    
    if significant_teams:
        print(f"  - Significant teams: {', '.join(significant_teams)}")
    
    if increase_teams:
        print(f"• Teams with increased burglary during match months: {', '.join(increase_teams)}")
    
    decrease_teams = [team for team, r in results.items() if r['difference'] < 0]
    if decrease_teams:
        print(f"• Teams with decreased burglary during match months: {', '.join(decrease_teams)}")
    
    # Save results
    summary_df = pd.DataFrame([
        {
            'Team': team,
            'Venue': r['venue'],
            'Total_Records': r['total_records'],
            'Match_Months': r['match_months'],
            'No_Match_Months': r['no_match_months'],
            'Avg_Burglary_Match': r['avg_burglary_match'],
            'Avg_Burglary_No_Match': r['avg_burglary_no_match'],
            'Difference': r['difference'],
            'Percentage_Change': r['pct_change'],
            'P_Value': r['p_value'],
            'Significant': r['significant']
        }
        for team, r in results.items()
    ])
    
    summary_df.to_csv('simplified_football_burglary_results.csv', index=False)
    print(f"\n• Detailed results saved to: simplified_football_burglary_results.csv")

def main():
    """Main analysis function"""
    print("SIMPLIFIED FOOTBALL STADIUM BURGLARY ANALYSIS")
    print("Using existing cleaned monthly crime data")
    print("="*60)
    
    # Load data
    crime_data = load_cleaned_monthly_data()
    if crime_data is None:
        print("Cannot proceed without crime data")
        return
    
    stadium_areas = define_stadium_areas()
    matches = load_premier_league_matches()
    
    if matches is None:
        print("Cannot proceed without match data")
        return
    
    # Analyze
    results = analyze_stadium_proximity_burglary(crime_data, stadium_areas, matches)
    
    if not results:
        print("No analysis results generated")
        return
    
    # Visualize and report
    create_stadium_burglary_visualization(results)
    generate_detailed_report(results)
    
    print("\nAnalysis completed successfully!")
    print("Generated files:")
    print("- simplified_football_burglary_analysis.png")
    print("- simplified_football_burglary_results.csv")

if __name__ == "__main__":
    main() 