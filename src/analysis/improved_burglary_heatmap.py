import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import calendar
import plotly.express as px
import plotly.io as pio
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')

# Base directory for data storage
base_dir = Path(r'./cleaned_monthly_burglary_data')
spatial_base_dir = Path(r'./cleaned_spatial_monthly_burglary_data')
output_dir = Path(r'./eda_on_cleaned_data')

def load_cleaned_data(use_spatial=False):
    """Load all cleaned burglary data files and combine them"""
    all_data = []
    
    directory = spatial_base_dir if use_spatial else base_dir
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    print(f"Loading files from {directory}:")
    for file in directory.glob('*.csv'):
        print(f"  - {file.name}")
        df = pd.read_csv(file)
        all_data.append(df)
    
    if not all_data:
        raise ValueError(f"No CSV files found in {directory}")
    
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
    df['MonthNum'] = df['Month'].dt.month
    df['YearMonth'] = df['Month'].dt.strftime('%Y-%m')
    
    return df

def create_improved_year_month_heatmap(df, output_dir, interactive=True, show_annotations=False):
    """Create an improved heatmap showing burglary patterns by year and month"""
    # Create the directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create interactive directory if needed
    if interactive:
        interactive_dir = output_dir / 'interactive'
        interactive_dir.mkdir(exist_ok=True)
    
    # Group by year and month
    heatmap_data = df.groupby(['Year', 'MonthNum']).size().reset_index(name='count')
    
    # Create pivot table
    heatmap_pivot = heatmap_data.pivot(index='Year', columns='MonthNum', values='count')
    
    # Create static heatmap with improved colors
    plt.figure(figsize=(16, 10))
    
    # Use YlOrBr (Yellow-Orange-Brown) color map with reversed=False to make higher values darker
    sns.heatmap(
        heatmap_pivot, 
        cmap='YlOrBr', 
        annot=show_annotations,  # Don't show annotations (numbers)
        fmt='.0f' if show_annotations else '', 
        cbar_kws={'label': 'Number of Burglaries'},
        linewidths=0.5
    )
    
    plt.title('Monthly Burglary Heatmap by Year', fontsize=18, pad=20)
    plt.xlabel('Month', fontsize=14, labelpad=15)
    plt.ylabel('Year', fontsize=14, labelpad=15)
    plt.xticks(ticks=np.arange(1, 13) + 0.5, labels=[calendar.month_name[i] for i in range(1, 13)], rotation=45)
    plt.tight_layout()
    
    # Save the static plot
    plt.savefig(output_dir / 'improved_year_month_heatmap_no_annotations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create interactive version with Plotly if requested
    if interactive:
        # Create heatmap with improved colors
        fig = px.imshow(
            heatmap_pivot,
            labels=dict(x="Month", y="Year", color="Number of Burglaries"),
            x=[calendar.month_name[i] for i in range(1, 13)],
            y=sorted(heatmap_data['Year'].unique()),
            color_continuous_scale="YlOrBr_r",  # Use reversed scale to make higher values darker
            title="Monthly Burglary Heatmap by Year",
            template="plotly_white",
            text_auto=show_annotations  # Don't show text annotations
        )
        
        # Customize layout
        fig.update_layout(
            xaxis={'side': 'top'},
            coloraxis_colorbar=dict(
                title="Number of<br>Burglaries"
            ),
            height=800,
            width=1300,
            title_font_size=24,
            font=dict(size=14)
        )
        
        # Save the interactive plot
        pio.write_html(fig, file=str(interactive_dir / 'improved_burglary_calendar_heatmap_no_annotations.html'), auto_open=False)
        
        return (output_dir / 'improved_year_month_heatmap_no_annotations.png', 
                interactive_dir / 'improved_burglary_calendar_heatmap_no_annotations.html')
    
    return output_dir / 'improved_year_month_heatmap_no_annotations.png'

def main():
    try:
        print("Loading cleaned burglary data...")
        df = load_cleaned_data()
        
        print("Preprocessing data...")
        df = preprocess_data(df)
        
        print("Creating improved heatmap visualization without annotations...")
        output_files = create_improved_year_month_heatmap(df, output_dir, show_annotations=False)
        
        print("\nImproved heatmap created successfully!")
        print(f"Static image saved to: {output_files[0]}")
        print(f"Interactive version saved to: {output_files[1]}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 