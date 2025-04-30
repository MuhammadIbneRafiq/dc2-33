import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Base directory for data storage
base_dir = r'C:\Users\x1 yoga\Documents\q4\dc2\data'

def create_date_range():
    """Create a list of dates from Jan 2024 to Dec 2025"""
    dates = []
    for year in [2024, 2025]:
        for month in range(1, 13):
            dates.append(f"{year}-{month:02d}")
    return dates

def ensure_directory(date):
    """Ensure the directory for a specific date exists"""
    directory = os.path.join(base_dir, date)
    os.makedirs(directory, exist_ok=True)
    return directory

def download_crime_data(date):
    """Download crime data for City of London for a specific month"""
    # City of London force code
    force = "city-of-london"
    
    # Construct the URL
    url = f"https://data.police.uk/api/crimes-street/all-crime?poly=51.5074,-0.1278:51.5174,-0.0778:51.5074,-0.0778:51.4974,-0.1278&date={date}"
    
    try:
        # Create directory if it doesn't exist
        directory = ensure_directory(date)
        
        # Construct file path
        file_path = os.path.join(directory, f"{date}-{force}-street.csv")
        
        # If file already exists, skip download
        if os.path.exists(file_path):
            print(f"File already exists for {date}")
            return True
        
        # Make the request
        response = requests.get(url)
        if response.status_code == 200:
            # Convert to DataFrame
            crimes = pd.DataFrame(response.json())
            
            # Save to CSV
            crimes.to_csv(file_path, index=False)
            print(f"Successfully downloaded data for {date}")
            return True
        else:
            print(f"Failed to download data for {date}: Status code {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error downloading data for {date}: {str(e)}")
        return False

def main():
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Get all dates
    dates = create_date_range()
    
    # Download data for each date
    for date in dates:
        print(f"Processing {date}...")
        download_crime_data(date)
        
if __name__ == "__main__":
    main() 