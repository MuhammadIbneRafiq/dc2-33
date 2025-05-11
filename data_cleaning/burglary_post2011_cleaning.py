#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

# Load datasets
print("Loading datasets...")
try:
    # Try different encodings
    encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
    for encoding in encodings:
        try:
            print(f"Trying encoding: {encoding}")
            lsoa_data = pd.read_csv("lsoa-data-post-2011.csv", encoding=encoding, low_memory=False)
            print(f"Successfully loaded with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            print(f"Failed with encoding: {encoding}")
            continue
    
    london_lsoa_codes = pd.read_csv("LSOA_codes.csv")
    
    # Display initial dataset information
    print(f"Original dataset shape: {lsoa_data.shape}")
    
    # Keep only London LSOAs by filtering the dataset
    print("Filtering to keep only London LSOAs...")
    lsoa_data = lsoa_data[lsoa_data.iloc[:, 0].isin(london_lsoa_codes["LSOA code"])]
    print(f"Dataset shape after filtering for London LSOAs: {lsoa_data.shape}")
    
    # Find all crime-related columns (searching for 'crime', 'offen', 'burgla', etc.)
    crime_columns = [col for col in lsoa_data.columns if 
                    any(term in col.lower() for term in ['crime', 'offen', 'burgla', 'theft', 'criminal'])]
    
    # Add essential identifier columns to the list of columns to keep
    essential_columns = ['Lower Super Output Area', 'Names']
    columns_to_keep = essential_columns + crime_columns
    
    print(f"Found {len(crime_columns)} crime-related columns:")
    for col in crime_columns:
        print(f"  - {col}")
    
    # Create a new DataFrame with only the crime-related columns
    crime_data = lsoa_data[columns_to_keep]
    
    print(f"Dataset shape after selecting crime columns: {crime_data.shape}")
    
    # Drop duplicates
    print("Removing duplicates...")
    crime_data = crime_data.drop_duplicates()
    print(f"Dataset shape after removing duplicates: {crime_data.shape}")
    
    # Check for null values
    null_counts = crime_data.isnull().sum()
    print("Columns with null values:")
    print(null_counts[null_counts > 0])
    
    # Save the cleaned dataset
    print("Saving cleaned dataset...")
    crime_data.to_csv("lsoa-data-crime-post-2011-london-cleaned.csv", index=False)
    print("Cleaned dataset saved as 'lsoa-data-crime-post-2011-london-cleaned.csv'")
except Exception as e:
    print(f"An error occurred: {e}") 