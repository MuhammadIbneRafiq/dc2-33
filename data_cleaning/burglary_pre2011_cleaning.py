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
            lsoa_data = pd.read_csv("lsoa-data-old-boundaries-DataSheet.csv", encoding=encoding, low_memory=False)
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
    lsoa_data = lsoa_data[lsoa_data["Lower Super Output Area"].isin(london_lsoa_codes["LSOA code"])]
    print(f"Dataset shape after filtering for London LSOAs: {lsoa_data.shape}")
    
    # Find all burglary-related columns
    burglary_columns = [col for col in lsoa_data.columns if 'burgl' in col.lower()]
    
    # Add essential identifier columns to the list of columns to keep
    essential_columns = ['Lower Super Output Area', 'Names']
    columns_to_keep = essential_columns + burglary_columns
    
    print(f"Found {len(burglary_columns)} burglary-related columns:")
    for col in burglary_columns:
        print(f"  - {col}")
    
    # Create a new DataFrame with only the burglary-related columns
    burglary_data = lsoa_data[columns_to_keep]
    
    print(f"Dataset shape after selecting burglary columns: {burglary_data.shape}")
    
    # Drop duplicates
    print("Removing duplicates...")
    burglary_data = burglary_data.drop_duplicates()
    print(f"Dataset shape after removing duplicates: {burglary_data.shape}")
    
    # Check for null values
    null_counts = burglary_data.isnull().sum()
    print("Columns with null values:")
    print(null_counts[null_counts > 0])
    
    # Save the cleaned dataset
    print("Saving cleaned dataset...")
    burglary_data.to_csv("lsoa-data-burglary-london-cleaned.csv", index=False)
    print("Cleaned dataset saved as 'lsoa-data-burglary-london-cleaned.csv'")
except Exception as e:
    print(f"An error occurred: {e}") 