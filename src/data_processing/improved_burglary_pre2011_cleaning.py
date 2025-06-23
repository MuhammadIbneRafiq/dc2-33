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
    
    # Filter to keep only London LSOAs
    print("Filtering to keep only London LSOAs...")
    london_lsoa_list = london_lsoa_codes['LSOA code'].tolist()
    lsoa_data_london = lsoa_data[lsoa_data['Lower Super Output Area'].isin(london_lsoa_list)]
    print(f"Dataset shape after filtering for London LSOAs: {lsoa_data_london.shape}")
    
    # Find burglary-related columns
    burglary_columns = [col for col in lsoa_data_london.columns if 'burgl' in col.lower()]
    
    # Always keep the identifier columns
    selected_columns = ['Lower Super Output Area', 'Names'] + burglary_columns
    
    print(f"Found {len(burglary_columns)} burglary-related columns:")
    for col in burglary_columns:
        print(f"  - {col}")
    
    # Create dataset with only burglary columns
    burglary_data = lsoa_data_london[selected_columns]
    print(f"Dataset shape after selecting burglary columns: {burglary_data.shape}")
    
    # Remove duplicates
    print("Removing duplicates...")
    burglary_data = burglary_data.drop_duplicates()
    print(f"Dataset shape after removing duplicates: {burglary_data.shape}")
    
    # Check for null values
    null_counts = burglary_data.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    print("Columns with null values:")
    print(columns_with_nulls)
    
    # Simple statistics for burglary columns
    print("\nBasic statistics for burglary columns:")
    for col in burglary_columns:
        # Convert to numeric, coercing errors to NaN
        burglary_data[col] = pd.to_numeric(burglary_data[col], errors='coerce')
        
        # Calculate statistics for non-null values
        non_null_values = burglary_data[col].dropna()
        if len(non_null_values) > 0:
            print(f"\nStats for {col}:")
            print(f"  Count: {len(non_null_values)}")
            print(f"  Mean: {non_null_values.mean():.2f}")
            print(f"  Min: {non_null_values.min():.2f}")
            print(f"  Max: {non_null_values.max():.2f}")
            print(f"  Std Dev: {non_null_values.std():.2f}")
        else:
            print(f"\nNo valid numeric data for {col}")
    
    # Save the cleaned dataset
    print("\nSaving cleaned dataset...")
    burglary_data.to_csv("lsoa-data-burglary-london-cleaned.csv", index=False)
    print("Cleaned dataset saved as 'lsoa-data-burglary-london-cleaned.csv'")
    
    # List the top 10 LSOAs with highest burglary numbers (for the most recent year available)
    if 'Crime (numbers);Burglary;2012/13' in burglary_data.columns:
        print("\nTop 10 LSOAs with highest burglary numbers in 2012/13:")
        top_burglary = burglary_data.sort_values(by='Crime (numbers);Burglary;2012/13', ascending=False).head(10)
        for _, row in top_burglary.iterrows():
            print(f"  {row['Lower Super Output Area']} - {row['Names']}: {row['Crime (numbers);Burglary;2012/13']}")
    
except Exception as e:
    print(f"Error: {e}") 