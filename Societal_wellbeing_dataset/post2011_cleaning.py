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
            lsoa_data = pd.read_csv("lsoa-data-post-2011.csv", encoding=encoding)
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
    
    # Drop duplicates
    print("Removing duplicates...")
    lsoa_data = lsoa_data.drop_duplicates()
    print(f"Dataset shape after removing duplicates: {lsoa_data.shape}")
    
    # Check for null values
    null_counts = lsoa_data.isnull().sum()
    print("Columns with null values:")
    print(null_counts[null_counts > 0])
    
    # Drop rows with all null values (if any)
    lsoa_data = lsoa_data.dropna(how='all')
    print(f"Dataset shape after removing rows with all null values: {lsoa_data.shape}")
    
    # Save the cleaned dataset
    print("Saving cleaned dataset...")
    lsoa_data.to_csv("lsoa-data-post-2011-london-cleaned.csv", index=False)
    print("Cleaned dataset saved as 'lsoa-data-post-2011-london-cleaned.csv'")
except Exception as e:
    print(f"An error occurred: {e}") 