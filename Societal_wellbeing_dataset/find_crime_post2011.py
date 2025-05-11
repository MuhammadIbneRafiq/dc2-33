#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

# Load dataset
print("Loading dataset...")
try:
    df = pd.read_csv("lsoa-data-post-2011.csv", encoding='latin1', low_memory=False)
    print(f"Successfully loaded with shape: {df.shape}")
    
    # Define a broader list of crime-related terms
    crime_terms = [
        'crime', 'burgl', 'theft', 'offen', 'robbery', 'assault', 'violen', 
        'damage', 'drug', 'weapon', 'polic', 'murder', 'criminal', 'anti-social',
        'antisocial', 'casualty', 'accident', 'secur', 'safety', 'safe'
    ]
    
    # Find columns that contain any of the terms
    crime_columns = []
    for col in df.columns:
        if any(term in col.lower() for term in crime_terms):
            crime_columns.append(col)
    
    # Print results
    if crime_columns:
        print(f"\nFound {len(crime_columns)} crime-related columns:")
        for col in crime_columns:
            print(f"  - {col}")
    else:
        print("\nNo crime-related columns found with the search terms used.")
        
except Exception as e:
    print(f"Error: {e}") 