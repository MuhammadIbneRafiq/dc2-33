# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 16:14:06 2025

@author: 20234754
"""

import pandas as pd

# === FILE PATHS ===
risk_predictions_path = r"monthly_risk_predictions_2025.csv"
lsoa_to_ward_path = r"LSOA11_WD21_LAD21_EW_LU_V2.xlsx"
population_path = r"sapelsoasyoa20192022.xlsx"

# === LOAD DATA ===
risk_df = pd.read_csv(risk_predictions_path)
lsoa_to_ward_df = pd.read_excel(lsoa_to_ward_path)
population_df = pd.read_excel(
    population_path,
    sheet_name='Mid-2021 LSOA 2021',
    skiprows=6,
    usecols=[2, 4],
    names=["LSOA11CD", "population"]
)

# === CLEANING ===
risk_df.rename(columns={"LSOA_code": "LSOA11CD"}, inplace=True)
lsoa_to_ward_df = lsoa_to_ward_df[['LSOA11CD', 'WD21CD', 'WD21NM']]

# === MERGE ===
merged_df = risk_df.merge(lsoa_to_ward_df, on="LSOA11CD", how="left")
merged_df = merged_df.merge(population_df, on="LSOA11CD", how="left")
merged_df.dropna(subset=["population"], inplace=True)

# === CALCULATE WEIGHTED RISK PER LSOA ===
merged_df["weighted_risk"] = merged_df["average_risk_score_2025"] * merged_df["population"]

# === AGGREGATE TO WARD LEVEL ===
ward_risk_df = merged_df.groupby(['WD21CD', 'WD21NM']).agg({
    "weighted_risk": "sum",
    "population": "sum"
}).reset_index()

# === CALCULATE POPULATION-WEIGHTED RISK SCORE ===
ward_risk_df["population_weighted_risk"] = ward_risk_df["weighted_risk"] / ward_risk_df["population"]

# === Normalize risk scores so they sum to 1
total_risk = ward_risk_df["population_weighted_risk"].sum()
ward_risk_df["normalized_risk"] = ward_risk_df["population_weighted_risk"] / total_risk

# === Total hours available across all wards (200 per ward)
TOTAL_HOURS_AVAILABLE = len(ward_risk_df) * 200

# === Allocate total available hours proportionally to risk
ward_risk_df["allocated_hours"] = ward_risk_df["normalized_risk"] * TOTAL_HOURS_AVAILABLE

# === Convert allocated hours to number of officers (2h per officer)
ward_risk_df["officers_needed"] = (ward_risk_df["allocated_hours"] / 2).round()

# === Rename if needed
ward_risk_df.rename(columns={
    "WD21CD": "ward_code",
    "WD21NM": "ward_name"
}, inplace=True)

# === Optional: display output
print(ward_risk_df[["ward_code", "ward_name", "population_weighted_risk", "allocated_hours", "officers_needed"]].head())

ward_risk_df.to_csv("ward_allocation_plan.csv", index=False)
