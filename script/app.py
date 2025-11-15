# -----------------------------------------------------
# Internship Project: Seasonal Reservoir Water Level Prediction
# Week 1 - Dataset Cleaning
# Author: Preetam Prashant Shet
# -----------------------------------------------------

import pandas as pd
import numpy as np

# Step 1: Load dataset
file_path = r"C:\Users\pc\OneDrive\Attachments\Desktop\.vscode\Daily_data_of_reservoir_level_of_Central_Water_Commission_(CWC)_Agency_during_March_2024.csv"
df = pd.read_csv(file_path)

print("✅ Data Loaded Successfully!")
print(df.head())

# -----------------------------------------------------
# Step 2: Inspect and Clean Column Names
# -----------------------------------------------------
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print("\n🧹 Cleaned Columns:", df.columns.tolist())

# -----------------------------------------------------
# Step 3: Drop Unnecessary Columns (if any)
# -----------------------------------------------------
# Drop irrelevant columns if present (you can adjust based on dataset)
for col in ['entity', 'code', 'sr_no', 'station_code']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        print(f"🗑️ Dropped column: {col}")

# -----------------------------------------------------
# Step 4: Handle Missing Values
# -----------------------------------------------------
print("\n📊 Missing values before cleaning:\n", df.isna().sum())

# Fill missing numeric values using forward fill and 0
df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)

print("\n✅ Missing values after cleaning:\n", df.isna().sum())

# -----------------------------------------------------
# Step 5: Ensure Numeric Columns Have Correct Types
# -----------------------------------------------------
for col in df.columns:
    if col not in ['date', 'reservoir_name', 'state', 'river', 'basin']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# -----------------------------------------------------
# Step 6: Remove Duplicates
# -----------------------------------------------------
before = df.shape[0]
df.drop_duplicates(inplace=True)
after = df.shape[0]
print(f"\n🧾 Removed {before - after} duplicate rows.")

# -----------------------------------------------------
# Step 7: Rename Columns for Clarity
# -----------------------------------------------------
# Adjust column names to make them clear and readable
df.rename(columns={
    'level_(m)': 'water_level_m',
    'storage_(mcm)': 'storage_mcm',
    'full_reservoir_level_(m)': 'full_level_m',
    'date_of_data': 'date'
}, inplace=True, errors='ignore')

print("\n🏷️ Renamed Columns:", df.columns.tolist())

# -----------------------------------------------------
# Step 8: Convert Dates to Datetime Type
# -----------------------------------------------------
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    print("\n📅 Date column converted to datetime format.")

# -----------------------------------------------------
# Step 9: Filter Outliers (Unrealistic Values)
# -----------------------------------------------------
if 'water_level_m' in df.columns:
    df = df[(df['water_level_m'] > 0) & (df['water_level_m'] < 1000)]
if 'storage_mcm' in df.columns:
    df = df[(df['storage_mcm'] >= 0) & (df['storage_mcm'] < 100000)]

print("\n🚿 Outlier filtering completed.")

# -----------------------------------------------------
# Step 10: Save Cleaned Dataset
# -----------------------------------------------------
output_path = r"C:\Users\pc\OneDrive\Attachments\Desktop\.vscode\cleaned_reservoir_data.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Data cleaning complete! Cleaned file saved at:\n{output_path}")

# -----------------------------------------------------
# Step 11: Quick Summary
# -----------------------------------------------------
print("\n📈 Final Dataset Info:")
print(df.info())
print("\n📊 Statistical Summary:")
print(df.describe())

# Suppose your cleaned DataFrame is named df

# Export to CSV (without index column)
df.to_csv(r"C:\Users\pc\OneDrive\Attachments\Desktop\.vscode\cleaned_reservoir_data.csv", index=False)

print("✅ Cleaned data exported successfully!")
