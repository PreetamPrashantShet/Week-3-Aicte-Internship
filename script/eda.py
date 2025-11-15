import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_reservoir_data.csv")

# Normalize column names
df.columns = df.columns.str.strip().str.lower()
print("Columns:", df.columns)

# Find date column
possible_date_cols = [col for col in df.columns if 'date' in col]

if len(possible_date_cols) == 0:
    raise ValueError("❌ No date column found in dataset. Check your CSV.")
else:
    date_col = possible_date_cols[0]
    print("Date column detected:", date_col)

# Convert to datetime
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

df = df.dropna(subset=[date_col])

# Rename to standard name
df = df.rename(columns={date_col: 'date'})

# Feature Engineering
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
df['month'] = df['date'].dt.month

# Save engineered file
df.to_csv("feature_engineered_data.csv", index=False)
print("Feature engineering complete!")