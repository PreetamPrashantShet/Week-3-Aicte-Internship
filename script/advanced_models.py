import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------------------------
# Load Dataset
# ------------------------------------------
df = pd.read_csv(r"C:\Users\pc\OneDrive\Attachments\Desktop\aicte-internship\cleaned_reservoir_data.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# ------------------------------------------
# FIX COLUMN NAMES
# ------------------------------------------
df.columns = df.columns.str.strip().str.lower()

# Rename level -> Water_Level_m
if 'level' in df.columns:
    df.rename(columns={'level': 'water_level_m'}, inplace=True)

# ------------------------------------------
# Convert date column
# ------------------------------------------
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
else:
    raise KeyError("Your dataset does NOT contain a 'Date' column!")

# ------------------------------------------
# Feature Engineering
# ------------------------------------------
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
df['month'] = df['date'].dt.month

df['rolling_3'] = df['water_level_m'].rolling(window=3).mean()
df['rolling_7'] = df['water_level_m'].rolling(window=7).mean()

# Fill NA for rolling windows
df.fillna(method='bfill', inplace=True)

# ------------------------------------------
# Select Features
# ------------------------------------------
features = ['day', 'weekday', 'month', 'rolling_3', 'rolling_7']
X = df[features]
y = df['water_level_m']

# ------------------------------------------
# Train-Test Split
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ------------------------------------------
# XGBoost Model
# ------------------------------------------
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6
)

model.fit(X_train, y_train)

# ------------------------------------------
# Predictions
# ------------------------------------------
y_pred = model.predict(X_test)

print("\n------ XGBoost Results ------")
print("MAE =", mean_absolute_error(y_test, y_pred))
print("R²  =", r2_score(y_test, y_pred))

print("\nXGBoost Training + Prediction Completed Successfully!")