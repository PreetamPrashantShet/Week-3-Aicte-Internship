# ============================================
# WEEK 2: Prediction & Forecasting System
# Author: Your Name
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet

# ------------------------------------------------
# STEP 1: Load Dataset
# ------------------------------------------------
file_path = r"C:\Users\pc\OneDrive\Attachments\Desktop\aicte-internship\cleaned_reservoir_data.csv"
df = pd.read_csv(file_path)
print("✅ Data Loaded Successfully!")
print("Columns Available:", df.columns.tolist())

# ------------------------------------------------
# STEP 2: Detect Date & Target Columns Automatically
# ------------------------------------------------
date_col = None
target_col = None

for c in df.columns:
    if 'date' in c.lower():
        date_col = c
    if 'level' in c.lower() or 'water' in c.lower():
        target_col = c

if date_col is None:
    raise ValueError("❌ No date column found. Please rename your date column to include 'Date'.")
if target_col is None:
    raise ValueError("❌ No target column found. Please rename your water level column to include 'Level' or 'Water'.")

# Rename columns for consistency
df.rename(columns={date_col: 'Date', target_col: 'Reservoir_Level'}, inplace=True)

# ------------------------------------------------
# STEP 3: Convert and Sort Dates
# ------------------------------------------------
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)
df = df.sort_values('Date')
print("\n📅 Date column processed successfully!")

# ------------------------------------------------
# STEP 4: Feature Engineering
# ------------------------------------------------
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

X = df[['Year', 'Month', 'Day']]
y = df['Reservoir_Level']

# ------------------------------------------------
# STEP 5: Split + Scale
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------
# STEP 6: Ridge Regression (auto-tuning)
# ------------------------------------------------
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 50, 100]}
ridge = Ridge()
ridge_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_absolute_error')
ridge_search.fit(X_train_scaled, y_train)
best_ridge = ridge_search.best_estimator_

y_pred_ridge = best_ridge.predict(X_test_scaled)

mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\n📈 Ridge Regression Results:")
print("Best alpha:", ridge_search.best_params_)
print(f"MAE={mae_ridge:.4f}, RMSE={rmse_ridge:.4f}, R²={r2_ridge:.4f}")

# ------------------------------------------------
# STEP 7: Prophet Forecasting
# ------------------------------------------------
prophet_df = df[['Date', 'Reservoir_Level']].rename(columns={'Date': 'ds', 'Reservoir_Level': 'y'})

best_score = float('inf')
best_model = None
best_params = {}

for cps in [0.1, 0.3, 0.5]:
    for seasonality_mode in ['additive', 'multiplicative']:
        model = Prophet(
            changepoint_prior_scale=cps,
            seasonality_mode=seasonality_mode,
            yearly_seasonality=True,
            daily_seasonality=False
        )
        model.fit(prophet_df)
        forecast = model.predict(prophet_df)
        mae = mean_absolute_error(prophet_df['y'], forecast['yhat'])
        if mae < best_score:
            best_score = mae
            best_params = {'changepoint_prior_scale': cps, 'seasonality_mode': seasonality_mode}
            best_model = model

print("\n🔮 Best Prophet Parameters:", best_params)
print(f"Prophet MAE={best_score:.4f}")

# Future prediction (30 days ahead)
future = best_model.make_future_dataframe(periods=30)
forecast = best_model.predict(future)

# ------------------------------------------------
# STEP 8: Compare Models
# ------------------------------------------------
comparison = pd.DataFrame({
    'Model': ['Ridge Regression', 'Prophet'],
    'MAE': [mae_ridge, best_score],
    'RMSE': [rmse_ridge, np.sqrt(mean_squared_error(prophet_df['y'], best_model.predict(prophet_df)['yhat']))],
    'R2': [r2_ridge, r2_score(prophet_df['y'], best_model.predict(prophet_df)['yhat'])]
})

print("\n📊 Model Comparison:\n", comparison)

# ------------------------------------------------
# STEP 9: Save Outputs
# ------------------------------------------------
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("forecast_results.csv", index=False)
comparison.to_csv("model_comparison.csv", index=False)
print("\n💾 Results saved successfully!")

# ------------------------------------------------
# STEP 10: Visualization
# ------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Reservoir_Level'], label='Actual', color='blue')
plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red')
plt.title("Reservoir Water Level Forecasting (Linear + Prophet)")
plt.xlabel("Date")
plt.ylabel("Reservoir Level")
plt.legend()
plt.show()
