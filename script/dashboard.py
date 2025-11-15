import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("feature_engineered_data.csv")

# ---------------------------
# FIX #1: Handle Date column
# ---------------------------
if "Date" not in df.columns:
    st.warning("⚠ 'Date' column not found — creating a date column automatically.")
    df["Date"] = pd.date_range(start="2020-01-01", periods=len(df), freq="D")
else:
    df["Date"] = pd.to_datetime(df["Date"])

# ---------------------------
# FIX #2: Detect water level column
# ---------------------------
possible_cols = ["Water_Level_m", "water_level", "level", "Level", "Water_Level", "waterlevel"]

water_col = None
for col in possible_cols:
    if col in df.columns:
        water_col = col
        break

if water_col is None:
    st.error("❌ ERROR: No water-level column found.\n\nAvailable columns:")
    st.write(df.columns)
    st.stop()

st.success(f"Using water level column: *{water_col}*")

# ---------------------------
# Dashboard
# ---------------------------
st.title("Reservoir Water Level Dashboard")

# Line chart
st.subheader("Water Level Over Time")
st.line_chart(df[['Date', water_col]].set_index('Date'))

# Monthly averages
df['Month'] = df['Date'].dt.month
monthly = df.groupby('Month')[water_col].mean()

st.subheader("Monthly Average Water Level")
st.bar_chart(monthly)

if st.checkbox("Show raw data"):
    st.write(df)