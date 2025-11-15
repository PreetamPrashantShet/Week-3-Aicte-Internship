# Week-3-Aicte-Internship
so for the next week 3...Perform detailed Exploratory Data Analysis (EDA) Add feature engineering (hour, weekday, rolling averages) Test advanced models like XGBoost or LSTM Build a small dashboard (Streamlit/Flask) for visualization
# 🌊 Seasonal Reservoir Water Level Prediction  
### AICTE – Edunet Foundation – Shell Green Skills Using AI Internship  
### Domain: Energy | Focus: Water Resource Forecasting

---

## 📌 Project Overview
This project predicts seasonal reservoir water levels using Machine Learning and AI-based forecasting models.  
The primary objective is to support efficient water management and energy planning by forecasting reservoir levels using:

- ✔ XGBoost Regression  
- ✔ NeuralProphet Forecasting (Prophet alternative)  
- ✔ Feature Engineering (Rolling averages, timestamps, seasonality)  
- ✔ Streamlit Dashboard for visualization & deployment  

This project aligns with *Green Skills* concepts by enabling sustainable water resource management using AI.

---

## 📁 Folder Structure
aicte-internship/ │ ├── data/ │   ├── raw_reservoir_data.csv │   ├── cleaned_data.csv │   └── feature_engineered_data.csv │ ├── notebooks/ │   ├── eda.ipynb │   └── forecasting_models.ipynb │ ├── scripts/ │   ├── clean_data.py │   ├── feature_engineering.py │   ├── advanced_models.py │   └── prophet_model.py (optional) │ ├── dashboard.py ├── requirements.txt ├── README.md └── .gitignore
---

## 🔍 Week-Wise Progress Summary

### *Week 1 – Data Collection & Cleaning*
- Collected real reservoir dataset (CWC Dataset)
- Cleaned missing values, fixed column names
- Converted datetime, removed duplicates
- Exported cleaned data to GitHub

### *Week 2 – Prediction & Forecasting*
- Built Linear Regression & XGBoost models  
- Added NeuralProphet forecasting model (since Prophet unsupported on Python 3.13)  
- Compared model accuracy (MAE, R²)
- Saved predictions for dashboard integration

### *Week 3 – EDA & Advanced ML*
- Performed detailed EDA  
- Added feature engineering:
  - Day, Month, Weekday  
  - Rolling averages (3-day, 7-day)  
- Tested XGBoost & NeuralProphet  
- Built initial Streamlit dashboard

### *Week 4 – Deployment*
- Streamlit dashboard integrated  
- Added prediction charts  
- Prepared requirements.txt  
- Deployed on Streamlit Cloud  

---

## 🚀 Deployment Instructions (Streamlit Cloud)

### *1. Push project to GitHub*
### *2. Deploy*
1. Go to: https://share.streamlit.io  
2. Sign in with GitHub  
3. Click *New App*  
4. Select repository and main branch  
5. Choose *dashboard.py* as entry file  
6. Deploy 🎉

---

## ✔ requirements.txt
---

## 📊 Streamlit Dashboard Features
- Water level time-series plot  
- Monthly averages visualization  
- XGBoost forecast for next 30 days  
- NeuralProphet forecast  
- Actual vs Predicted comparison  
- Toggle to view raw data  

---

## 🧠 Technologies Used
- Python  
- Pandas, NumPy  
- Matplotlib  
- Scikit-Learn  
- XGBoost  
- NeuralProphet  
- Streamlit  

---

## 🙌 Acknowledgements
This project was developed as part of the *AICTE–Edunet Foundation Green Skills Using AI Internship, supported by **Shell*.  
Special thanks to mentors and the open-source community for tools and datasets.

---

## 📞 Contact
For any queries or clarifications:  
*Name:* Prashanth Dattatraya Shet  
*GitHub:* (Add your link here)  
*Email:* (Optional)

Make sure these files exist at root:
