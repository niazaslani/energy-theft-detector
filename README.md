# ⚡ Energy Theft Detector

A machine learning-powered web application to detect electricity theft based on half-hourly energy usage profiles.

## 🚀 Features

- Upload `.csv` files containing electricity usage data
- Detect suspicious usage patterns using a trained Random Forest model
- Display theft predictions instantly
- Streamlit-powered web interface

## 🧪 How It Works

1. A Random Forest model is trained on a labeled dataset with simulated theft.
2. The app takes new data with features like:
   - Mean energy usage
   - Standard deviation
   - Minimum and maximum usage
3. The model predicts whether usage is **normal** or **suspicious**.

## 📊 Demo

🌐 Live App: [energy-theft-detector.streamlit.app](https://energy-theft-detector.streamlit.app/)

Upload a `.csv` file like:

```csv
LCLid,KWH/hh (per half hour)
Test001,0.45
Test002,0.30
Test003,0.60