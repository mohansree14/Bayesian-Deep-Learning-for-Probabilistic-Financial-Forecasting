# Streamlit Cloud Deployment Guide

## 🚀 Quick Deploy

### Streamlit Cloud Configuration

**App Entry Point:** `app/streamlit_app.py`

### Required Files for Deployment

1. ✅ `requirements.txt` - Python dependencies
2. ✅ `packages.txt` - System dependencies 
3. ✅ `.streamlit/config.toml` - Streamlit configuration
4. ✅ `src/models/__init__.py` - Package initialization

### Deployment Steps

1. **Fork/Clone** this repository to your GitHub account

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Connect your GitHub** account and select this repository

4. **Configure the app:**
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
   - **Python version:** 3.11

5. **Deploy!** The app will automatically install dependencies and start

### 🔧 Troubleshooting

#### Import Errors
If you see `ModuleNotFoundError` for `src` modules:

1. **Verify file structure:**
   ```
   project-root/
   ├── src/
   │   ├── __init__.py
   │   └── models/
   │       ├── __init__.py
   │       ├── lstm.py
   │       └── mc_dropout_lstm.py
   ├── app/
   │   └── streamlit_app.py
   ├── requirements.txt
   └── data/
   ```

2. **Check that all `__init__.py` files exist**

3. **Ensure Python path setup is working** - the app includes debug information

#### Missing Data
The app includes synthetic demo data generation, so it will work even without pre-trained models.

### 📊 Demo Mode
When deployed without trained models, the app automatically runs in demo mode with synthetic predictions, allowing you to explore all features.

### 🎯 Features Available on Cloud
- ✅ 8 Stock tickers (AAPL, MSFT, GOOGL, TSLA, NVDA, AMZN, META, NFLX)
- ✅ Interactive Plotly visualizations
- ✅ Uncertainty quantification
- ✅ Multiple chart types
- ✅ Time period filtering
- ✅ Performance metrics

### 📝 Sample Data
The app can generate sample financial data using yfinance, or use demo synthetic data for immediate deployment.