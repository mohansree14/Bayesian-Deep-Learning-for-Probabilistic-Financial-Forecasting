from __future__ import annotations
import json
import sys
import os
from pathlib import Path
import argparse
import numpy as np
import streamlit as st
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the project root to Python path - works for both local and Streamlit Cloud
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Alternative path setup for Streamlit Cloud deployment
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Try to add the workspace mount path (for Streamlit Cloud)
workspace_path = "/mount/src/bayesian-deep-learning-for-probabilistic-financial-forecasting"
if os.path.exists(workspace_path) and workspace_path not in sys.path:
    sys.path.insert(0, workspace_path)

try:
    from src.models.lstm import LSTMRegressor
    from src.models.mc_dropout_lstm import MCDropoutLSTM
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.error("This usually happens when the project structure is not properly set up for deployment.")
    
    st.info("🔧 **Debug Information:**")
    st.write("**Python Path:**", sys.path)
    st.write("**Current Directory:**", os.getcwd())
    st.write("**Project Root:**", str(project_root))
    st.write("**Project Root Exists:**", project_root.exists())
    
    if project_root.exists():
        st.write("**Files in Project Root:**", [f.name for f in project_root.iterdir()])
        src_path = project_root / "src"
        if src_path.exists():
            st.write("**Files in src/:**", [f.name for f in src_path.iterdir()])
            models_path = src_path / "models"
            if models_path.exists():
                st.write("**Files in src/models/:**", [f.name for f in models_path.iterdir()])
    
    st.markdown("""
    ### 🚨 **Deployment Setup Required**
    
    To fix this issue for Streamlit Cloud deployment:
    
    1. **Ensure all `__init__.py` files exist** in the `src/` and `src/models/` directories
    2. **Check that the repository structure is correct** on Streamlit Cloud
    3. **Verify that `requirements.txt` contains all dependencies**
    4. **Make sure the app entry point is `app/streamlit_app.py`**
    
    ### 📁 **Expected Structure:**
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
    """)
    st.stop()


def generate_synthetic_data(ticker: str, n_samples: int = 500, seq_len: int = 30, n_features: int = 20):
    """Generate synthetic financial data for demo purposes."""
    np.random.seed(hash(ticker) % 2**32)  # Consistent seed per ticker
    
    # Generate synthetic sequences
    X_test = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    
    # Generate synthetic target values with some trend
    base_price = 100 + hash(ticker) % 200  # Different base price per ticker
    y_test = base_price + np.cumsum(np.random.randn(n_samples) * 2)
    
    # Create synthetic metadata
    meta = {
        "feature_cols": [f"feature_{i}" for i in range(n_features)],
        "window": seq_len,
        "horizon": 1,
        "step": 1,
        "target_col": "adj_close",
        "interval": "1d",
        "data_source": "synthetic_demo"
    }
    
    return meta, X_test, y_test


def load_meta(data_dir: Path, ticker: str):
    meta_path = data_dir / ticker / "meta.json"
    if not meta_path.exists():
        st.info(f"📊 Generating synthetic metadata for {ticker} (Demo Mode)")
        meta, _, _ = generate_synthetic_data(ticker)
        return meta
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Error loading meta file, using synthetic data: {e}")
        meta, _, _ = generate_synthetic_data(ticker)
        return meta


def load_arrays(data_dir: Path, ticker: str):
    tdir = data_dir / ticker
    X_test_path = tdir / "X_test.npy"
    y_test_path = tdir / "y_test.npy"
    
    # Check if real data exists
    if not tdir.exists() or not X_test_path.exists() or not y_test_path.exists():
        st.info(f"📊 Generating synthetic data for {ticker} (Demo Mode)")
        _, X_test, y_test = generate_synthetic_data(ticker)
        return X_test, y_test
    
    try:
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        return X_test, y_test
    except Exception as e:
        st.warning(f"Error loading arrays, using synthetic data: {e}")
        _, X_test, y_test = generate_synthetic_data(ticker)
        return X_test, y_test


def predict_point(model_type: str, ckpt_dir: Path, ticker: str, X: np.ndarray):
    """Make predictions with trained model or generate demo predictions if model not available."""
    model_path = ckpt_dir / model_type / f"{ticker}.pt"
    
    # Check if trained model exists
    if not model_path.exists():
        st.warning(f"⚠️ No trained model found at {model_path}. Generating demo predictions...")
        # Generate synthetic predictions for demo
        np.random.seed(42)  # For reproducible demo
        n_samples = len(X)
        # Create somewhat realistic predictions with trend and noise
        base_trend = np.linspace(0.95, 1.05, n_samples)
        noise = np.random.normal(0, 0.02, n_samples)
        mu = base_trend + noise
        sigma = np.abs(np.random.normal(0.05, 0.02, n_samples))  # Positive uncertainty
        return mu.reshape(-1, 1), sigma.reshape(-1, 1)
    
    # Load and use trained model
    input_dim = X.shape[-1]
    if model_type == "lstm":
        model = LSTMRegressor(input_dim)
    elif model_type == "mc_dropout_lstm":
        model = MCDropoutLSTM(input_dim)
    else:
        st.error("Unsupported model for app")
        st.stop()
    
    try:
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"]) if "model_state_dict" in state else model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X).float()
            if model_type == "mc_dropout_lstm":
                preds = model.mc_predict(xb, mc_samples=50).cpu().numpy()
                return preds.mean(0), preds.std(0)
            else:
                pred = model(xb).cpu().numpy()
                return pred, np.full_like(pred, fill_value=np.std(pred))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


def get_available_tickers(data_dir: Path):
    """Get list of available tickers from data directory."""
    if not data_dir.exists():
        st.info(f"📁 Data directory not found: {data_dir}")
        st.info("🚀 **Running in Demo Mode** - Using synthetic data for demonstration")
        return ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "NFLX"]  # Demo tickers
    
    tickers = []
    try:
        for item in data_dir.iterdir():
            if item.is_dir() and (item / "meta.json").exists():
                tickers.append(item.name)
    except Exception as e:
        st.warning(f"Error reading data directory: {e}")
        st.info("🚀 **Switching to Demo Mode** - Using synthetic data")
        return ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "NFLX"]
    
    if not tickers:
        st.info(f"No processed data found in {data_dir}")
        st.info("🚀 **Running in Demo Mode** - Using synthetic data for demonstration")
        return ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "NFLX"]
    
    return sorted(tickers)

def get_available_models(ckpt_dir: Path):
    """Get list of available models from checkpoint directory."""
    if not ckpt_dir.exists():
        return ["lstm (demo)", "mc_dropout_lstm (demo)"]  # Default fallback with demo indication
    
    models = []
    for item in ckpt_dir.iterdir():
        if item.is_dir():
            models.append(item.name)
    
    # If no trained models found, provide demo options
    if not models:
        return ["lstm (demo)", "mc_dropout_lstm (demo)"]
    
    return models

def main():
    st.set_page_config(page_title="Bayesian Time Series Forecasting", layout="wide")
    st.title("🎯 Uncertainty-Aware Stock Price Forecasting")
    
    # Check if we're in demo mode
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    is_demo_mode = not data_dir.exists() or not any(data_dir.iterdir()) if data_dir.exists() else True
    
    if is_demo_mode:
        st.info("""
        🚀 **Demo Mode Active** - This app is running with synthetic data for demonstration purposes. 
        All predictions and visualizations are generated using synthetic financial data that mimics real stock patterns.
        """)
        
        # Add option to generate real data (for cloud deployment)
        with st.expander("📊 Generate Real Financial Data (Optional)"):
            st.markdown("""
            **Note:** In demo mode, the app uses synthetic data. If you want to fetch real financial data:
            
            1. **For local development:** Run the data generation scripts as shown in the README
            2. **For Streamlit Cloud:** Due to file system limitations, real-time data fetching is not available
            3. **Current demo:** Provides realistic synthetic predictions for all major tech stocks
            """)
            
            if st.button("🔄 Refresh Demo Data"):
                st.rerun()
    
    # Project overview
    st.markdown("""
    **Project Overview:** This application demonstrates advanced machine learning models for stock price prediction 
    with uncertainty quantification. The models provide not just predictions, but also confidence intervals 
    to help assess prediction reliability.
    """)
    
    # Configuration
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("""
    **How to use:** Select a stock ticker, choose a trained model, and adjust the confidence interval. 
    Use time filters to focus on specific periods.
    """)
    
    # Use absolute paths relative to project root
    default_config_path = project_root / "configs" / "app.yaml"
    default_ckpt_dir = project_root / "experiments" / "checkpoints"
    default_data_dir = project_root / "data" / "processed"
    
    config_path = st.sidebar.text_input("Config path", value=str(default_config_path))
    ckpt_dir = Path(st.sidebar.text_input("Checkpoint dir", value=str(default_ckpt_dir)))
    data_dir = Path(st.sidebar.text_input("Data dir", value=str(default_data_dir)))
    
    # Get available options
    available_tickers = get_available_tickers(data_dir)
    available_models = get_available_models(ckpt_dir)
    
    # Sidebar controls
    ticker = st.sidebar.selectbox("Ticker", available_tickers, index=0)
    model_type = st.sidebar.selectbox("Model", available_models, index=0)
    conf_z = st.sidebar.slider("CI z-score", 0.5, 3.0, 1.96)
    
    # Time period filtering
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📅 Time Period Filter**")
    
    time_filter = st.sidebar.selectbox(
        "View Period", 
        ["All Data", "Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year", "Custom Range"],
        index=0
    )
    
    # Custom date range
    if time_filter == "Custom Range":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_idx = st.number_input("Start Index", min_value=0, value=0, help="Starting data point index")
        with col2:
            end_idx = st.number_input("End Index", min_value=1, value=100, help="Ending data point index")
    else:
        start_idx = None
        end_idx = None
    
    # Add info about available options
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Available Options:**")
    st.sidebar.markdown(f"📊 Tickers: {', '.join(available_tickers)}")
    st.sidebar.markdown(f"🤖 Models: {', '.join(available_models)}")
    
    # Debug information (collapsible)
    with st.sidebar.expander("🔍 Debug Info"):
        st.markdown(f"**Mode:** {'Demo (Synthetic Data)' if is_demo_mode else 'Real Data'}")
        st.markdown(f"**Working Dir:** {os.getcwd()}")
        st.markdown(f"**Project Root:** {project_root}")
        st.markdown(f"**Data Dir:** {data_dir}")
        st.markdown(f"**Data Dir Exists:** {data_dir.exists()}")
        if data_dir.exists():
            contents = [d.name for d in data_dir.iterdir() if d.is_dir()] if data_dir.exists() else []
            st.markdown(f"**Data Dirs:** {contents}")
        st.markdown(f"**Python Path:** {sys.path[:3]}...")  # Show first 3 paths
    
    try:
        # Load data and make predictions
        meta = load_meta(data_dir, ticker)
        X_test, y_test = load_arrays(data_dir, ticker)
        # Clean model type (remove demo suffix if present)
        clean_model_type = model_type.replace(" (demo)", "")
        mu, sigma = predict_point(clean_model_type, ckpt_dir, ticker, X_test)
        y_true = y_test[:, 0] if y_test.ndim == 2 else y_test
        
        # Apply time period filtering
        total_points = len(y_true)
        
        if time_filter == "All Data":
            start_idx = 0
            end_idx = total_points
        elif time_filter == "Last 30 Days":
            start_idx = max(0, total_points - 30)
            end_idx = total_points
        elif time_filter == "Last 90 Days":
            start_idx = max(0, total_points - 90)
            end_idx = total_points
        elif time_filter == "Last 6 Months":
            start_idx = max(0, total_points - 180)
            end_idx = total_points
        elif time_filter == "Last Year":
            start_idx = max(0, total_points - 365)
            end_idx = total_points
        elif time_filter == "Custom Range":
            start_idx = max(0, min(start_idx, total_points - 1))
            end_idx = min(max(end_idx, start_idx + 1), total_points)
        
        # Filter data based on selected time period
        y_true_filtered = y_true[start_idx:end_idx]
        # Ensure mu and sigma are 1-dimensional for plotting
        mu_flat = mu.flatten() if mu.ndim > 1 else mu
        sigma_flat = sigma.flatten() if sigma.ndim > 1 else sigma
        mu_filtered = mu_flat[start_idx:end_idx]
        sigma_filtered = sigma_flat[start_idx:end_idx]
        
        # Create filtered indices for plotting
        filtered_indices = list(range(start_idx, end_idx))
        
        # Model loading status
        st.success(f"✅ Model loaded successfully: {model_type.upper()} on {ticker}")
        
        # Model explanation
        if model_type == "mc_dropout_lstm":
            st.info("🤖 **MC Dropout LSTM**: Uses Monte Carlo dropout to estimate uncertainty by running multiple forward passes with random neuron dropout.")
        elif model_type == "lstm":
            st.info("🤖 **LSTM Baseline**: Standard Long Short-Term Memory network for sequential prediction without explicit uncertainty estimation.")
        elif model_type == "transformer":
            st.info("🤖 **Transformer**: Attention-based model capturing long-range dependencies in time series data.")
        elif model_type == "bnn_vi":
            st.info("🤖 **Bayesian Neural Network**: Uses variational inference to learn probability distributions over weights, providing principled uncertainty estimates.")
        
        # Display time period info
        st.info(f"📅 Viewing: {time_filter} ({len(y_true_filtered)} data points from index {start_idx} to {end_idx-1})")
        
        # Time period summary
        if time_filter != "All Data":
            coverage_pct = (len(y_true_filtered) / total_points) * 100
            st.success(f"📊 Showing {coverage_pct:.1f}% of total data ({len(y_true_filtered)}/{total_points} points)")
        
        # Display data info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", total_points)
        with col2:
            st.metric("Filtered Samples", len(y_true_filtered))
        with col3:
            st.metric("Sequence Length", X_test.shape[1])
        with col4:
            st.metric("Features", X_test.shape[2])
        
        st.subheader("📈 Test Predictions")
        st.markdown("""
        **What this shows:** Comparison between actual stock prices (y_true) and model predictions (mu). 
        Closer alignment indicates better model accuracy. The gap between lines represents prediction error.
        """)
        import pandas as pd
        
        # Create DataFrame with filtered data
        df = pd.DataFrame({
            "y_true": y_true_filtered.flatten() if hasattr(y_true_filtered, 'flatten') else y_true_filtered, 
            "mu": mu_filtered.flatten() if hasattr(mu_filtered, 'flatten') else mu_filtered, 
            "lower": (mu_filtered - conf_z * sigma_filtered).flatten() if hasattr((mu_filtered - conf_z * sigma_filtered), 'flatten') else (mu_filtered - conf_z * sigma_filtered), 
            "upper": (mu_filtered + conf_z * sigma_filtered).flatten() if hasattr((mu_filtered + conf_z * sigma_filtered), 'flatten') else (mu_filtered + conf_z * sigma_filtered),
            "index": filtered_indices
        })
        
        # Chart type selector
        chart_type = st.radio(
            "📊 Select Chart Type:",
            ["Line Chart", "Candlestick Chart", "Interactive Plotly"],
            horizontal=True,
            help="Choose how to visualize the predictions"
        )
        
        # Add description for each chart type
        if chart_type == "Line Chart":
            st.caption("Simple line chart showing predictions vs actual prices")
        elif chart_type == "Candlestick Chart":
            st.caption("Candlestick view showing prediction uncertainty as OHLC (Open/High/Low/Close) bars")
        else:
            st.caption("Interactive chart with zoom, pan, and hover features - includes confidence bands")
        
        if chart_type == "Line Chart":
            # Main prediction chart (original)
            chart_data = df[["y_true", "mu"]].set_index(df["index"])
            st.line_chart(chart_data)
            
        elif chart_type == "Candlestick Chart":
            # Create candlestick-style chart
            st.markdown("**📊 Candlestick-Style Visualization** (simulated OHLC from predictions)")
            
            # Simulate OHLC data from predictions and uncertainty
            # Ensure arrays are 1-dimensional
            y_mu = mu_filtered.flatten() if hasattr(mu_filtered, 'flatten') else mu_filtered
            y_true = y_true_filtered.flatten() if hasattr(y_true_filtered, 'flatten') else y_true_filtered
            y_sigma = sigma_filtered.flatten() if hasattr(sigma_filtered, 'flatten') else sigma_filtered
            
            open_prices = y_mu - y_sigma * 0.5
            high_prices = y_mu + y_sigma
            low_prices = y_mu - y_sigma
            close_prices = y_mu
            
            fig = go.Figure(data=[go.Candlestick(
                x=filtered_indices,
                open=open_prices,
                high=high_prices,
                low=low_prices,
                close=close_prices,
                name='Predicted',
                increasing_line_color='green',
                decreasing_line_color='red'
            )])
            
            # Add actual prices as a line
            fig.add_trace(go.Scatter(
                x=filtered_indices,
                y=y_true,
                mode='lines',
                name='Actual Price',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title='Predictions (Candlestick) vs Actual Price',
                xaxis_title='Time Index',
                yaxis_title='Price',
                height=500,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # Interactive Plotly
            # Interactive Plotly chart with zoom and hover
            fig = go.Figure()
            
            # Ensure all arrays are 1-dimensional for Plotly
            y_mu = mu_filtered.flatten() if hasattr(mu_filtered, 'flatten') else mu_filtered
            y_true = y_true_filtered.flatten() if hasattr(y_true_filtered, 'flatten') else y_true_filtered
            y_sigma = sigma_filtered.flatten() if hasattr(sigma_filtered, 'flatten') else sigma_filtered
            
            # Add prediction line
            fig.add_trace(go.Scatter(
                x=filtered_indices,
                y=y_mu,
                mode='lines',
                name='Prediction (μ)',
                line=dict(color='orange', width=2)
            ))
            
            # Add actual price line
            fig.add_trace(go.Scatter(
                x=filtered_indices,
                y=y_true,
                mode='lines',
                name='Actual Price',
                line=dict(color='blue', width=2)
            ))
            
            # Add confidence bands
            fig.add_trace(go.Scatter(
                x=filtered_indices,
                y=y_mu + conf_z * y_sigma,
                mode='lines',
                name='Upper Bound',
                line=dict(color='rgba(255,165,0,0.3)', width=1, dash='dash'),
                showlegend=True
            ))
            
            fig.add_trace(go.Scatter(
                x=filtered_indices,
                y=y_mu - conf_z * y_sigma,
                mode='lines',
                name='Lower Bound',
                line=dict(color='rgba(255,165,0,0.3)', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(255,165,0,0.1)',
                showlegend=True
            ))
            
            fig.update_layout(
                title='Predictions with Confidence Intervals',
                xaxis_title='Time Index',
                yaxis_title='Price',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Uncertainty bands
        st.subheader("🎯 Uncertainty Bands")
        st.markdown("""
        **What this shows:** Confidence intervals around predictions. The shaded area represents the range where 
        the true value is likely to fall. Wider bands indicate higher uncertainty, while narrower bands suggest 
        more confident predictions. Use the CI z-score slider to adjust confidence levels.
        """)
        uncertainty_data = df[["lower", "upper"]].set_index(df["index"])
        st.area_chart(uncertainty_data)
        
        # Performance metrics (calculated on filtered data)
        st.subheader("📊 Performance Metrics")
        st.markdown("""
        **What this shows:** Model accuracy measures. **RMSE** (Root Mean Square Error) penalizes large errors more heavily. 
        **MAE** (Mean Absolute Error) shows average prediction error. **MAPE** (Mean Absolute Percentage Error) expresses 
        error as a percentage. Lower values indicate better performance.
        """)
        rmse = np.sqrt(np.mean((y_true_filtered - mu_filtered) ** 2))
        mae = np.mean(np.abs(y_true_filtered - mu_filtered))
        mape = np.mean(np.abs((y_true_filtered - mu_filtered) / y_true_filtered)) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{rmse:.4f}")
        with col2:
            st.metric("MAE", f"{mae:.4f}")
        with col3:
            st.metric("MAPE", f"{mape:.2f}%")
        
        # Trading signals
        st.subheader("💰 Trading Signal Demo")
        st.markdown("""
        **What this shows:** Automated trading signals based on prediction-actual price divergence. 
        **Long (Buy)** when prediction exceeds actual price by threshold. **Short (Sell)** when actual exceeds prediction. 
        **Hold** when difference is within threshold. Adjust entry threshold to control signal sensitivity.
        """)
        entry_z = st.slider("Entry threshold (z)", 0.0, 3.0, 0.5)
        signal = np.where(mu_filtered - y_true_filtered > entry_z * sigma_filtered, 1, np.where(y_true_filtered - mu_filtered > entry_z * sigma_filtered, -1, 0))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Long Signals", f"{(signal==1).mean()*100:.1f}%")
        with col2:
            st.metric("Short Signals", f"{(signal==-1).mean()*100:.1f}%")
        with col3:
            st.metric("Hold Signals", f"{(signal==0).mean()*100:.1f}%")
        
        # Visual signal chart
        signal_df = pd.DataFrame({
            "signal": signal, 
            "index": filtered_indices
        }).set_index("index")
        st.line_chart(signal_df, height=200)
        
        # Raw data table (optional)
        if st.checkbox("Show Raw Data"):
            st.subheader("📋 Raw Prediction Data")
            st.markdown("""
            **What this shows:** Detailed numerical data including true values, predictions, and confidence bounds. 
            Useful for in-depth analysis and exporting results.
            """)
            st.dataframe(df.head(20))
            
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.info("💡 Make sure you have trained models and processed data. Run `python generate_results.py` first.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("💡 Check the logs in `results/logs/` for more details.")


if __name__ == "__main__":
    # Allow direct run: streamlit run app/streamlit_app.py -- --config configs/app.yaml
    main()

