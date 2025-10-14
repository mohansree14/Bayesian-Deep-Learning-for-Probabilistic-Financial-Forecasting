#!/usr/bin/env python
"""
Train models for all available tickers.
"""
import subprocess
import sys
from pathlib import Path
import yaml

def get_available_tickers(data_dir: Path):
    """Get list of available tickers from data directory."""
    if not data_dir.exists():
        return []
    
    tickers = []
    for item in data_dir.iterdir():
        if item.is_dir() and (item / "meta.json").exists():
            tickers.append(item.name)
    
    return tickers

def train_ticker_model(ticker: str, model_type: str, config_template: str):
    """Train a model for a specific ticker."""
    print(f"\n{'='*60}")
    print(f"🚀 Training {model_type.upper()} model for {ticker}")
    print('='*60)
    
    # Create temporary config for this ticker
    with open(config_template, 'r') as f:
        config = yaml.safe_load(f)
    
    config['data']['tickers'] = [ticker]
    config['model']['type'] = model_type
    
    temp_config = f"temp_config_{ticker}_{model_type}.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    try:
        # Run training
        cmd = [sys.executable, "train.py", "--config", temp_config]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ SUCCESS: {model_type.upper()} model trained for {ticker}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {model_type.upper()} model for {ticker}")
        print(f"Error: {e.stderr}")
        return False
    finally:
        # Clean up temp config
        Path(temp_config).unlink(missing_ok=True)

def main():
    print("🎯 TRAINING MODELS FOR ALL TICKERS")
    print("=" * 60)
    
    data_dir = Path("data/processed")
    available_tickers = get_available_tickers(data_dir)
    
    if not available_tickers:
        print("❌ No tickers found in data/processed directory")
        print("💡 Run data preprocessing first: python scripts/make_dataset.py")
        return 1
    
    print(f"📊 Found tickers: {', '.join(available_tickers)}")
    
    models_to_train = ["lstm", "mc_dropout_lstm"]
    success_count = 0
    total_tasks = len(available_tickers) * len(models_to_train)
    
    for ticker in available_tickers:
        for model_type in models_to_train:
            if train_ticker_model(ticker, model_type, "configs/lstm_baseline.yaml"):
                success_count += 1
    
    print(f"\n{'='*60}")
    print("📊 TRAINING SUMMARY")
    print('='*60)
    print(f"✅ Successful: {success_count}/{total_tasks}")
    print(f"📁 Models saved to: experiments/checkpoints/")
    
    if success_count == total_tasks:
        print("🎉 All models trained successfully!")
        print("\n🌐 You can now use all tickers in the Streamlit app!")
    else:
        print("⚠️  Some models failed to train. Check the output above.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

