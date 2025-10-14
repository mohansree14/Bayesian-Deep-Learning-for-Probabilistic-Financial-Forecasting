# AWS Quickstart (EC2 + MLflow)

## EC2 setup
- Launch Ubuntu 22.04 instance (g5.2xlarge for GPU or c7i.large for CPU)
- Security group: open TCP 22 (SSH), 8501 (Streamlit if needed), 5000 (MLflow if public), or use SSH tunnel
- Install Docker: `curl -fsSL https://get.docker.com | sh`
- Clone repo and build image: `docker build -t bayes-finance:latest .`

## Running training
```
# Mount project and data, run training inside Docker
docker run --rm -it -v $PWD:/app bayes-finance:latest \
  bash -lc "python scripts/fetch_data.py --tickers AAPL --start 2015-01-01 --end 2024-12-31 --interval 1d --out data/raw && \
           python scripts/make_dataset.py --tickers AAPL --window 64 --horizon 1 && \
           python train.py --config configs/lstm_baseline.yaml"
```

## MLflow Tracking Server (optional)
- On a persistent instance or local server:
```
pip install mlflow
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns
```
- Set `MLFLOW_TRACKING_URI=http://<host>:5000` or edit configs.

## Streamlit
```
docker run --rm -it -p 8501:8501 -v $PWD:/app bayes-finance:latest
# then open http://<ec2-public-ip>:8501
```

