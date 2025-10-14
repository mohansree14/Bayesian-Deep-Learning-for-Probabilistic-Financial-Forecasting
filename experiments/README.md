# Experiments Log

- Use MLflow to track runs. Default local tracking URI is `./mlruns`.
- Suggested naming: `<model>_<ticker>_<date>`
- Record configs under `configs/` and commit any changes.

Example:
- `python train.py --config configs/lstm_baseline.yaml`
- `python train.py --config configs/mc_dropout.yaml`
- `python train_bnn.py --config configs/bnn_vi.yaml`

