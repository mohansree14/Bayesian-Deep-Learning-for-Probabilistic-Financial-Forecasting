"""MLflow logging utilities."""
from __future__ import annotations
from contextlib import contextmanager
from typing import Dict, Any
import mlflow


def init_mlflow(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


@contextmanager
def mlflow_run(run_name: str, params: Dict[str, Any] | None = None):
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        yield

