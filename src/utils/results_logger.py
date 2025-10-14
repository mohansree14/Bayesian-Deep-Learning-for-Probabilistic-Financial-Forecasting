"""
Enhanced logging utility for saving all outputs to results folder.
"""
from __future__ import annotations
import logging
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import torch

class ResultsLogger:
    """Professional results logger for ML experiments."""
    
    def __init__(self, results_dir: Path = Path("results")):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.logs_dir = self.results_dir / "logs"
        self.plots_dir = self.results_dir / "plots"
        self.models_dir = self.results_dir / "models"
        self.data_dir = self.results_dir / "data"
        self.reports_dir = self.results_dir / "reports"
        
        for dir_path in [self.logs_dir, self.plots_dir, self.models_dir, 
                        self.data_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Experiment tracking
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.results_dir / f"experiment_{self.experiment_id}"
        self.experiment_dir.mkdir(exist_ok=True)
        
    def setup_logging(self):
        """Setup comprehensive logging."""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handlers
        log_file = self.logs_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Detailed logger for debugging
        debug_handler = logging.FileHandler(log_file)
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(detailed_formatter)
        
        # Simple logger for general info
        info_handler = logging.FileHandler(self.logs_dir / "general.log")
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(simple_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Setup root logger
        self.logger = logging.getLogger('ml_experiment')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(debug_handler)
        self.logger.addHandler(info_handler)
        self.logger.addHandler(console_handler)
        
    def log_experiment_start(self, config: Dict[str, Any], 
                           model_type: str, ticker: str):
        """Log experiment start with configuration."""
        self.logger.info(f"Starting experiment: {model_type} on {ticker}")
        self.logger.info(f"Experiment ID: {self.experiment_id}")
        
        # Save configuration
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.logger.info(f"Configuration saved to: {config_path}")
        
    def log_training_start(self, model_info: Dict[str, Any]):
        """Log training start."""
        self.logger.info("Starting model training...")
        self.logger.info(f"Model info: {model_info}")
        
        # Save model info
        model_info_path = self.experiment_dir / "model_info.json"
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
    
    def log_training_progress(self, epoch: int, train_loss: float, 
                            valid_loss: Optional[float] = None):
        """Log training progress."""
        if valid_loss is not None:
            self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, "
                           f"Valid Loss = {valid_loss:.6f}")
        else:
            self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}")
    
    def log_training_complete(self, history: Dict[str, List[float]], 
                            best_epoch: int, best_loss: float):
        """Log training completion."""
        self.logger.info(f"Training completed. Best epoch: {best_epoch}, "
                        f"Best loss: {best_loss:.6f}")
        
        # Save training history
        history_path = self.experiment_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        self.logger.info(f"Training history saved to: {history_path}")
    
    def log_evaluation_results(self, metrics: Dict[str, float]):
        """Log evaluation results."""
        self.logger.info("Evaluation Results:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.6f}")
        
        # Save metrics
        metrics_path = self.experiment_dir / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation metrics saved to: {metrics_path}")
    
    def log_backtesting_results(self, backtest_results: Dict[str, Any]):
        """Log backtesting results."""
        self.logger.info("Backtesting Results:")
        for metric, value in backtest_results.items():
            # Handle numpy arrays and other types
            try:
                if hasattr(value, 'item') and value.size == 1:  # numpy scalar
                    value = value.item()
                elif hasattr(value, 'tolist'):  # numpy array
                    value = value.tolist()
                elif hasattr(value, '__len__') and len(value) == 1:  # single element array
                    value = value[0]
            except (ValueError, AttributeError):
                pass  # Keep original value if conversion fails
            
            self.logger.info(f"  {metric}: {value:.6f}" if isinstance(value, (int, float)) else f"  {metric}: {value}")
        
        # Save backtest results
        backtest_path = self.experiment_dir / "backtest_results.json"
        with open(backtest_path, 'w') as f:
            json.dump(backtest_results, f, indent=2, default=str)
        
        self.logger.info(f"Backtest results saved to: {backtest_path}")
    
    def save_model(self, model: torch.nn.Module, model_name: str, 
                  additional_info: Optional[Dict] = None):
        """Save model checkpoint."""
        model_path = self.models_dir / f"{model_name}_{self.experiment_id}.pt"
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_info:
            save_dict.update(additional_info)
        
        torch.save(save_dict, model_path)
        self.logger.info(f"Model saved to: {model_path}")
        
        return model_path
    
    def save_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_std: Optional[np.ndarray] = None,
                        timestamps: Optional[List] = None,
                        filename: str = "predictions"):
        """Save prediction results."""
        pred_data = {
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist(),
            'timestamps': timestamps
        }
        
        if y_std is not None:
            pred_data['y_std'] = y_std.tolist()
        
        pred_path = self.experiment_dir / f"{filename}.json"
        with open(pred_path, 'w') as f:
            json.dump(pred_data, f, indent=2, default=str)
        
        self.logger.info(f"Predictions saved to: {pred_path}")
        
        # Also save as numpy arrays for easy loading
        np.save(self.experiment_dir / f"{filename}_y_true.npy", y_true)
        np.save(self.experiment_dir / f"{filename}_y_pred.npy", y_pred)
        if y_std is not None:
            np.save(self.experiment_dir / f"{filename}_y_std.npy", y_std)
    
    def save_data_summary(self, data_info: Dict[str, Any]):
        """Save data summary information."""
        data_path = self.experiment_dir / "data_summary.json"
        with open(data_path, 'w') as f:
            json.dump(data_info, f, indent=2, default=str)
        
        self.logger.info(f"Data summary saved to: {data_path}")
    
    def create_experiment_summary(self):
        """Create a comprehensive experiment summary."""
        summary = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'experiment_dir': str(self.experiment_dir),
            'files_created': []
        }
        
        # List all files created
        for file_path in self.experiment_dir.rglob('*'):
            if file_path.is_file():
                summary['files_created'].append({
                    'filename': file_path.name,
                    'path': str(file_path.relative_to(self.results_dir)),
                    'size_bytes': file_path.stat().st_size
                })
        
        # Save summary
        summary_path = self.experiment_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Experiment summary saved to: {summary_path}")
        
        return summary
    
    def log_error(self, error: Exception, context: str = ""):
        """Log errors with context."""
        self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    
    def log_warning(self, message: str, context: str = ""):
        """Log warnings with context."""
        self.logger.warning(f"Warning in {context}: {message}")
    
    def log_info(self, message: str, context: str = ""):
        """Log info messages with context."""
        self.logger.info(f"Info in {context}: {message}")
    
    def close(self):
        """Close logger and create final summary."""
        self.create_experiment_summary()
        self.logger.info(f"Experiment {self.experiment_id} completed")
        
        # Close all handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
