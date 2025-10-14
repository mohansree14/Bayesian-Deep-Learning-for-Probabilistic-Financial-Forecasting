"""
Professional visualization utilities for ML results.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Professional color scheme
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
    'neutral': '#6C757D',
    'light': '#F8F9FA',
    'dark': '#212529'
}

def setup_plot_style():
    """Set up professional plotting style."""
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 300,
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white'
    })

def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[Path] = None,
                         title: str = "Training Progress") -> plt.Figure:
    """Plot training and validation loss curves."""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], 
            color=COLORS['primary'], linewidth=2.5, 
            label='Training Loss', marker='o', markersize=6)
    
    if 'valid_loss' in history:
        ax.plot(epochs, history['valid_loss'], 
                color=COLORS['secondary'], linewidth=2.5, 
                label='Validation Loss', marker='s', markersize=6)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    # Add final values as text
    final_train = history['train_loss'][-1]
    ax.text(0.02, 0.98, f'Final Train Loss: {final_train:.4f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if 'valid_loss' in history:
        final_valid = history['valid_loss'][-1]
        ax.text(0.02, 0.90, f'Final Valid Loss: {final_valid:.4f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    return fig

def plot_forecasts(y_true: np.ndarray, 
                  y_pred: np.ndarray, 
                  y_std: Optional[np.ndarray] = None,
                  timestamps: Optional[List] = None,
                  save_path: Optional[Path] = None,
                  title: str = "Forecast vs Actual") -> plt.Figure:
    """Plot forecast predictions with uncertainty bands."""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    n_points = len(y_true)
    x_vals = timestamps if timestamps else range(n_points)
    
    # Plot actual values
    ax.plot(x_vals, y_true, color=COLORS['dark'], linewidth=2.5, 
            label='Actual', alpha=0.8)
    
    # Plot predictions
    ax.plot(x_vals, y_pred, color=COLORS['primary'], linewidth=2.5, 
            label='Predicted', alpha=0.9)
    
    # Plot uncertainty bands if available
    if y_std is not None:
        upper_bound = y_pred + 1.96 * y_std  # 95% confidence interval
        lower_bound = y_pred - 1.96 * y_std
        
        ax.fill_between(x_vals, lower_bound, upper_bound, 
                       color=COLORS['primary'], alpha=0.2, 
                       label='95% Confidence Interval')
    
    ax.set_xlabel('Time', fontweight='bold')
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis for dates if timestamps provided
    if timestamps and isinstance(timestamps[0], (pd.Timestamp, str)):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    return fig

def plot_residuals(y_true: np.ndarray, 
                  y_pred: np.ndarray,
                  save_path: Optional[Path] = None,
                  title: str = "Residual Analysis") -> plt.Figure:
    """Plot residual analysis."""
    setup_plot_style()
    
    residuals = y_true - y_pred
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.6, color=COLORS['primary'])
    ax1.axhline(y=0, color=COLORS['dark'], linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Values', fontweight='bold')
    ax1.set_ylabel('Residuals', fontweight='bold')
    ax1.set_title('Residuals vs Predicted', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Histogram of residuals
    ax2.hist(residuals, bins=30, alpha=0.7, color=COLORS['primary'], 
             edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Residuals', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Distribution of Residuals', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Residuals over time
    ax4.plot(residuals, color=COLORS['primary'], linewidth=1.5)
    ax4.axhline(y=0, color=COLORS['dark'], linestyle='--', linewidth=2)
    ax4.set_xlabel('Time Index', fontweight='bold')
    ax4.set_ylabel('Residuals', fontweight='bold')
    ax4.set_title('Residuals Over Time', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    return fig

def plot_metrics_comparison(metrics: Dict[str, float],
                          save_path: Optional[Path] = None,
                          title: str = "Model Performance Metrics") -> plt.Figure:
    """Plot metrics comparison bar chart."""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_names = list(metrics.keys())
    metrics_values = list(metrics.values())
    
    bars = ax.bar(metrics_names, metrics_values, 
                 color=[COLORS['primary'], COLORS['secondary'], 
                       COLORS['accent'], COLORS['success']][:len(metrics_names)],
                 alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Metric Value', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    return fig

def plot_backtesting_results(returns: np.ndarray,
                           cumulative_returns: np.ndarray,
                           drawdown: np.ndarray,
                           timestamps: Optional[List] = None,
                           save_path: Optional[Path] = None,
                           title: str = "Backtesting Results") -> plt.Figure:
    """Plot backtesting results."""
    setup_plot_style()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    x_vals = timestamps if timestamps else range(len(returns))
    
    # Cumulative returns
    ax1.plot(x_vals, cumulative_returns, color=COLORS['primary'], linewidth=2.5)
    ax1.set_title('Cumulative Returns', fontweight='bold')
    ax1.set_ylabel('Cumulative Return', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Daily returns
    ax2.plot(x_vals, returns, color=COLORS['secondary'], linewidth=1.5, alpha=0.7)
    ax2.axhline(y=0, color=COLORS['dark'], linestyle='--', linewidth=2)
    ax2.set_title('Daily Returns', fontweight='bold')
    ax2.set_ylabel('Daily Return', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Drawdown
    ax3.fill_between(x_vals, drawdown, 0, color=COLORS['success'], alpha=0.7)
    ax3.plot(x_vals, drawdown, color=COLORS['dark'], linewidth=1.5)
    ax3.set_title('Drawdown', fontweight='bold')
    ax3.set_ylabel('Drawdown', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Returns distribution
    ax4.hist(returns, bins=30, alpha=0.7, color=COLORS['accent'], 
             edgecolor='black', linewidth=0.5)
    ax4.axvline(x=0, color=COLORS['dark'], linestyle='--', linewidth=2)
    ax4.set_title('Returns Distribution', fontweight='bold')
    ax4.set_xlabel('Daily Return', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    return fig

def create_summary_dashboard(history: Dict[str, List[float]],
                           metrics: Dict[str, float],
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           save_path: Optional[Path] = None) -> plt.Figure:
    """Create a comprehensive summary dashboard."""
    setup_plot_style()
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Training history
    ax1 = fig.add_subplot(gs[0, :2])
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], color=COLORS['primary'], 
             linewidth=2.5, label='Training Loss', marker='o')
    if 'valid_loss' in history:
        ax1.plot(epochs, history['valid_loss'], color=COLORS['secondary'], 
                 linewidth=2.5, label='Validation Loss', marker='s')
    ax1.set_title('Training Progress', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Metrics bar chart
    ax2 = fig.add_subplot(gs[0, 2:])
    metrics_names = list(metrics.keys())
    metrics_values = list(metrics.values())
    bars = ax2.bar(metrics_names, metrics_values, 
                   color=[COLORS['primary'], COLORS['secondary'], 
                         COLORS['accent'], COLORS['success']][:len(metrics_names)])
    ax2.set_title('Performance Metrics', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Value')
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Forecast vs Actual
    ax3 = fig.add_subplot(gs[1, :])
    n_points = min(len(y_true), len(y_pred), 200)  # Limit for visibility
    x_vals = range(n_points)
    ax3.plot(x_vals, y_true[:n_points], color=COLORS['dark'], 
             linewidth=2, label='Actual', alpha=0.8)
    ax3.plot(x_vals, y_pred[:n_points], color=COLORS['primary'], 
             linewidth=2, label='Predicted', alpha=0.9)
    ax3.set_title('Forecast vs Actual (Sample)', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Time Index')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Residuals
    ax4 = fig.add_subplot(gs[2, :2])
    residuals = y_true - y_pred
    ax4.scatter(y_pred, residuals, alpha=0.6, color=COLORS['primary'])
    ax4.axhline(y=0, color=COLORS['dark'], linestyle='--', linewidth=2)
    ax4.set_title('Residuals vs Predicted', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Predicted Values')
    ax4.set_ylabel('Residuals')
    ax4.grid(True, alpha=0.3)
    
    # Residuals distribution
    ax5 = fig.add_subplot(gs[2, 2:])
    ax5.hist(residuals, bins=30, alpha=0.7, color=COLORS['accent'], 
             edgecolor='black', linewidth=0.5)
    ax5.axvline(x=0, color=COLORS['dark'], linestyle='--', linewidth=2)
    ax5.set_title('Residuals Distribution', fontweight='bold', fontsize=14)
    ax5.set_xlabel('Residuals')
    ax5.set_ylabel('Frequency')
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle('ML Model Performance Dashboard', fontsize=20, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    return fig

def save_plot(fig: plt.Figure, save_path: Path, 
              title: str = "Plot", dpi: int = 300) -> None:
    """Save plot with professional formatting."""
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close(fig)

