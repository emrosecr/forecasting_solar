"""
Plotting functions for solar energy forecasting evaluation.
Creates prediction vs truth plots and residual diagnostics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
import warnings


def plot_predictions_vs_truth(y_true: pd.Series, y_pred: pd.Series, 
                             model_name: str, target_type: str,
                             output_path: str, figsize: Tuple[int, int] = (10, 8)):
    """
    Create prediction vs truth scatter plot.
    
    Parameters:
    -----------
    y_true : pd.Series
        True values
    y_pred : pd.Series
        Predicted values
    model_name : str
        Name of the model
    target_type : str
        'anom' or 'total'
    output_path : str
        Path to save the plot
    figsize : Tuple[int, int]
        Figure size
    """
    # Align indices and remove NaN
    common_idx = y_true.index.intersection(y_pred.index)
    y_true_aligned = y_true.loc[common_idx]
    y_pred_aligned = y_pred.loc[common_idx]
    
    valid_mask = ~(y_true_aligned.isna() | y_pred_aligned.isna())
    if valid_mask.sum() == 0:
        warnings.warn(f"No valid data for plotting {model_name}")
        return
    
    y_true_clean = y_true_aligned[valid_mask]
    y_pred_clean = y_pred_aligned[valid_mask]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot
    ax1.scatter(y_true_clean, y_pred_clean, alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(y_true_clean.min(), y_pred_clean.min())
    max_val = max(y_true_clean.max(), y_pred_clean.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate metrics for annotation
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    ax1.set_xlabel(f'True {target_type.title()} Values')
    ax1.set_ylabel(f'Predicted {target_type.title()} Values')
    ax1.set_title(f'{model_name}: Predictions vs Truth\nRMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_pred_clean - y_true_clean
    ax2.scatter(y_pred_clean, residuals, alpha=0.6, s=20)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    ax2.set_xlabel(f'Predicted {target_type.title()} Values')
    ax2.set_ylabel('Residuals (Predicted - True)')
    ax2.set_title(f'{model_name}: Residuals vs Predictions')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_time_series(y_true: pd.Series, y_pred: pd.Series,
                    model_name: str, target_type: str,
                    output_path: str, figsize: Tuple[int, int] = (15, 6),
                    start_date: Optional[str] = None, end_date: Optional[str] = None):
    """
    Create time series plot comparing true and predicted values.
    
    Parameters:
    -----------
    y_true : pd.Series
        True values
    y_pred : pd.Series
        Predicted values
    model_name : str
        Name of the model
    target_type : str
        'anom' or 'total'
    output_path : str
        Path to save the plot
    figsize : Tuple[int, int]
        Figure size
    start_date : str, optional
        Start date for plotting (YYYY-MM-DD)
    end_date : str, optional
        End date for plotting (YYYY-MM-DD)
    """
    # Align indices
    common_idx = y_true.index.intersection(y_pred.index)
    y_true_aligned = y_true.loc[common_idx]
    y_pred_aligned = y_pred.loc[common_idx]
    
    # Apply date filter if specified
    if start_date:
        y_true_aligned = y_true_aligned[y_true_aligned.index >= start_date]
        y_pred_aligned = y_pred_aligned[y_pred_aligned.index >= start_date]
    
    if end_date:
        y_true_aligned = y_true_aligned[y_true_aligned.index <= end_date]
        y_pred_aligned = y_pred_aligned[y_pred_aligned.index <= end_date]
    
    # Remove NaN values
    valid_mask = ~(y_true_aligned.isna() | y_pred_aligned.isna())
    y_true_clean = y_true_aligned[valid_mask]
    y_pred_clean = y_pred_aligned[valid_mask]
    
    if len(y_true_clean) == 0:
        warnings.warn(f"No valid data for time series plot {model_name}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot time series
    ax.plot(y_true_clean.index, y_true_clean.values, label='True', alpha=0.8, linewidth=1)
    ax.plot(y_pred_clean.index, y_pred_clean.values, label='Predicted', alpha=0.8, linewidth=1)
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{target_type.title()} Values')
    ax.set_title(f'{model_name}: Time Series Comparison\nRMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_residual_histogram(y_true: pd.Series, y_pred: pd.Series,
                           model_name: str, target_type: str,
                           output_path: str, figsize: Tuple[int, int] = (8, 6)):
    """
    Create residual histogram and Q-Q plot.
    
    Parameters:
    -----------
    y_true : pd.Series
        True values
    y_pred : pd.Series
        Predicted values
    model_name : str
        Name of the model
    target_type : str
        'anom' or 'total'
    output_path : str
        Path to save the plot
    figsize : Tuple[int, int]
        Figure size
    """
    # Align indices and remove NaN
    common_idx = y_true.index.intersection(y_pred.index)
    y_true_aligned = y_true.loc[common_idx]
    y_pred_aligned = y_pred.loc[common_idx]
    
    valid_mask = ~(y_true_aligned.isna() | y_pred_aligned.isna())
    if valid_mask.sum() == 0:
        warnings.warn(f"No valid data for residual plot {model_name}")
        return
    
    y_true_clean = y_true_aligned[valid_mask]
    y_pred_clean = y_pred_aligned[valid_mask]
    residuals = y_pred_clean - y_true_clean
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax1.hist(residuals, bins=30, alpha=0.7, density=True, edgecolor='black')
    ax1.axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {residuals.mean():.2f}')
    ax1.set_xlabel('Residuals')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{model_name}: Residual Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title(f'{model_name}: Q-Q Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(baseline_results: Dict, rf_results: Dict,
                         output_dir: str = "outputs/plots"):
    """
    Create comparison plots for all models.
    
    Parameters:
    -----------
    baseline_results : Dict
        Baseline model results
    rf_results : Dict
        Random Forest results
    output_dir : str
        Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create RMSE comparison bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Anomaly models
    anom_models = []
    anom_rmse = []
    
    if 'anom' in baseline_results:
        for model_name, metrics in baseline_results['anom'].items():
            anom_models.append(f"Baseline {model_name}")
            anom_rmse.append(metrics.get('rmse', np.nan))
    
    if 'anom' in rf_results:
        for feature_type, metrics in rf_results['anom'].items():
            anom_models.append(f"RF {feature_type}")
            anom_rmse.append(metrics.get('rmse', np.nan))
    
    bars1 = ax1.bar(range(len(anom_models)), anom_rmse, alpha=0.7)
    ax1.set_xticks(range(len(anom_models)))
    ax1.set_xticklabels(anom_models, rotation=45, ha='right')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Anomaly Models: RMSE Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, anom_rmse):
        if not np.isnan(value):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
    
    # Total models
    total_models = []
    total_rmse = []
    
    if 'total' in baseline_results:
        for model_name, metrics in baseline_results['total'].items():
            total_models.append(f"Baseline {model_name}")
            total_rmse.append(metrics.get('rmse', np.nan))
    
    if 'total' in rf_results:
        for feature_type, metrics in rf_results['total'].items():
            total_models.append(f"RF {feature_type}")
            total_rmse.append(metrics.get('rmse', np.nan))
    
    bars2 = ax2.bar(range(len(total_models)), total_rmse, alpha=0.7)
    ax2.set_xticks(range(len(total_models)))
    ax2.set_xticklabels(total_models, rotation=45, ha='right')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Total Models: RMSE Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, total_rmse):
        if not np.isnan(value):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    comparison_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved model comparison plot: {comparison_path}")


def create_all_plots(baseline_suite, rf_suite, gen_series: pd.Series, 
                    anom_series: pd.Series, config: dict,
                    output_dir: str = "outputs/plots"):
    """
    Create all evaluation plots for baseline and RF models.
    
    Parameters:
    -----------
    baseline_suite
        Trained baseline model suite
    rf_suite
        Trained RF model suite
    gen_series : pd.Series
        Generation series
    anom_series : pd.Series
        Anomaly series
    config : dict
        Configuration dictionary
    output_dir : str
        Output directory
    """
    print("Creating evaluation plots...")
    
    # Create train/val/test masks
    from .models_baseline import create_train_val_test_masks
    train_mask, val_mask, test_mask = create_train_val_test_masks(gen_series.index, config)
    
    # Generate predictions for plotting
    X_val = gen_series[val_mask].index.to_frame(index=False)
    X_test = gen_series[test_mask].index.to_frame(index=False)
    
    # Baseline predictions
    baseline_pred_anom_val = baseline_suite.predict(
        pd.DataFrame(index=gen_series[val_mask].index), 
        anom_series[train_mask | val_mask], 
        gen_series[train_mask | val_mask]
    )
    
    baseline_pred_total_val = baseline_suite.predict(
        pd.DataFrame(index=gen_series[val_mask].index), 
        anom_series[train_mask | val_mask], 
        gen_series[train_mask | val_mask]
    )
    
    # RF predictions (simplified - you may need to load features)
    # This is a placeholder - in practice you'd load the actual features
    print("Note: RF prediction plots require feature data - implement as needed")
    
    # Create plots for best baseline models
    # Anomaly models
    for model_name in ['linear', 'seasonal']:
        if model_name in baseline_pred_anom_val['anom']:
            # Validation plots
            plot_predictions_vs_truth(
                anom_series[val_mask], 
                baseline_pred_anom_val['anom'][model_name],
                f"Baseline {model_name}", "anomaly",
                os.path.join(output_dir, f"baseline_{model_name}_anom_val_scatter.png")
            )
            
            plot_time_series(
                anom_series[val_mask], 
                baseline_pred_anom_val['anom'][model_name],
                f"Baseline {model_name}", "anomaly",
                os.path.join(output_dir, f"baseline_{model_name}_anom_val_timeseries.png")
            )
            
            plot_residual_histogram(
                anom_series[val_mask], 
                baseline_pred_anom_val['anom'][model_name],
                f"Baseline {model_name}", "anomaly",
                os.path.join(output_dir, f"baseline_{model_name}_anom_val_residuals.png")
            )
    
    # Total models
    for model_name in ['linear', 'seasonal']:
        if model_name in baseline_pred_total_val['total']:
            # Validation plots
            plot_predictions_vs_truth(
                gen_series[val_mask], 
                baseline_pred_total_val['total'][model_name],
                f"Baseline {model_name}", "total",
                os.path.join(output_dir, f"baseline_{model_name}_total_val_scatter.png")
            )
            
            plot_time_series(
                gen_series[val_mask], 
                baseline_pred_total_val['total'][model_name],
                f"Baseline {model_name}", "total",
                os.path.join(output_dir, f"baseline_{model_name}_total_val_timeseries.png")
            )
            
            plot_residual_histogram(
                gen_series[val_mask], 
                baseline_pred_total_val['total'][model_name],
                f"Baseline {model_name}", "total",
                os.path.join(output_dir, f"baseline_{model_name}_total_val_residuals.png")
            )
    
    # Model comparison plot
    plot_model_comparison(baseline_suite.results, rf_suite.results, output_dir)
    
    print(f"Created evaluation plots in {output_dir}")
