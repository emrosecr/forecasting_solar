"""
Evaluation functions for solar energy forecasting models.
Calculates metrics and creates comparison tables.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Tuple, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series, target_type: str) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters:
    -----------
    y_true : pd.Series
        True values
    y_pred : pd.Series
        Predicted values
    target_type : str
        'anom' or 'total' (affects which metrics are calculated)
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of metrics
    """
    # Align indices
    common_idx = y_true.index.intersection(y_pred.index)
    y_true_aligned = y_true.loc[common_idx]
    y_pred_aligned = y_pred.loc[common_idx]
    
    # Remove NaN values
    valid_mask = ~(y_true_aligned.isna() | y_pred_aligned.isna())
    if valid_mask.sum() == 0:
        warnings.warn("No valid data points for metric calculation")
        return {key: np.nan for key in ['mae', 'rmse', 'r2', 'mape', 'n_samples']}
    
    y_true_clean = y_true_aligned[valid_mask]
    y_pred_clean = y_pred_aligned[valid_mask]
    
    # Basic metrics
    metrics = {
        'mae': mean_absolute_error(y_true_clean, y_pred_clean),
        'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
        'r2': r2_score(y_true_clean, y_pred_clean),
        'n_samples': len(y_true_clean)
    }
    
    # MAPE for total targets (avoid division by zero)
    if target_type == 'total':
        # Avoid division by zero
        non_zero_mask = y_true_clean != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_true_clean[non_zero_mask] - y_pred_clean[non_zero_mask]) / y_true_clean[non_zero_mask])) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = np.nan
    
    # Additional metrics for anomaly targets
    if target_type == 'anom':
        # RMSE of anomalies (already calculated as rmse)
        # Correlation coefficient
        corr = np.corrcoef(y_true_clean, y_pred_clean)[0, 1]
        metrics['correlation'] = corr
        
        # Skill score (1 - MSE/MSE_persistence)
        # Assuming persistence is lag-1
        if len(y_true_clean) > 1:
            persistence_pred = y_true_clean.shift(1)[1:]
            persistence_true = y_true_clean[1:]
            mse_persistence = mean_squared_error(persistence_true, persistence_pred)
            mse_model = mean_squared_error(y_true_clean, y_pred_clean)
            if mse_persistence > 0:
                skill_score = 1 - mse_model / mse_persistence
                metrics['skill_score'] = skill_score
            else:
                metrics['skill_score'] = np.nan
    
    return metrics


def evaluate_model_suite(baseline_results: Dict, rf_results: Dict,
                        target_type: str, split: str = 'val') -> pd.DataFrame:
    """
    Evaluate and compare all models for a given target type and split.
    
    Parameters:
    -----------
    baseline_results : Dict
        Baseline model results
    rf_results : Dict
        Random Forest results
    target_type : str
        'anom' or 'total'
    split : str
        'val' or 'test'
        
    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    comparison_data = []
    
    # Add baseline models
    if target_type in baseline_results:
        for model_name, metrics in baseline_results[target_type].items():
            comparison_data.append({
                'Model': f"Baseline {model_name}",
                'MAE': metrics.get('mae', np.nan),
                'RMSE': metrics.get('rmse', np.nan),
                'R²': metrics.get('r2', np.nan),
                'MAPE': metrics.get('mape', np.nan)
            })
    
    # Add Random Forest models
    if target_type in rf_results:
        for feature_type, metrics in rf_results[target_type].items():
            comparison_data.append({
                'Model': f"RF {feature_type}",
                'MAE': metrics.get('mae', np.nan),
                'RMSE': metrics.get('rmse', np.nan),
                'R²': metrics.get('r2', np.nan),
                'MAPE': metrics.get('mape', np.nan)
            })
    
    return pd.DataFrame(comparison_data)


def create_comparison_table(baseline_results: Dict, rf_results: Dict,
                           output_dir: str = "outputs/metrics") -> Dict[str, pd.DataFrame]:
    """
    Create comprehensive comparison tables for all models.
    
    Parameters:
    -----------
    baseline_results : Dict
        Baseline model results
    rf_results : Dict
        Random Forest results
    output_dir : str
        Output directory
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Comparison tables for each target type and split
    """
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_tables = {}
    
    for target_type in ['anom', 'total']:
        for split in ['val', 'test']:
            # Note: This assumes results are stored with split information
            # You may need to modify based on how results are structured
            table = evaluate_model_suite(baseline_results, rf_results, target_type, split)
            comparison_tables[f"{target_type}_{split}"] = table
            
            # Save to CSV
            csv_path = os.path.join(output_dir, f"comparison_{target_type}_{split}.csv")
            table.to_csv(csv_path, index=False)
            print(f"Saved comparison table: {csv_path}")
    
    return comparison_tables


def save_metrics_json(baseline_results: Dict, rf_results: Dict,
                     output_dir: str = "outputs/metrics"):
    """
    Save all metrics to JSON files.
    
    Parameters:
    -----------
    baseline_results : Dict
        Baseline model results
    rf_results : Dict
        Random Forest results
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save baseline results
    baseline_path = os.path.join(output_dir, "baseline_results.json")
    with open(baseline_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        baseline_serializable = _convert_to_serializable(baseline_results)
        json.dump(baseline_serializable, f, indent=2)
    
    # Save RF results
    rf_path = os.path.join(output_dir, "rf_results.json")
    with open(rf_path, 'w') as f:
        rf_serializable = _convert_to_serializable(rf_results)
        json.dump(rf_serializable, f, indent=2)
    
    print(f"Saved metrics JSON files to {output_dir}")


def _convert_to_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def print_model_comparison(baseline_results: Dict, rf_results: Dict):
    """
    Print a formatted comparison of all models.
    
    Parameters:
    -----------
    baseline_results : Dict
        Baseline model results
    rf_results : Dict
        Random Forest results
    """
    print("\n" + "="*100)
    print("MODEL COMPARISON SUMMARY")
    print("="*100)
    
    for target_type in ['anom', 'total']:
        print(f"\n{target_type.upper()} TARGET:")
        print("-" * 80)
        
        # Collect all models for this target type
        all_models = []
        
        # Add baseline models
        if target_type in baseline_results:
            for model_name, metrics in baseline_results[target_type].items():
                all_models.append({
                    'name': f"Baseline {model_name}",
                    'rmse': metrics.get('rmse', np.nan),
                    'mae': metrics.get('mae', np.nan),
                    'r2': metrics.get('r2', np.nan)
                })
        
        # Add RF models
        if target_type in rf_results:
            for feature_type, metrics in rf_results[target_type].items():
                all_models.append({
                    'name': f"RF {feature_type}",
                    'rmse': metrics.get('rmse', np.nan),
                    'mae': metrics.get('mae', np.nan),
                    'r2': metrics.get('r2', np.nan)
                })
        
        # Sort by RMSE (ascending)
        all_models.sort(key=lambda x: x['rmse'] if not np.isnan(x['rmse']) else np.inf)
        
        # Print header
        print(f"{'Model':<25} {'RMSE':<12} {'MAE':<12} {'R²':<10}")
        print("-" * 80)
        
        # Print models
        for model in all_models:
            rmse_str = f"{model['rmse']:.2f}" if not np.isnan(model['rmse']) else "N/A"
            mae_str = f"{model['mae']:.2f}" if not np.isnan(model['mae']) else "N/A"
            r2_str = f"{model['r2']:.3f}" if not np.isnan(model['r2']) else "N/A"
            
            print(f"{model['name']:<25} {rmse_str:<12} {mae_str:<12} {r2_str:<10}")


def calculate_improvement_over_baseline(baseline_results: Dict, rf_results: Dict) -> Dict:
    """
    Calculate improvement of RF models over baseline models.
    
    Parameters:
    -----------
    baseline_results : Dict
        Baseline model results
    rf_results : Dict
        Random Forest results
        
    Returns:
    --------
    Dict
        Improvement statistics
    """
    improvements = {}
    
    for target_type in ['anom', 'total']:
        improvements[target_type] = {}
        
        # Find best baseline model
        if target_type in baseline_results:
            best_baseline_rmse = min(
                metrics.get('rmse', np.inf) 
                for metrics in baseline_results[target_type].values()
                if not np.isnan(metrics.get('rmse', np.nan))
            )
            
            if best_baseline_rmse < np.inf:
                # Calculate improvement for each RF model
                if target_type in rf_results:
                    for feature_type, metrics in rf_results[target_type].items():
                        rf_rmse = metrics.get('rmse', np.nan)
                        if not np.isnan(rf_rmse):
                            improvement_pct = ((best_baseline_rmse - rf_rmse) / best_baseline_rmse) * 100
                            improvements[target_type][feature_type] = {
                                'rmse_improvement_pct': improvement_pct,
                                'baseline_rmse': best_baseline_rmse,
                                'rf_rmse': rf_rmse
                            }
    
    return improvements


def create_evaluation_report(baseline_results: Dict, rf_results: Dict,
                           output_dir: str = "outputs/metrics") -> str:
    """
    Create a comprehensive evaluation report.
    
    Parameters:
    -----------
    baseline_results : Dict
        Baseline model results
    rf_results : Dict
        Random Forest results
    output_dir : str
        Output directory
        
    Returns:
    --------
    str
        Path to the generated report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("SOLAR ENERGY FORECASTING - EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Model comparison
        f.write("MODEL COMPARISON\n")
        f.write("-"*30 + "\n\n")
        
        for target_type in ['anom', 'total']:
            f.write(f"{target_type.upper()} TARGET:\n")
            
            # Collect all models
            all_models = []
            
            if target_type in baseline_results:
                for model_name, metrics in baseline_results[target_type].items():
                    all_models.append({
                        'name': f"Baseline {model_name}",
                        'rmse': metrics.get('rmse', np.nan),
                        'mae': metrics.get('mae', np.nan),
                        'r2': metrics.get('r2', np.nan)
                    })
            
            if target_type in rf_results:
                for feature_type, metrics in rf_results[target_type].items():
                    all_models.append({
                        'name': f"RF {feature_type}",
                        'rmse': metrics.get('rmse', np.nan),
                        'mae': metrics.get('mae', np.nan),
                        'r2': metrics.get('r2', np.nan)
                    })
            
            # Sort by RMSE
            all_models.sort(key=lambda x: x['rmse'] if not np.isnan(x['rmse']) else np.inf)
            
            for model in all_models:
                f.write(f"  {model['name']:<25}: RMSE={model['rmse']:.2f}, "
                       f"MAE={model['mae']:.2f}, R²={model['r2']:.3f}\n")
            
            f.write("\n")
        
        # Improvement analysis
        improvements = calculate_improvement_over_baseline(baseline_results, rf_results)
        
        f.write("IMPROVEMENT OVER BASELINE\n")
        f.write("-"*30 + "\n")
        
        for target_type in ['anom', 'total']:
            if target_type in improvements:
                f.write(f"{target_type.upper()} TARGET:\n")
                for feature_type, stats in improvements[target_type].items():
                    f.write(f"  RF {feature_type}: {stats['rmse_improvement_pct']:.1f}% improvement "
                           f"({stats['baseline_rmse']:.2f} → {stats['rf_rmse']:.2f})\n")
                f.write("\n")
        
        # Summary
        f.write("SUMMARY\n")
        f.write("-"*30 + "\n")
        
        best_models = {}
        for target_type in ['anom', 'total']:
            all_models = []
            
            if target_type in baseline_results:
                for model_name, metrics in baseline_results[target_type].items():
                    all_models.append(('baseline', model_name, metrics.get('rmse', np.nan)))
            
            if target_type in rf_results:
                for feature_type, metrics in rf_results[target_type].items():
                    all_models.append(('rf', feature_type, metrics.get('rmse', np.nan)))
            
            # Find best model
            valid_models = [(model_type, model_name, rmse) for model_type, model_name, rmse in all_models 
                           if not np.isnan(rmse)]
            
            if valid_models:
                best_model = min(valid_models, key=lambda x: x[2])
                best_models[target_type] = best_model
        
        for target_type, (model_type, model_name, rmse) in best_models.items():
            f.write(f"Best {target_type} model: {model_type} {model_name} (RMSE: {rmse:.2f})\n")
    
    print(f"Created evaluation report: {report_path}")
    return report_path
