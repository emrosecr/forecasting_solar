"""
Baseline statistical models for solar energy forecasting.
Implements persistence, seasonal, and simple linear regression models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import warnings


class PersistenceModel:
    """Simple persistence model (yesterday's value)."""
    
    def __init__(self, lag: int = 1):
        self.lag = lag
        self.name = f"Persistence (lag {lag})"
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit persistence model (no training needed)."""
        pass
    
    def predict(self, X: pd.DataFrame, y_hist: pd.Series) -> pd.Series:
        """Predict using historical values."""
        predictions = y_hist.shift(self.lag)
        return predictions.loc[X.index]


class SeasonalModel:
    """Seasonal model (same day of year from previous year)."""
    
    def __init__(self):
        self.name = "Seasonal (same DOY)"
        self.climatology = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit climatology using day of year."""
        # Calculate day of year climatology
        doy = X.index.dayofyear
        self.climatology = y.groupby(doy).mean()
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict using climatology."""
        doy = X.index.dayofyear
        predictions = doy.map(self.climatology)
        return pd.Series(predictions, index=X.index)


class LinearBaselineModel:
    """Linear regression baseline with selected features."""
    
    def __init__(self, feature_names: list):
        self.feature_names = feature_names
        self.model = LinearRegression()
        self.name = f"Linear ({len(feature_names)} features)"
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit linear model."""
        # Select features
        X_selected = X[self.feature_names].copy()
        
        # Remove rows with missing values
        valid_mask = ~(X_selected.isna().any(axis=1) | y.isna())
        X_train = X_selected[valid_mask]
        y_train = y[valid_mask]
        
        if len(X_train) == 0:
            warnings.warn("No valid training data for linear model")
            return
        
        # Fit model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict using fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Select features
        X_selected = X[self.feature_names].copy()
        
        # Handle missing values by filling with median
        for col in X_selected.columns:
            if X_selected[col].isna().any():
                X_selected[col] = X_selected[col].fillna(X_selected[col].median())
        
        # Predict
        predictions = self.model.predict(X_selected)
        return pd.Series(predictions, index=X.index)


class BaselineModelSuite:
    """Suite of baseline models for both anomaly and total targets."""
    
    def __init__(self, config: dict):
        self.config = config
        self.models = {}
        self.results = {}
    
    def fit_models(self, X_train: pd.DataFrame, y_train_anom: pd.Series, 
                  y_train_total: pd.Series):
        """Fit all baseline models on training data."""
        print("Fitting baseline models...")
        
        # Get linear model features
        linear_features = self.config['models']['baseline']['linear_features']
        
        # Anomaly models
        self.models['anom'] = {
            'persistence_1': PersistenceModel(lag=1),
            'persistence_7': PersistenceModel(lag=7),
            'seasonal': SeasonalModel(),
            'linear': LinearBaselineModel(linear_features)
        }
        
        # Total models
        self.models['total'] = {
            'persistence_1': PersistenceModel(lag=1),
            'persistence_7': PersistenceModel(lag=7),
            'seasonal': SeasonalModel(),
            'linear': LinearBaselineModel(linear_features)
        }
        
        # Fit anomaly models
        print("  Fitting anomaly models...")
        for name, model in self.models['anom'].items():
            print(f"    {model.name}")
            model.fit(X_train, y_train_anom)
        
        # Fit total models
        print("  Fitting total models...")
        for name, model in self.models['total'].items():
            print(f"    {model.name}")
            model.fit(X_train, y_train_total)
    
    def predict(self, X: pd.DataFrame, y_hist_anom: pd.Series, 
               y_hist_total: pd.Series) -> Dict[str, pd.Series]:
        """Generate predictions for all models."""
        predictions = {}
        
        # Anomaly predictions
        predictions['anom'] = {}
        for name, model in self.models['anom'].items():
            if 'persistence' in name:
                pred = model.predict(X, y_hist_anom)
            else:
                pred = model.predict(X)
            predictions['anom'][name] = pred
        
        # Total predictions
        predictions['total'] = {}
        for name, model in self.models['total'].items():
            if 'persistence' in name:
                pred = model.predict(X, y_hist_total)
            else:
                pred = model.predict(X)
            predictions['total'][name] = pred
        
        return predictions
    
    def evaluate_models(self, X_val: pd.DataFrame, y_val_anom: pd.Series,
                       y_val_total: pd.Series, y_hist_anom: pd.Series,
                       y_hist_total: pd.Series) -> Dict[str, Dict[str, Dict]]:
        """Evaluate all models on validation data."""
        print("Evaluating baseline models...")
        
        # Generate predictions
        predictions = self.predict(X_val, y_hist_anom, y_hist_total)
        
        # Calculate metrics
        results = {}
        
        # Anomaly results
        results['anom'] = {}
        for name, pred in predictions['anom'].items():
            # Align indices
            common_idx = pred.index.intersection(y_val_anom.index)
            pred_aligned = pred.loc[common_idx]
            y_true = y_val_anom.loc[common_idx]
            
            # Remove NaN values
            valid_mask = ~(pred_aligned.isna() | y_true.isna())
            if valid_mask.sum() > 0:
                pred_clean = pred_aligned[valid_mask]
                y_clean = y_true[valid_mask]
                
                results['anom'][name] = self._calculate_metrics(y_clean, pred_clean, 'anom')
            else:
                results['anom'][name] = {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}
        
        # Total results
        results['total'] = {}
        for name, pred in predictions['total'].items():
            # Align indices
            common_idx = pred.index.intersection(y_val_total.index)
            pred_aligned = pred.loc[common_idx]
            y_true = y_val_total.loc[common_idx]
            
            # Remove NaN values
            valid_mask = ~(pred_aligned.isna() | y_true.isna())
            if valid_mask.sum() > 0:
                pred_clean = pred_aligned[valid_mask]
                y_clean = y_true[valid_mask]
                
                results['total'][name] = self._calculate_metrics(y_clean, pred_clean, 'total')
            else:
                results['total'][name] = {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}
        
        self.results = results
        return results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series, 
                          target_type: str) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
        
        # Add MAPE for total targets
        if target_type == 'total' and y_true.abs().sum() > 0:
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['mape'] = mape
        
        return metrics
    
    def save_models(self, output_dir: str = "outputs/models"):
        """Save trained models."""
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, "baseline_stats.pkl")
        
        # Prepare model data for saving
        model_data = {
            'anom_models': self.models['anom'],
            'total_models': self.models['total'],
            'results': self.results,
            'config': self.config
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Saved baseline models: {model_path}")
    
    def load_models(self, model_path: str):
        """Load trained models."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = {
            'anom': model_data['anom_models'],
            'total': model_data['total_models']
        }
        self.results = model_data.get('results', {})
        self.config = model_data.get('config', {})
        
        print(f"Loaded baseline models: {model_path}")


def create_train_val_test_masks(data_index: pd.DatetimeIndex, config: dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Create boolean masks for train/validation/test splits.
    
    Parameters:
    -----------
    data_index : pd.DatetimeIndex
        Index of the data
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    Tuple[pd.Series, pd.Series, pd.Series]
        Train, validation, test masks
    """
    timeframe = config['timeframe']
    
    train_end = pd.to_datetime(timeframe['train_end'])
    val_end = pd.to_datetime(timeframe['val_end'])
    test_end = pd.to_datetime(timeframe['test_end'])
    
    train_mask = data_index <= train_end
    val_mask = (data_index > train_end) & (data_index <= val_end)
    test_mask = (data_index > val_end) & (data_index <= test_end)
    
    return train_mask, val_mask, test_mask


def run_baseline_models(gen_series: pd.Series, anom_series: pd.Series,
                       X_local: pd.DataFrame, config: dict) -> BaselineModelSuite:
    """
    Run complete baseline model training and evaluation.
    
    Parameters:
    -----------
    gen_series : pd.Series
        Solar generation series
    anom_series : pd.Series
        Detrended anomaly series
    X_local : pd.DataFrame
        Local features
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    BaselineModelSuite
        Trained and evaluated baseline models
    """
    print("Running baseline models...")
    
    # Create train/val/test masks
    train_mask, val_mask, test_mask = create_train_val_test_masks(
        X_local.index, config
    )
    
    # Split data
    X_train = X_local[train_mask]
    X_val = X_local[val_mask]
    X_test = X_local[test_mask]
    
    y_train_total = gen_series[train_mask]
    y_val_total = gen_series[val_mask]
    y_test_total = gen_series[test_mask]
    
    y_train_anom = anom_series[train_mask]
    y_val_anom = anom_series[val_mask]
    y_test_anom = anom_series[test_mask]
    
    print(f"Training data: {len(X_train)} samples")
    print(f"Validation data: {len(X_val)} samples")
    print(f"Test data: {len(X_test)} samples")
    
    # Initialize and fit models
    baseline_suite = BaselineModelSuite(config)
    baseline_suite.fit_models(X_train, y_train_anom, y_train_total)
    
    # Evaluate on validation set
    val_results = baseline_suite.evaluate_models(
        X_val, y_val_anom, y_val_total, 
        gen_series[train_mask | val_mask],  # Historical data for persistence
        gen_series[train_mask | val_mask]
    )
    
    # Evaluate on test set
    test_results = baseline_suite.evaluate_models(
        X_test, y_test_anom, y_test_total,
        gen_series,  # All historical data
        gen_series
    )
    
    # Save models
    baseline_suite.save_models()
    
    # Print results
    print("\nBaseline Model Results:")
    print("=" * 50)
    
    for target_type in ['anom', 'total']:
        print(f"\n{target_type.upper()} TARGET:")
        print("-" * 30)
        for model_name in val_results[target_type].keys():
            val_rmse = val_results[target_type][model_name]['rmse']
            test_rmse = test_results[target_type][model_name]['rmse']
            print(f"{model_name:20s}: Val RMSE = {val_rmse:8.2f}, Test RMSE = {test_rmse:8.2f}")
    
    return baseline_suite
