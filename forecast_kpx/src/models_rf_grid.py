"""
Random Forest models for gridded solar energy forecasting.
Implements local and extended feature Random Forest models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import pickle
import os
import warnings


class RandomForestModel:
    """Random Forest model wrapper with preprocessing."""
    
    def __init__(self, config: dict, feature_type: str, target_type: str):
        self.config = config
        self.feature_type = feature_type
        self.target_type = target_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
        # Model parameters
        rf_config = config['models']['random_forest']
        self.n_estimators = rf_config.get('n_estimators', 500)
        self.max_features = rf_config.get('max_features', 'sqrt')
        self.random_state = rf_config.get('random_state', 42)
        self.n_jobs = rf_config.get('n_jobs', -1)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit Random Forest model with preprocessing."""
        print(f"Fitting RF {self.feature_type} {self.target_type} model...")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Remove rows with missing values
        valid_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
        X_clean = X_train[valid_mask]
        y_clean = y_train[valid_mask]
        
        if len(X_clean) == 0:
            warnings.warn(f"No valid training data for RF {self.feature_type} {self.target_type}")
            return
        
        print(f"  Training samples: {len(X_clean)}")
        print(f"  Features: {len(self.feature_names)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Initialize and fit Random Forest
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        self.model.fit(X_scaled, y_clean)
        self.is_fitted = True
        
        print(f"  Feature importance (top 10):")
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"    {i+1:2d}. {row['feature']:25s}: {row['importance']:.4f}")
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict using fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Ensure same features as training
        X_aligned = X[self.feature_names].copy()
        
        # Handle missing values
        for col in X_aligned.columns:
            if X_aligned[col].isna().any():
                # Use median from training data
                median_val = self.scaler.data_min_[X_aligned.columns.get_loc(col)]
                X_aligned[col] = X_aligned[col].fillna(median_val)
        
        # Scale features
        X_scaled = self.scaler.transform(X_aligned)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        return pd.Series(predictions, index=X.index)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    def save_model(self, output_dir: str):
        """Save trained model."""
        os.makedirs(output_dir, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config,
            'feature_type': self.feature_type,
            'target_type': self.target_type
        }
        
        model_path = os.path.join(output_dir, f"rf_grid_{self.feature_type}_{self.target_type}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Saved RF model: {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_type = model_data['feature_type']
        self.target_type = model_data['target_type']
        self.is_fitted = True
        
        print(f"Loaded RF model: {model_path}")


class RandomForestSuite:
    """Suite of Random Forest models for local and extended features."""
    
    def __init__(self, config: dict):
        self.config = config
        self.models = {}
        self.results = {}
    
    def fit_models(self, X_local_train: pd.DataFrame, X_extended_train: pd.DataFrame,
                  y_train_anom: pd.Series, y_train_total: pd.Series):
        """Fit Random Forest models for both feature sets and targets."""
        print("Fitting Random Forest models...")
        
        # Local models
        print("\n=== LOCAL FEATURES ===")
        self.models['local'] = {}
        
        # Local anomaly model
        self.models['local']['anom'] = RandomForestModel(
            self.config, 'local', 'anom'
        )
        self.models['local']['anom'].fit(X_local_train, y_train_anom)
        
        # Local total model
        self.models['local']['total'] = RandomForestModel(
            self.config, 'local', 'total'
        )
        self.models['local']['total'].fit(X_local_train, y_train_total)
        
        # Extended models
        print("\n=== EXTENDED FEATURES ===")
        self.models['extended'] = {}
        
        # Extended anomaly model
        self.models['extended']['anom'] = RandomForestModel(
            self.config, 'extended', 'anom'
        )
        self.models['extended']['anom'].fit(X_extended_train, y_train_anom)
        
        # Extended total model
        self.models['extended']['total'] = RandomForestModel(
            self.config, 'extended', 'total'
        )
        self.models['extended']['total'].fit(X_extended_train, y_train_total)
    
    def predict(self, X_local: pd.DataFrame, X_extended: pd.DataFrame,
                target_type: str) -> Dict[str, pd.Series]:
        """Generate predictions for both feature sets."""
        predictions = {}
        
        # Local predictions
        predictions['local'] = self.models['local'][target_type].predict(X_local)
        
        # Extended predictions
        predictions['extended'] = self.models['extended'][target_type].predict(X_extended)
        
        return predictions
    
    def evaluate_models(self, X_local_val: pd.DataFrame, X_extended_val: pd.DataFrame,
                       y_val_anom: pd.Series, y_val_total: pd.Series) -> Dict[str, Dict[str, Dict]]:
        """Evaluate all Random Forest models."""
        print("Evaluating Random Forest models...")
        
        results = {}
        
        # Evaluate anomaly models
        print("\n=== ANOMALY MODELS ===")
        results['anom'] = {}
        
        for feature_type in ['local', 'extended']:
            print(f"\n{feature_type.upper()} FEATURES:")
            model = self.models[feature_type]['anom']
            pred = model.predict(X_local_val if feature_type == 'local' else X_extended_val)
            
            # Align indices and remove NaN
            common_idx = pred.index.intersection(y_val_anom.index)
            pred_aligned = pred.loc[common_idx]
            y_true = y_val_anom.loc[common_idx]
            
            valid_mask = ~(pred_aligned.isna() | y_true.isna())
            if valid_mask.sum() > 0:
                pred_clean = pred_aligned[valid_mask]
                y_clean = y_true[valid_mask]
                
                metrics = self._calculate_metrics(y_clean, pred_clean, 'anom')
                results['anom'][feature_type] = metrics
                
                print(f"  RMSE: {metrics['rmse']:.2f}")
                print(f"  MAE:  {metrics['mae']:.2f}")
                print(f"  R²:   {metrics['r2']:.3f}")
            else:
                results['anom'][feature_type] = {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}
        
        # Evaluate total models
        print("\n=== TOTAL MODELS ===")
        results['total'] = {}
        
        for feature_type in ['local', 'extended']:
            print(f"\n{feature_type.upper()} FEATURES:")
            model = self.models[feature_type]['total']
            pred = model.predict(X_local_val if feature_type == 'local' else X_extended_val)
            
            # Align indices and remove NaN
            common_idx = pred.index.intersection(y_val_total.index)
            pred_aligned = pred.loc[common_idx]
            y_true = y_val_total.loc[common_idx]
            
            valid_mask = ~(pred_aligned.isna() | y_true.isna())
            if valid_mask.sum() > 0:
                pred_clean = pred_aligned[valid_mask]
                y_clean = y_true[valid_mask]
                
                metrics = self._calculate_metrics(y_clean, pred_clean, 'total')
                results['total'][feature_type] = metrics
                
                print(f"  RMSE: {metrics['rmse']:.2f}")
                print(f"  MAE:  {metrics['mae']:.2f}")
                print(f"  R²:   {metrics['r2']:.3f}")
                if 'mape' in metrics:
                    print(f"  MAPE: {metrics['mape']:.1f}%")
            else:
                results['total'][feature_type] = {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}
        
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
        """Save all trained models."""
        os.makedirs(output_dir, exist_ok=True)
        
        for feature_type in ['local', 'extended']:
            for target_type in ['anom', 'total']:
                self.models[feature_type][target_type].save_model(output_dir)
        
        # Save results summary
        results_path = os.path.join(output_dir, "rf_results_summary.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"Saved RF results summary: {results_path}")
    
    def get_feature_importance_summary(self) -> Dict[str, pd.DataFrame]:
        """Get feature importance for all models."""
        importance_summary = {}
        
        for feature_type in ['local', 'extended']:
            importance_summary[feature_type] = {}
            for target_type in ['anom', 'total']:
                importance_summary[feature_type][target_type] = \
                    self.models[feature_type][target_type].get_feature_importance()
        
        return importance_summary
    
    def save_feature_importance(self, output_dir: str = "outputs/models"):
        """Save feature importance analysis."""
        importance_summary = self.get_feature_importance_summary()
        
        for feature_type in ['local', 'extended']:
            for target_type in ['anom', 'total']:
                importance_df = importance_summary[feature_type][target_type]
                
                # Save as CSV
                csv_path = os.path.join(output_dir, f"feature_importance_{feature_type}_{target_type}.csv")
                importance_df.to_csv(csv_path, index=False)
                
                print(f"Saved feature importance: {csv_path}")


def run_random_forest_models(gen_series: pd.Series, anom_series: pd.Series,
                           X_local: pd.DataFrame, X_extended: pd.DataFrame,
                           config: dict) -> RandomForestSuite:
    """
    Run complete Random Forest model training and evaluation.
    
    Parameters:
    -----------
    gen_series : pd.Series
        Solar generation series
    anom_series : pd.Series
        Detrended anomaly series
    X_local : pd.DataFrame
        Local features
    X_extended : pd.DataFrame
        Extended features
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    RandomForestSuite
        Trained and evaluated Random Forest models
    """
    print("Running Random Forest models...")
    
    # Create train/val/test masks
    train_mask, val_mask, test_mask = create_train_val_test_masks(
        X_local.index, config
    )
    
    # Split local features
    X_local_train = X_local[train_mask]
    X_local_val = X_local[val_mask]
    X_local_test = X_local[test_mask]
    
    # Split extended features
    X_extended_train = X_extended[train_mask]
    X_extended_val = X_extended[val_mask]
    X_extended_test = X_extended[test_mask]
    
    # Split targets
    y_train_total = gen_series[train_mask]
    y_val_total = gen_series[val_mask]
    y_test_total = gen_series[test_mask]
    
    y_train_anom = anom_series[train_mask]
    y_val_anom = anom_series[val_mask]
    y_test_anom = anom_series[test_mask]
    
    print(f"Training data: {len(X_local_train)} samples")
    print(f"Validation data: {len(X_local_val)} samples")
    print(f"Test data: {len(X_local_test)} samples")
    print(f"Local features: {X_local_train.shape[1]}")
    print(f"Extended features: {X_extended_train.shape[1]}")
    
    # Initialize and fit models
    rf_suite = RandomForestSuite(config)
    rf_suite.fit_models(X_local_train, X_extended_train, y_train_anom, y_train_total)
    
    # Evaluate on validation set
    print("\n=== VALIDATION RESULTS ===")
    val_results = rf_suite.evaluate_models(X_local_val, X_extended_val, y_val_anom, y_val_total)
    
    # Evaluate on test set
    print("\n=== TEST RESULTS ===")
    test_results = rf_suite.evaluate_models(X_local_test, X_extended_test, y_test_anom, y_test_total)
    
    # Save models and results
    rf_suite.save_models()
    rf_suite.save_feature_importance()
    
    # Print comparison table
    print("\n" + "="*80)
    print("RANDOM FOREST MODEL COMPARISON")
    print("="*80)
    
    for target_type in ['anom', 'total']:
        print(f"\n{target_type.upper()} TARGET:")
        print("-" * 50)
        print(f"{'Model':<20} {'Val RMSE':<12} {'Test RMSE':<12} {'Val R²':<10} {'Test R²':<10}")
        print("-" * 50)
        
        for feature_type in ['local', 'extended']:
            val_rmse = val_results[target_type][feature_type]['rmse']
            test_rmse = test_results[target_type][feature_type]['rmse']
            val_r2 = val_results[target_type][feature_type]['r2']
            test_r2 = test_results[target_type][feature_type]['r2']
            
            print(f"RF {feature_type:<15} {val_rmse:<12.2f} {test_rmse:<12.2f} {val_r2:<10.3f} {test_r2:<10.3f}")
    
    return rf_suite


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
