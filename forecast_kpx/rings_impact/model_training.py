"""
Model training with Optuna tuning and SHAP analysis for concentric ring features.
Supports RandomForestRegressor and LightGBM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


@dataclass
class TrainConfig:
    model_type: str = "rf"  # "rf" or "lgbm"
    n_splits: int = 5
    random_state: int = 42
    n_trials: int = 30


def _objective_rf(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, n_splits: int, random_state: int) -> float:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
        'random_state': random_state,
        'n_jobs': -1
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses = []
    for train_idx, val_idx in tscv.split(X):
        model = RandomForestRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[val_idx])
        rmse = mean_squared_error(y.iloc[val_idx], pred, squared=False)
        rmses.append(rmse)
    return float(np.mean(rmses))


def _objective_lgbm(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, n_splits: int, random_state: int) -> float:
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'goss']),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 7),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'seed': random_state,
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses = []
    for train_idx, val_idx in tscv.split(X):
        lgb_train = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
        lgb_val = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], reference=lgb_train)
        gbm = lgb.train(params, lgb_train, valid_sets=[lgb_val], num_boost_round=200, verbose_eval=False)
        pred = gbm.predict(X.iloc[val_idx])
        rmse = mean_squared_error(y.iloc[val_idx], pred, squared=False)
        rmses.append(rmse)
    return float(np.mean(rmses))


def tune_and_train(X: pd.DataFrame, y: pd.Series, cfg: TrainConfig) -> Tuple[object, Dict]:
    """
    Optuna tune, then train final model on all data. Returns model and best params.
    """
    def objective(trial: optuna.Trial) -> float:
        if cfg.model_type == 'rf':
            return _objective_rf(trial, X, y, cfg.n_splits, cfg.random_state)
        elif cfg.model_type == 'lgbm' and HAS_LGB:
            return _objective_lgbm(trial, X, y, cfg.n_splits, cfg.random_state)
        else:
            raise ValueError("Invalid model_type or LightGBM not installed")

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=cfg.n_trials)
    best_params = study.best_params

    # Train final model
    if cfg.model_type == 'rf':
        final_params = {
            'n_estimators': best_params.get('n_estimators', 500),
            'max_features': best_params.get('max_features', 'sqrt'),
            'min_samples_split': best_params.get('min_samples_split', 2),
            'min_samples_leaf': best_params.get('min_samples_leaf', 1),
            'n_jobs': -1,
            'random_state': cfg.random_state
        }
        model = RandomForestRegressor(**final_params)
        model.fit(X, y)
    else:
        params = best_params.copy()
        params.update({'objective': 'regression', 'metric': 'rmse', 'verbosity': -1, 'seed': cfg.random_state})
        dtrain = lgb.Dataset(X, label=y)
        model = lgb.train(params, dtrain, num_boost_round=300)

    return model, best_params


def compute_shap_values(model: object, X: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute SHAP values for model and dataset X. Returns feature importance summary
    (mean |SHAP| per feature) and shap values array.
    """
    if not HAS_SHAP:
        raise RuntimeError("SHAP not installed")

    if hasattr(model, 'predict_proba'):
        raise ValueError("Regression model expected")

    if 'lightgbm' in type(model).__module__:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

    # shap_values shape: (n_samples, n_features)
    importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({'feature': X.columns, 'mean_abs_shap': importance}).sort_values('mean_abs_shap', ascending=False)
    return importance_df, shap_values


def summarize_distance_of_impact(importance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize ring-level importance by collapsing variables per ring index and ranking rings.
    Assumes feature names include suffix _ringN.
    """
    ring_scores: Dict[int, float] = {}
    for _, row in importance_df.iterrows():
        name = str(row['feature'])
        score = float(row['mean_abs_shap'])
        if '_ring' in name:
            try:
                ring_idx = int(name.split('_ring')[-1])
                ring_scores[ring_idx] = ring_scores.get(ring_idx, 0.0) + score
            except Exception:
                continue
    out = pd.DataFrame({'ring': list(ring_scores.keys()), 'total_importance': list(ring_scores.values())})
    out = out.sort_values('total_importance', ascending=False).reset_index(drop=True)
    return out

