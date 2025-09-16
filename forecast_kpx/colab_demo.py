"""
Google Colab Demo Script for KPX Solar Energy Forecasting
This script demonstrates the complete pipeline with synthetic data.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

# Add src to path
sys.path.append('src')

# Import custom modules
from io_load import load_kpx, load_kpx_multi_file, create_korea_and_extended_datasets
from anomalies import daily_doy_anom_detrended, create_targets
from corr_maps import create_correlation_maps
from features import create_feature_sets, save_features
from models_baseline import run_baseline_models
from models_rf_grid import run_random_forest_models
from eval import print_model_comparison, save_metrics_json
from plotting import plot_model_comparison

def create_synthetic_data():
    """Create synthetic data for demonstration."""
    print("Creating synthetic data for demonstration...")
    
    # Create synthetic KPX data
    dates = pd.date_range('2017-01-01', '2021-12-31', freq='D')
    
    # Seasonal pattern with trend and noise
    seasonal = 1000 + 500 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    trend = 50 * np.arange(len(dates)) / len(dates)
    noise = np.random.normal(0, 100, len(dates))
    
    # Add some autocorrelation
    for i in range(1, len(dates)):
        noise[i] += 0.3 * noise[i-1]
    
    gen_series = pd.Series(seasonal + trend + noise, index=dates, name='gen_mwh')
    
    # Create synthetic ERA5 data
    # Korea domain
    lats_kr = np.linspace(33, 39.5, 8)
    lons_kr = np.linspace(124, 132, 8)
    
    # Extended domain
    lats_ext = np.linspace(-10, 60, 12)
    lons_ext = np.linspace(60, 250, 16)
    
    # Create SSRD data with correlation to generation
    ssrd_kr = np.random.normal(20, 5, (len(dates), len(lats_kr), len(lons_kr)))
    ssrd_ext = np.random.normal(20, 5, (len(dates), len(lats_ext), len(lons_ext)))
    
    # Add correlation with generation
    gen_norm = (gen_series - gen_series.mean()) / gen_series.std()
    
    for i in range(len(lats_kr)):
        for j in range(len(lons_kr)):
            correlation = 0.2 + 0.3 * np.random.random()
            ssrd_kr[:, i, j] += correlation * gen_norm.values
    
    for i in range(len(lats_ext)):
        for j in range(len(lons_ext)):
            correlation = 0.1 + 0.2 * np.random.random()
            ssrd_ext[:, i, j] += correlation * gen_norm.values
    
    # Create xarray datasets
    ds_korea = xr.Dataset({
        'ssrd_sum': (['time', 'lat', 'lon'], ssrd_kr)
    }, coords={
        'time': dates,
        'lat': lats_kr,
        'lon': lons_kr
    })
    
    ds_extended = xr.Dataset({
        'ssrd_sum': (['time', 'lat', 'lon'], ssrd_ext)
    }, coords={
        'time': dates,
        'lat': lats_ext,
        'lon': lons_ext
    })
    
    print(f"✓ Synthetic data created:")
    print(f"  KPX: {len(gen_series)} days")
    print(f"  Korea domain: {ds_korea.dims}")
    print(f"  Extended domain: {ds_extended.dims}")
    
    return gen_series, ds_korea, ds_extended

def run_demo():
    """Run the complete demo pipeline."""
    print("KPX Solar Energy Forecasting - Google Colab Demo")
    print("="*60)
    
    # Create output directories
    os.makedirs('outputs/correlation_maps', exist_ok=True)
    os.makedirs('outputs/features/local', exist_ok=True)
    os.makedirs('outputs/features/extended', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    
    # Configuration
    config = {
        'timeframe': {
            'start': '2017-01-01',
            'end': '2021-12-31',
            'train_end': '2019-12-31',
            'val_end': '2020-12-31',
            'test_end': '2021-12-31',
            'bad_window': {
                'start': '2020-03-15',
                'end': '2020-04-15'
            }
        },
        'models': {
            'baseline': {
                'linear_features': ['gen_lag1', 'gen_lag2', 'gen_lag3', 'gen_lag7', 
                                  'ssrd_kr_lag1', 'tcc_kr_lag1', 'sinDoy', 'cosDoy']
            },
            'random_forest': {
                'n_estimators': 100,  # Reduced for demo
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1,
                'pca_components': 5,
                'grid_downsample': 2
            }
        },
        'features': {
            'lags': {
                'energy': [1, 2, 3, 7],
                'meteo': [1, 2, 3]
            },
            'calendar': {
                'include_dow': True,
                'include_seasonal': True
            }
        },
        'memory': {
            'max_memory_gb': 4,
            'use_pca_extended': True,
            'grid_downsample_extended': True
        }
    }
    
    # Create synthetic data
    gen_series, ds_korea, ds_extended = create_synthetic_data()
    anom_base_series = None
    anom_detrended_series = None
    
    # Use pre-calculated anomalies or calculate if not available
    print("\n1. Processing anomalies...")
    if anom_detrended_series is not None:
        print("Using pre-calculated detrended anomalies")
        anom_series = anom_detrended_series
    else:
        print("Calculating detrended anomalies from original data...")
        bad_window = (config['timeframe']['bad_window']['start'], 
                      config['timeframe']['bad_window']['end'])
        anom_df = daily_doy_anom_detrended(gen_series, exclude=bad_window)
        anom_series = anom_df['anom_detrended']
    
    # Create targets
    y_total, y_anom = create_targets(gen_series, anom_series, config)
    print(f"✓ Targets created: {len(y_total)} samples")
    
    # Correlation analysis
    print("\n2. Correlation analysis...")
    try:
        ssrd_data = ds_extended['ssrd_sum']
        pearson_corr, spearman_corr = create_correlation_maps(
            anom_series, ssrd_data, config
        )
        print(f"✓ Correlation maps created")
        print(f"  Pearson range: {pearson_corr.min().values:.3f} to {pearson_corr.max().values:.3f}")
    except Exception as e:
        print(f"✗ Correlation analysis failed: {e}")
    
    # Create features
    print("\n3. Creating features...")
    train_mask = gen_series.index <= pd.to_datetime(config['timeframe']['train_end'])
    X_local, X_extended = create_feature_sets(gen_series, ds_korea, ds_extended, config, train_mask)
    save_features(X_local, X_extended)
    print(f"✓ Features created: Local {X_local.shape}, Extended {X_extended.shape}")
    
    # Train baseline models
    print("\n4. Training baseline models...")
    try:
        baseline_suite = run_baseline_models(gen_series, anom_series, X_local, config)
        print("✓ Baseline models trained")
    except Exception as e:
        print(f"✗ Baseline training failed: {e}")
        baseline_suite = None
    
    # Train Random Forest models
    print("\n5. Training Random Forest models...")
    try:
        rf_suite = run_random_forest_models(gen_series, anom_series, X_local, X_extended, config)
        print("✓ Random Forest models trained")
    except Exception as e:
        print(f"✗ Random Forest training failed: {e}")
        rf_suite = None
    
    # Create comparison plots
    print("\n6. Creating evaluation plots...")
    try:
        baseline_results = baseline_suite.results if baseline_suite else {}
        rf_results = rf_suite.results if rf_suite else {}
        plot_model_comparison(baseline_results, rf_results)
        print("✓ Evaluation plots created")
    except Exception as e:
        print(f"✗ Plot creation failed: {e}")
    
    # Print results
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    
    if baseline_suite and rf_suite:
        print_model_comparison(baseline_suite.results, rf_suite.results)
    
    print("\nOutputs created:")
    print("  - outputs/correlation_maps/  : Correlation analysis")
    print("  - outputs/features/          : Feature engineering")
    print("  - outputs/models/            : Trained models")
    print("  - outputs/metrics/           : Evaluation metrics")
    print("  - outputs/plots/             : Visualization plots")
    
    # Display some key results
    print("\nKey Results:")
    if baseline_suite:
        print("\nBaseline Models (Validation RMSE):")
        for target_type in ['anom', 'total']:
            print(f"  {target_type.upper()}:")
            for model_name, metrics in baseline_suite.results[target_type].items():
                rmse = metrics.get('rmse', np.nan)
                print(f"    {model_name}: {rmse:.2f}")
    
    if rf_suite:
        print("\nRandom Forest Models (Validation RMSE):")
        for target_type in ['anom', 'total']:
            print(f"  {target_type.upper()}:")
            for feature_type, metrics in rf_suite.results[target_type].items():
                rmse = metrics.get('rmse', np.nan)
                print(f"    {feature_type}: {rmse:.2f}")
    
    return baseline_suite, rf_suite

if __name__ == "__main__":
    baseline_suite, rf_suite = run_demo()
