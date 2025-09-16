"""
Example usage script for KPX Solar Energy Forecasting with your data files.
This shows how to use the updated code with your three CSV files.
"""

import sys
import os
import pandas as pd
import numpy as np
import yaml

# Add src to path
sys.path.append('src')

# Import custom modules
from io_load import load_kpx_multi_file
from anomalies import create_targets
from corr_maps import create_correlation_maps
from features import create_feature_sets, save_features
from models_baseline import run_baseline_models
from models_rf_grid import run_random_forest_models
from eval import print_model_comparison, save_metrics_json

def example_with_your_data():
    """Example using your specific data files."""
    print("KPX Solar Energy Forecasting - Example with Your Data")
    print("="*60)
    
    # Configuration for your data files
    config = {
        'data': {
            'kpx_original': 'data/solarenergy2.csv',
            'kpx_anomaly_base': 'data/solarenergy2_daily_anomaly_base_only.csv',
            'kpx_anomaly_detrended': 'data/solarenergy2_daily_anomaly_detrended.csv',
            'era5_path': 'data/era5_*.nc'  # Update this path if you have ERA5 data
        },
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
    
    # Create output directories
    os.makedirs('outputs/correlation_maps', exist_ok=True)
    os.makedirs('outputs/features/local', exist_ok=True)
    os.makedirs('outputs/features/extended', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    
    # Step 1: Load your data files
    print("\n1. Loading your data files...")
    try:
        gen_series, anom_base_series, anom_detrended_series = load_kpx_multi_file(config)
        
        if gen_series is not None:
            print(f"✓ Original data: {len(gen_series)} days")
            print(f"  Date range: {gen_series.index.min()} to {gen_series.index.max()}")
        
        if anom_base_series is not None:
            print(f"✓ Base anomaly data: {len(anom_base_series)} days")
        
        if anom_detrended_series is not None:
            print(f"✓ Detrended anomaly data: {len(anom_detrended_series)} days")
            
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("Please check that your CSV files are in the data/ directory")
        return
    
    # Step 2: Use your pre-calculated anomalies
    print("\n2. Using your pre-calculated anomalies...")
    if anom_detrended_series is not None:
        anom_series = anom_detrended_series
        print("✓ Using your detrended anomaly data")
    else:
        print("✗ Detrended anomaly data not found")
        return
    
    # Step 3: Create targets
    print("\n3. Creating forecasting targets...")
    y_total, y_anom = create_targets(gen_series, anom_series, config)
    print(f"✓ Targets created: {len(y_total)} samples")
    
    # Step 4: Create synthetic ERA5 data (since you might not have it yet)
    print("\n4. Creating synthetic ERA5 data for demonstration...")
    dates = pd.date_range('2017-01-01', '2021-12-31', freq='D')
    
    # Korea domain
    lats_kr = np.linspace(33, 39.5, 8)
    lons_kr = np.linspace(124, 132, 8)
    
    # Create SSRD data with some correlation to generation
    ssrd_kr = np.random.normal(20, 5, (len(dates), len(lats_kr), len(lons_kr)))
    
    # Add correlation with generation
    gen_norm = (gen_series - gen_series.mean()) / gen_series.std()
    for i in range(len(lats_kr)):
        for j in range(len(lons_kr)):
            correlation = 0.2 + 0.3 * np.random.random()
            ssrd_kr[:, i, j] += correlation * gen_norm.values
    
    # Create xarray dataset
    import xarray as xr
    ds_korea = xr.Dataset({
        'ssrd_sum': (['time', 'lat', 'lon'], ssrd_kr)
    }, coords={
        'time': dates,
        'lat': lats_kr,
        'lon': lons_kr
    })
    
    ds_extended = ds_korea  # Use same for demo
    print(f"✓ Synthetic ERA5 data created: {ds_korea.dims}")
    
    # Step 5: Create features
    print("\n5. Creating features...")
    train_mask = gen_series.index <= pd.to_datetime(config['timeframe']['train_end'])
    X_local, X_extended = create_feature_sets(gen_series, ds_korea, ds_extended, config, train_mask)
    save_features(X_local, X_extended)
    print(f"✓ Features created: Local {X_local.shape}, Extended {X_extended.shape}")
    
    # Step 6: Train baseline models
    print("\n6. Training baseline models...")
    try:
        baseline_suite = run_baseline_models(gen_series, anom_series, X_local, config)
        print("✓ Baseline models trained")
    except Exception as e:
        print(f"✗ Baseline training failed: {e}")
        baseline_suite = None
    
    # Step 7: Train Random Forest models
    print("\n7. Training Random Forest models...")
    try:
        rf_suite = run_random_forest_models(gen_series, anom_series, X_local, X_extended, config)
        print("✓ Random Forest models trained")
    except Exception as e:
        print(f"✗ Random Forest training failed: {e}")
        rf_suite = None
    
    # Step 8: Save results
    print("\n8. Saving results...")
    try:
        baseline_results = baseline_suite.results if baseline_suite else {}
        rf_results = rf_suite.results if rf_suite else {}
        save_metrics_json(baseline_results, rf_results, 'outputs/metrics')
        print("✓ Results saved")
    except Exception as e:
        print(f"✗ Error saving results: {e}")
    
    # Step 9: Print comparison
    print("\n9. Model comparison:")
    if baseline_suite and rf_suite:
        print_model_comparison(baseline_suite.results, rf_suite.results)
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETE")
    print("="*60)
    print("Your data has been processed successfully!")
    print("\nOutputs created:")
    print("  - outputs/features/          : Feature engineering")
    print("  - outputs/models/            : Trained models")
    print("  - outputs/metrics/           : Evaluation metrics")
    print("  - outputs/plots/             : Visualization plots")
    
    print("\nTo use with real ERA5 data:")
    print("1. Download ERA5 data for your domain")
    print("2. Update the era5_path in config.yaml")
    print("3. Run the full pipeline with: python main.py")

if __name__ == "__main__":
    example_with_your_data()
