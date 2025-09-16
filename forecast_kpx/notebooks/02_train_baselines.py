"""
Notebook 02: Train Baseline Models
Convert this to Jupyter notebook format for interactive analysis.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# Add src to path
sys.path.append('../src')

# Import custom modules
from io_load import load_kpx, create_korea_and_extended_datasets
from anomalies import daily_doy_anom_detrended, create_targets
from features import create_feature_sets, save_features
from models_baseline import run_baseline_models
from eval import print_model_comparison, save_metrics_json
from plotting import create_all_plots

print("=== KPX Solar Energy Forecasting - Baseline Models ===")

# Load configuration
with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"Configuration loaded: {config['timeframe']['start']} to {config['timeframe']['end']}")

# Load data
print("\n1. Loading data...")
try:
    gen_series = load_kpx(config['data']['kpx_path'])
    ds_korea, ds_extended = create_korea_and_extended_datasets(config['data']['era5_path'], config)
    print(f"Data loaded: {len(gen_series)} days, {ds_korea.dims}")
except FileNotFoundError as e:
    print(f"Data files not found: {e}")
    print("Creating synthetic data for demonstration...")
    
    # Synthetic data
    dates = pd.date_range('2017-01-01', '2021-12-31', freq='D')
    seasonal = 1000 + 500 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    noise = np.random.normal(0, 100, len(dates))
    gen_series = pd.Series(seasonal + noise, index=dates, name='gen_mwh')
    
    # Synthetic ERA5 data
    lats_kr = np.linspace(33, 39.5, 5)
    lons_kr = np.linspace(124, 132, 5)
    ssrd_kr = xr.DataArray(
        np.random.normal(20, 5, (len(dates), len(lats_kr), len(lons_kr))),
        coords={'time': dates, 'lat': lats_kr, 'lon': lons_kr},
        dims=['time', 'lat', 'lon'],
        name='ssrd_sum'
    )
    ds_korea = xr.Dataset({'ssrd_sum': ssrd_kr})
    ds_extended = ds_korea

# Calculate anomalies
print("\n2. Calculating anomalies...")
bad_window = (config['timeframe']['bad_window']['start'], config['timeframe']['bad_window']['end'])
anom_df = daily_doy_anom_detrended(gen_series, exclude=bad_window)
anom_series = anom_df['anom_detrended']

# Create targets
y_total, y_anom = create_targets(gen_series, anom_series, config)

# Create feature sets
print("\n3. Creating features...")
train_mask = gen_series.index <= pd.to_datetime(config['timeframe']['train_end'])
X_local, X_extended = create_feature_sets(gen_series, ds_korea, ds_extended, config, train_mask)

# Save features
save_features(X_local, X_extended)

# Train baseline models
print("\n4. Training baseline models...")
baseline_suite = run_baseline_models(gen_series, anom_series, X_local, config)

# Save metrics
print("\n5. Saving results...")
save_metrics_json(baseline_suite.results, {}, '../outputs/metrics')

# Create plots
print("\n6. Creating plots...")
try:
    create_all_plots(baseline_suite, None, gen_series, anom_series, config)
except Exception as e:
    print(f"Plotting failed: {e}")

print("\nBaseline model training complete!")
print("Results saved in outputs/ directory")
