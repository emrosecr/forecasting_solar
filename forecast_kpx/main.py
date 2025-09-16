"""
Main execution script for KPX Solar Energy Forecasting.
Runs the complete pipeline from data loading to model evaluation.
"""

import sys
import os
import argparse
import yaml
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path

# Add src to path
sys.path.append('src')

# Import custom modules
from io_load import load_kpx, load_kpx_multi_file, create_korea_and_extended_datasets, check_data_availability, build_boxed_climate_series
from anomalies import daily_doy_anom_detrended, create_targets
from corr_maps import create_correlation_maps, create_correlation_summary, correlate_boxed_series
from features import create_feature_sets, save_features
from models_baseline import run_baseline_models
from models_rf_grid import run_random_forest_models
from eval import print_model_comparison, save_metrics_json, create_evaluation_report
from plotting import create_all_plots, plot_model_comparison


def setup_directories(config):
    """Create output directories."""
    output_dirs = [
        'outputs/correlation_maps',
        'outputs/features/local',
        'outputs/features/extended', 
        'outputs/models',
        'outputs/metrics',
        'outputs/plots'
    ]
    
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("Output directories created")


def load_data(config):
    """Load KPX and ERA5 data."""
    print("Loading data...")
    
    # Load KPX data from multiple files
    try:
        gen_series, anom_base_series, anom_detrended_series = load_kpx_multi_file(config)
        
        if gen_series is not None:
            print(f"✓ KPX original data loaded: {len(gen_series)} days")
            print(f"  Date range: {gen_series.index.min()} to {gen_series.index.max()}")
            print(f"  Missing values: {gen_series.isna().sum()}")
        
        if anom_base_series is not None:
            print(f"✓ Base anomaly data loaded: {len(anom_base_series)} days")
        
        if anom_detrended_series is not None:
            print(f"✓ Detrended anomaly data loaded: {len(anom_detrended_series)} days")
            
    except Exception as e:
        print(f"✗ Error loading KPX data: {e}")
        print("Creating synthetic data for demonstration...")
        
        # Create synthetic data
        dates = pd.date_range('2017-01-01', '2021-12-31', freq='D')
        # Seasonal pattern with noise
        seasonal = 1000 + 500 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
        # Add some trend
        trend = 50 * np.arange(len(dates)) / len(dates)
        noise = np.random.normal(0, 100, len(dates))
        gen_series = pd.Series(seasonal + trend + noise, index=dates, name='gen_mwh')
        anom_base_series = None
        anom_detrended_series = None
        print(f"✓ Synthetic KPX data created: {len(gen_series)} days")
    
    # Load ERA5 data
    try:
        ds_korea, ds_extended = create_korea_and_extended_datasets(
            config['data']['era5_path'], config
        )
        print(f"✓ ERA5 data loaded")
        print(f"  Korea domain: {ds_korea.dims}")
        print(f"  Extended domain: {ds_extended.dims}")
        print(f"  Variables: {list(ds_korea.data_vars.keys())}")
    except FileNotFoundError:
        print(f"✗ ERA5 data not found at {config['data']['era5_path']}")
        print("Creating synthetic ERA5 data for demonstration...")
        
        # Create synthetic ERA5 data
        dates = pd.date_range('2017-01-01', '2021-12-31', freq='D')
        
        # Korea domain
        lats_kr = np.linspace(33, 39.5, 8)
        lons_kr = np.linspace(124, 132, 8)
        ssrd_kr = xr.DataArray(
            np.random.normal(20, 5, (len(dates), len(lats_kr), len(lons_kr))),
            coords={'time': dates, 'lat': lats_kr, 'lon': lons_kr},
            dims=['time', 'lat', 'lon'],
            name='ssrd_sum'
        )
        ds_korea = xr.Dataset({'ssrd_sum': ssrd_kr})
        
        # Extended domain (larger)
        lats_ext = np.linspace(-10, 60, 15)
        lons_ext = np.linspace(60, 250, 20)
        ssrd_ext = xr.DataArray(
            np.random.normal(20, 5, (len(dates), len(lats_ext), len(lons_ext))),
            coords={'time': dates, 'lat': lats_ext, 'lon': lons_ext},
            dims=['time', 'lat', 'lon'],
            name='ssrd_sum'
        )
        ds_extended = xr.Dataset({'ssrd_sum': ssrd_ext})
        
        print(f"✓ Synthetic ERA5 data created")
        print(f"  Korea domain: {ds_korea.dims}")
        print(f"  Extended domain: {ds_extended.dims}")
    
    return gen_series, anom_base_series, anom_detrended_series, ds_korea, ds_extended


def run_correlation_analysis(gen_series, ds_korea, ds_extended, anom_series, config):
    """Run correlation analysis."""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    try:
        # 1) Boxed series correlations (series-to-series)
        try:
            raw_boxed, anom_boxed = build_boxed_climate_series(ds_korea, config)
            correlate_boxed_series(anom_series, anom_boxed, output_dir=config['outputs']['correlation_maps'])
        except Exception as e:
            print(f"! Boxed series correlation step skipped: {e}")
        
        # 2) Spatial correlation maps (existing path)
        # Extract SSRD data
        ssrd_data = ds_extended['ssrd_sum']
        
        # Create correlation maps
        pearson_corr, spearman_corr = create_correlation_maps(
            anom_series, ssrd_data, config
        )
        
        # Create summary
        create_correlation_summary(pearson_corr, spearman_corr)
        
        print("✓ Correlation analysis complete")
        print("  Files saved:")
        print("    - outputs/correlation_maps/pearson_ssrd_vs_energy.png")
        print("    - outputs/correlation_maps/spearman_ssrd_vs_energy.png")
        print("    - outputs/correlation_maps/correlation_maps.nc")
        print("    - outputs/correlation_maps/summary.txt")
        
        return pearson_corr, spearman_corr
        
    except Exception as e:
        print(f"✗ Correlation analysis failed: {e}")
        return None, None


def run_baseline_training(gen_series, anom_series, ds_korea, config):
    """Train baseline models."""
    print("\n" + "="*60)
    print("BASELINE MODEL TRAINING")
    print("="*60)
    
    try:
        # Create features
        print("Creating local features...")
        train_mask = gen_series.index <= pd.to_datetime(config['timeframe']['train_end'])
        X_local, _ = create_feature_sets(gen_series, ds_korea, None, config, train_mask)
        
        # Train baseline models
        print("Training baseline models...")
        baseline_suite = run_baseline_models(gen_series, anom_series, X_local, config)
        
        # Save results
        save_metrics_json(baseline_suite.results, {}, 'outputs/metrics')
        
        print("✓ Baseline model training complete")
        print("  Models saved: outputs/models/baseline_stats.pkl")
        
        return baseline_suite, X_local
        
    except Exception as e:
        print(f"✗ Baseline training failed: {e}")
        return None, None


def run_rf_training(gen_series, anom_series, X_local, ds_korea, ds_extended, config):
    """Train Random Forest models."""
    print("\n" + "="*60)
    print("RANDOM FOREST MODEL TRAINING")
    print("="*60)
    
    try:
        # Create extended features if not already done
        if ds_extended is not None:
            print("Creating extended features...")
            train_mask = gen_series.index <= pd.to_datetime(config['timeframe']['train_end'])
            _, X_extended = create_feature_sets(gen_series, ds_korea, ds_extended, config, train_mask)
        else:
            print("Using local features for both local and extended models")
            X_extended = X_local
        
        # Train RF models
        print("Training Random Forest models...")
        rf_suite = run_random_forest_models(gen_series, anom_series, X_local, X_extended, config)
        
        # Save results
        save_metrics_json({}, rf_suite.results, 'outputs/metrics')
        
        print("✓ Random Forest training complete")
        print("  Models saved:")
        print("    - outputs/models/rf_grid_local_anom.pkl")
        print("    - outputs/models/rf_grid_local_total.pkl")
        print("    - outputs/models/rf_grid_extended_anom.pkl")
        print("    - outputs/models/rf_grid_extended_total.pkl")
        
        return rf_suite
        
    except Exception as e:
        print(f"✗ Random Forest training failed: {e}")
        return None


def create_evaluation_plots(baseline_suite, rf_suite, gen_series, anom_series, config):
    """Create evaluation plots."""
    print("\n" + "="*60)
    print("CREATING EVALUATION PLOTS")
    print("="*60)
    
    try:
        # Create all plots
        if baseline_suite is not None and rf_suite is not None:
            create_all_plots(baseline_suite, rf_suite, gen_series, anom_series, config)
            plot_model_comparison(baseline_suite.results, rf_suite.results)
        elif baseline_suite is not None:
            create_all_plots(baseline_suite, None, gen_series, anom_series, config)
        elif rf_suite is not None:
            plot_model_comparison({}, rf_suite.results)
        
        print("✓ Evaluation plots created")
        print("  Plots saved in: outputs/plots/")
        
    except Exception as e:
        print(f"✗ Plot creation failed: {e}")


def create_final_report(baseline_suite, rf_suite):
    """Create final evaluation report."""
    print("\n" + "="*60)
    print("CREATING FINAL REPORT")
    print("="*60)
    
    try:
        # Combine results
        baseline_results = baseline_suite.results if baseline_suite else {}
        rf_results = rf_suite.results if rf_suite else {}
        
        # Create evaluation report
        create_evaluation_report(baseline_results, rf_results)
        
        # Print comparison
        if baseline_results or rf_results:
            print_model_comparison(baseline_results, rf_results)
        
        print("✓ Final report created")
        print("  Report saved: outputs/metrics/evaluation_report.txt")
        
    except Exception as e:
        print(f"✗ Report creation failed: {e}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='KPX Solar Energy Forecasting Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--skip-correlations', action='store_true', help='Skip correlation analysis')
    parser.add_argument('--skip-baselines', action='store_true', help='Skip baseline training')
    parser.add_argument('--skip-rf', action='store_true', help='Skip Random Forest training')
    parser.add_argument('--plots-only', action='store_true', help='Only create plots (requires trained models)')
    
    args = parser.parse_args()
    
    print("KPX Solar Energy Forecasting Pipeline")
    print("="*60)
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Configuration loaded from {args.config}")
    except FileNotFoundError:
        print(f"✗ Configuration file not found: {args.config}")
        return 1
    
    # Setup directories
    setup_directories(config)
    
    # Load data
    gen_series, anom_base_series, anom_detrended_series, ds_korea, ds_extended = load_data(config)
    
    # Check data availability
    if ds_korea is not None:
        availability = check_data_availability(gen_series, ds_korea, config)
        print(f"Data availability: {availability}")
    
    # Use pre-calculated anomalies or calculate if not available
    print("\nProcessing anomalies...")
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
    
    # Run pipeline components
    pearson_corr = spearman_corr = None
    baseline_suite = None
    rf_suite = None
    
    if not args.plots_only:
        # Correlation analysis
        if not args.skip_correlations:
            pearson_corr, spearman_corr = run_correlation_analysis(
                gen_series, ds_korea, ds_extended, anom_series, config
            )
        
        # Baseline training
        if not args.skip_baselines:
            baseline_suite, X_local = run_baseline_training(
                gen_series, anom_series, ds_korea, config
            )
        
        # Random Forest training
        if not args.skip_rf:
            if baseline_suite is not None and 'X_local' in locals():
                rf_suite = run_rf_training(
                    gen_series, anom_series, X_local, ds_korea, ds_extended, config
                )
            else:
                # Create features if baseline wasn't run
                train_mask = gen_series.index <= pd.to_datetime(config['timeframe']['train_end'])
                X_local, _ = create_feature_sets(gen_series, ds_korea, ds_extended, config, train_mask)
                rf_suite = run_rf_training(
                    gen_series, anom_series, X_local, ds_korea, ds_extended, config
                )
    
    # Create evaluation plots
    create_evaluation_plots(baseline_suite, rf_suite, gen_series, anom_series, config)
    
    # Create final report
    create_final_report(baseline_suite, rf_suite)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("All outputs saved in the 'outputs/' directory:")
    print("  - outputs/correlation_maps/  : Correlation maps and analysis")
    print("  - outputs/features/          : Feature engineering outputs")
    print("  - outputs/models/            : Trained models")
    print("  - outputs/metrics/           : Evaluation metrics and reports")
    print("  - outputs/plots/             : Visualization plots")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
