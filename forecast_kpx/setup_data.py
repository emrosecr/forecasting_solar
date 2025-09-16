"""
Data setup script for KPX Solar Energy Forecasting.
This script helps you organize your data files and verify their format.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def check_csv_format(file_path, file_type="data"):
    """
    Check the format of a CSV file and display information about it.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    file_type : str
        Type of file for display purposes
    """
    print(f"\nChecking {file_type} file: {file_path}")
    print("-" * 50)
    
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        return False
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        print(f"✓ File loaded successfully")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Data types:")
        for col, dtype in df.dtypes.items():
            print(f"    {col}: {dtype}")
        
        # Check for date column
        date_cols = ['date', 'Date', 'DATE', 'time', 'Time', 'TIME', 'datetime']
        date_col = None
        for col in date_cols:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            print(f"  Date column found: {date_col}")
            # Convert to datetime to check format
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                print(f"  Date range: {df[date_col].min()} to {df[date_col].max()}")
            except:
                print(f"  ⚠ Warning: Could not parse dates in {date_col}")
        else:
            print(f"  ⚠ Warning: No standard date column found")
        
        # Check for value column
        value_cols = ['gen_mwh', 'generation_mwh', 'solar_mwh', 'power_mwh', 'energy_mwh', 'value', 'Value', 'VALUE']
        value_col = None
        for col in value_cols:
            if col in df.columns:
                value_col = col
                break
        
        if value_col:
            print(f"  Value column found: {value_col}")
            print(f"  Value range: {df[value_col].min():.2f} to {df[value_col].max():.2f}")
            print(f"  Missing values: {df[value_col].isna().sum()}")
        else:
            print(f"  ⚠ Warning: No standard value column found")
            # Show numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(f"  Numeric columns available: {list(numeric_cols)}")
        
        # Show first few rows
        print(f"  First 3 rows:")
        print(df.head(3).to_string())
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False

def setup_data_directory():
    """Set up the data directory structure."""
    print("Setting up data directory structure...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print(f"✓ Created data directory: {data_dir}")
    
    # Create subdirectories
    subdirs = ["raw", "processed", "external"]
    for subdir in subdirs:
        subdir_path = data_dir / subdir
        subdir_path.mkdir(exist_ok=True)
        print(f"✓ Created subdirectory: {subdir_path}")
    
    return data_dir

def create_sample_config():
    """Create a sample configuration file with your data paths."""
    config_content = """# Configuration for KPX Solar Energy Forecasting
data:
  # KPX solar energy data files - UPDATE THESE PATHS
  kpx_original: "data/solarenergy2.csv"  # Original solar generation data
  kpx_anomaly_base: "data/solarenergy2_daily_anomaly_base_only.csv"  # Base anomaly data
  kpx_anomaly_detrended: "data/solarenergy2_daily_anomaly_detrended.csv"  # Detrended anomaly data
  era5_path: "data/era5_*.nc"  # Path pattern for ERA5 NetCDF files
  
  # Geographic bounds
  extended_bbox:
    lat_min: -10
    lat_max: 60
    lon_min: 60
    lon_max: 250  # 110W in 0-360 format
    
  korea_bbox:
    lat_min: 33
    lat_max: 39.5
    lon_min: 124
    lon_max: 132

# Time periods
timeframe:
  start: "2017-01-01"
  end: "2021-12-31"
  train_end: "2019-12-31"
  val_end: "2020-12-31"
  test_end: "2021-12-31"
  
  # Known bad data window to exclude from baseline fitting
  bad_window:
    start: "2020-03-15"
    end: "2020-04-15"

# Model parameters
models:
  baseline:
    linear_features: ["gen_lag1", "gen_lag2", "gen_lag3", "gen_lag7", 
                     "ssrd_kr_lag1", "tcc_kr_lag1", "sinDoy", "cosDoy"]
    
  random_forest:
    n_estimators: 500
    max_features: "sqrt"
    random_state: 42
    n_jobs: -1
    
    # For extended grid mode
    pca_components: 10  # Per variable
    grid_downsample: 2  # Every Nth grid point

# Feature engineering
features:
  lags:
    energy: [1, 2, 3, 7]  # Days
    meteo: [1, 2, 3]     # Days
    
  calendar:
    include_dow: true
    include_seasonal: true

# Output paths
outputs:
  correlation_maps: "outputs/correlation_maps"
  features: "outputs/features"
  models: "outputs/models"
  metrics: "outputs/metrics"
  plots: "outputs/plots"

# Memory management (for Google Colab)
memory:
  max_memory_gb: 8
  use_pca_extended: true
  grid_downsample_extended: true
"""
    
    with open("config_sample.yaml", "w") as f:
        f.write(config_content)
    
    print("✓ Created sample configuration file: config_sample.yaml")
    print("  Please copy this to config.yaml and update the data paths")

def main():
    """Main setup function."""
    print("KPX Solar Energy Forecasting - Data Setup")
    print("="*50)
    
    # Set up directory structure
    data_dir = setup_data_directory()
    
    # Check if data files exist
    data_files = {
        "Original data": "data/solarenergy2.csv",
        "Base anomaly": "data/solarenergy2_daily_anomaly_base_only.csv", 
        "Detrended anomaly": "data/solarenergy2_daily_anomaly_detrended.csv"
    }
    
    print("\nChecking data files...")
    all_files_ok = True
    
    for file_type, file_path in data_files.items():
        if check_csv_format(file_path, file_type):
            print(f"✓ {file_type} file is ready")
        else:
            print(f"✗ {file_type} file needs attention")
            all_files_ok = False
    
    # Create sample configuration
    create_sample_config()
    
    print("\n" + "="*50)
    print("SETUP SUMMARY")
    print("="*50)
    
    if all_files_ok:
        print("✓ All data files are ready!")
        print("\nNext steps:")
        print("1. Copy config_sample.yaml to config.yaml")
        print("2. Update the data paths in config.yaml if needed")
        print("3. Run: python main.py")
    else:
        print("⚠ Some data files need attention")
        print("\nPlease ensure your data files are in the correct format:")
        print("- CSV format with date and value columns")
        print("- Date column named: date, Date, DATE, time, Time, TIME, or datetime")
        print("- Value column named: gen_mwh, value, Value, or similar")
        print("- Files placed in the data/ directory")
    
    print("\nFor help with data format, check the README.md file")

if __name__ == "__main__":
    main()
