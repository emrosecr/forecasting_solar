"""
Test script to verify installation and basic functionality.
Run this to check if all dependencies are properly installed.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("✓ pandas")
    except ImportError as e:
        print(f"✗ pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import xarray as xr
        print("✓ xarray")
    except ImportError as e:
        print(f"✗ xarray: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ seaborn")
    except ImportError as e:
        print(f"✗ seaborn: {e}")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError as e:
        print(f"✗ scikit-learn: {e}")
        return False
    
    try:
        import cartopy
        print("✓ cartopy")
    except ImportError as e:
        print(f"✗ cartopy: {e}")
        return False
    
    try:
        import yaml
        print("✓ PyYAML")
    except ImportError as e:
        print(f"✗ PyYAML: {e}")
        return False
    
    return True

def test_custom_modules():
    """Test if custom modules can be imported."""
    print("\nTesting custom modules...")
    
    # Add src to path
    sys.path.append('src')
    
    try:
        from io_load import load_kpx, load_era5
        print("✓ io_load")
    except ImportError as e:
        print(f"✗ io_load: {e}")
        return False
    
    try:
        from anomalies import daily_doy_anom_detrended
        print("✓ anomalies")
    except ImportError as e:
        print(f"✗ anomalies: {e}")
        return False
    
    try:
        from corr_maps import create_correlation_maps
        print("✓ corr_maps")
    except ImportError as e:
        print(f"✗ corr_maps: {e}")
        return False
    
    try:
        from features import create_local_features
        print("✓ features")
    except ImportError as e:
        print(f"✗ features: {e}")
        return False
    
    try:
        from models_baseline import BaselineModelSuite
        print("✓ models_baseline")
    except ImportError as e:
        print(f"✗ models_baseline: {e}")
        return False
    
    try:
        from models_rf_grid import RandomForestSuite
        print("✓ models_rf_grid")
    except ImportError as e:
        print(f"✗ models_rf_grid: {e}")
        return False
    
    try:
        from eval import calculate_metrics
        print("✓ eval")
    except ImportError as e:
        print(f"✗ eval: {e}")
        return False
    
    try:
        from plotting import plot_predictions_vs_truth
        print("✓ plotting")
    except ImportError as e:
        print(f"✗ plotting: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality with synthetic data."""
    print("\nTesting basic functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        import xarray as xr
        
        # Test data creation
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        gen_series = pd.Series(np.random.normal(1000, 100, len(dates)), index=dates)
        print("✓ Data creation")
        
        # Test anomaly calculation
        sys.path.append('src')
        from anomalies import daily_doy_anom_detrended
        anom_df = daily_doy_anom_detrended(gen_series)
        print("✓ Anomaly calculation")
        
        # Test feature creation
        lats = np.linspace(33, 39.5, 5)
        lons = np.linspace(124, 132, 5)
        ssrd_data = xr.DataArray(
            np.random.normal(20, 5, (len(dates), len(lats), len(lons))),
            coords={'time': dates, 'lat': lats, 'lon': lons},
            dims=['time', 'lat', 'lon']
        )
        print("✓ XArray data creation")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("KPX Solar Energy Forecasting - Installation Test")
    print("="*50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test custom modules
    modules_ok = test_custom_modules()
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    
    if imports_ok and modules_ok and functionality_ok:
        print("✓ All tests passed! Installation is successful.")
        print("\nYou can now run:")
        print("  python main.py")
        print("  python colab_demo.py")
        return 0
    else:
        print("✗ Some tests failed. Please check the installation.")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
