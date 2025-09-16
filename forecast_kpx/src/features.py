"""
Feature engineering for solar energy forecasting.
Creates local (Korea) and extended (East Asia/Pacific) feature sets.
"""

import pandas as pd
import xarray as xr
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings


def area_weighted_mean(da: xr.DataArray) -> pd.Series:
    """
    Calculate area-weighted mean across lat/lon dimensions.
    
    Parameters:
    -----------
    da : xr.DataArray
        Gridded data with lat/lon dimensions
        
    Returns:
    --------
    pd.Series
        Area-weighted time series
    """
    # Calculate cosine latitude weights
    weights = np.cos(np.deg2rad(da.lat))
    
    # Calculate area-weighted mean
    series = da.weighted(weights).mean(['lat', 'lon'], skipna=True)
    
    # Convert to pandas Series
    return series.to_pandas()


def create_local_features(ds_korea: xr.Dataset, config: dict) -> pd.DataFrame:
    """
    Create local (Korea box) area-weighted features.
    
    Parameters:
    -----------
    ds_korea : xr.Dataset
        Korea subset of ERA5 data
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    pd.DataFrame
        Local features with daily index
    """
    print("Creating local features...")
    
    features = pd.DataFrame(index=pd.to_datetime(ds_korea.time.values))
    features.index.name = 'date'
    
    # SSRD sum (daily cumulative radiation)
    if 'ssrd_sum' in ds_korea.data_vars:
        features['ssrd_kr_sum'] = area_weighted_mean(ds_korea['ssrd_sum'])
    elif 'ssrd' in ds_korea.data_vars:
        features['ssrd_kr_sum'] = area_weighted_mean(ds_korea['ssrd'])
    
    # Other meteorological variables (daily means)
    meteo_vars = ['tcc', 'tcwv', 'u10', 'v10', 't2m']
    for var in meteo_vars:
        if var in ds_korea.data_vars:
            features[f'{var}_kr'] = area_weighted_mean(ds_korea[var])
    
    # Wind speed (magnitude of u10, v10)
    if 'u10_kr' in features.columns and 'v10_kr' in features.columns:
        features['wind10_kr'] = np.sqrt(
            features['u10_kr']**2 + features['v10_kr']**2
        )
    
    # Temperature anomaly (departure from mean)
    if 't2m_kr' in features.columns:
        t2m_mean = features['t2m_kr'].mean()
        features['t2m_kr_anom'] = features['t2m_kr'] - t2m_mean
    
    return features


def create_extended_features(ds_extended: xr.Dataset, config: dict,
                           train_mask: pd.Series) -> pd.DataFrame:
    """
    Create extended features from East Asia/Pacific domain.
    
    Parameters:
    -----------
    ds_extended : xr.Dataset
        Extended domain ERA5 data
    config : dict
        Configuration dictionary
    train_mask : pd.Series
        Boolean mask for training period
        
    Returns:
    --------
    pd.DataFrame
        Extended features
    """
    print("Creating extended features...")
    
    # Check memory constraints
    memory_config = config.get('memory', {})
    use_pca = memory_config.get('use_pca_extended', True)
    downsample = memory_config.get('grid_downsample_extended', True)
    
    if downsample:
        # Downsample grid for memory efficiency
        stride = config['models']['random_forest'].get('grid_downsample', 2)
        ds_extended = ds_extended.isel(
            lat=slice(None, None, stride),
            lon=slice(None, None, stride)
        )
        print(f"Downsampled grid by factor {stride}")
    
    features = pd.DataFrame(index=pd.to_datetime(ds_extended.time.values))
    features.index.name = 'date'
    
    if use_pca:
        # Option A: PCA compression approach
        features = _create_extended_features_pca(ds_extended, config, train_mask)
    else:
        # Option B: Flatten grid approach (memory intensive)
        features = _create_extended_features_flatten(ds_extended, config)
    
    return features


def _create_extended_features_pca(ds_extended: xr.Dataset, config: dict,
                                 train_mask: pd.Series) -> pd.DataFrame:
    """
    Create extended features using PCA compression.
    """
    features = pd.DataFrame(index=pd.to_datetime(ds_extended.time.values))
    features.index.name = 'date'
    
    # Area-weighted means for key upstream regions
    features.update(_create_upstream_regions(ds_extended))
    
    # PCA features for SSRD anomalies
    features.update(_create_pca_features(ds_extended, config, train_mask))
    
    return features


def _create_extended_features_flatten(ds_extended: xr.Dataset, config: dict) -> pd.DataFrame:
    """
    Create extended features by flattening grid (memory intensive).
    """
    features = pd.DataFrame(index=pd.to_datetime(ds_extended.time.values))
    features.index.name = 'date'
    
    # Flatten each variable's grid to features
    for var in ds_extended.data_vars:
        da = ds_extended[var]
        
        # Reshape to (time, lat*lon)
        n_time = len(da.time)
        n_lat = len(da.lat)
        n_lon = len(da.lon)
        
        da_flat = da.values.reshape(n_time, n_lat * n_lon)
        
        # Create feature names
        for i in range(n_lat * n_lon):
            lat_idx = i // n_lon
            lon_idx = i % n_lon
            lat_val = float(da.lat[lat_idx])
            lon_val = float(da.lon[lon_idx])
            
            feature_name = f"{var}_{lat_val:.1f}N_{lon_val:.1f}E"
            features[feature_name] = da_flat[:, i]
    
    print(f"Created {len(features.columns)} flattened features")
    return features


def _create_upstream_regions(ds_extended: xr.Dataset) -> pd.DataFrame:
    """
    Create area-weighted means for key upstream regions.
    """
    features = pd.DataFrame(index=pd.to_datetime(ds_extended.time.values))
    features.index.name = 'date'
    
    # Define upstream regions
    regions = {
        'east_china': {'lat': (25, 35), 'lon': (115, 125)},  # East China Sea
        'wnp': {'lat': (15, 25), 'lon': (120, 140)},        # Western North Pacific
        'north_pacific': {'lat': (35, 45), 'lon': (140, 160)}  # North Pacific
    }
    
    for region_name, bounds in regions.items():
        # Subset to region
        ds_region = ds_extended.sel(
            lat=slice(bounds['lat'][1], bounds['lat'][0]),  # Note: ERA5 lat descending
            lon=slice(bounds['lon'][0], bounds['lon'][1])
        )
        
        # Calculate area-weighted means
        for var in ds_region.data_vars:
            feature_name = f"{var}_{region_name}"
            features[feature_name] = area_weighted_mean(ds_region[var])
    
    return features


def _create_pca_features(ds_extended: xr.Dataset, config: dict,
                        train_mask: pd.Series) -> pd.DataFrame:
    """
    Create PCA features for meteorological variables.
    """
    features = pd.DataFrame(index=pd.to_datetime(ds_extended.time.values))
    features.index.name = 'date'
    
    # Get training data for PCA fitting
    train_ds = ds_extended.sel(time=ds_extended.time[train_mask])
    
    # Number of PCA components
    n_components = config['models']['random_forest'].get('pca_components', 10)
    
    # Apply PCA to each variable
    for var in ds_extended.data_vars:
        if var in ['ssrd_sum', 'ssrd']:  # Focus on radiation
            da = ds_extended[var]
            
            # Reshape for PCA: (time, lat*lon)
            n_time = len(da.time)
            n_lat = len(da.lat)
            n_lon = len(da.lon)
            
            # Flatten spatial dimensions
            da_flat = da.values.reshape(n_time, n_lat * n_lon)
            
            # Fit PCA on training data only
            train_data = da_flat[train_mask]
            
            # Remove NaN values for PCA
            valid_mask = ~np.isnan(train_data).any(axis=1)
            if valid_mask.sum() < n_components:
                warnings.warn(f"Insufficient valid data for PCA on {var}")
                continue
            
            train_clean = train_data[valid_mask]
            
            # Fit PCA
            pca = PCA(n_components=min(n_components, train_clean.shape[1]))
            pca.fit(train_clean)
            
            # Transform full dataset
            pca_features = pca.transform(da_flat)
            
            # Add PCA features
            for i in range(pca_features.shape[1]):
                features[f"{var}_pc{i+1}"] = pca_features[:, i]
            
            print(f"Created {pca_features.shape[1]} PCA features for {var}")
    
    return features


def add_calendar_features(features: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add calendar-based features.
    
    Parameters:
    -----------
    features : pd.DataFrame
        Input features DataFrame
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    pd.DataFrame
        Features with calendar variables added
    """
    calendar_config = config.get('features', {}).get('calendar', {})
    
    # Day of year (seasonal cycle)
    if calendar_config.get('include_seasonal', True):
        features['doy'] = features.index.dayofyear
        features['sinDoy'] = np.sin(2 * np.pi * features['doy'] / 365.25)
        features['cosDoy'] = np.cos(2 * np.pi * features['doy'] / 365.25)
    
    # Day of week (weekly cycle)
    if calendar_config.get('include_dow', True):
        features['dow'] = features.index.dayofweek  # 0=Monday, 6=Sunday
    
    return features


def add_lag_features(features: pd.DataFrame, energy_series: pd.Series,
                    meteo_features: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add lagged features to avoid leakage.
    
    Parameters:
    -----------
    features : pd.DataFrame
        Base features DataFrame
    energy_series : pd.Series
        Energy generation series
    meteo_features : pd.DataFrame
        Meteorological features
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    pd.DataFrame
        Features with lag variables added
    """
    lag_config = config.get('features', {}).get('lags', {})
    
    # Energy lags
    energy_lags = lag_config.get('energy', [1, 2, 3, 7])
    for lag in energy_lags:
        features[f'gen_lag{lag}'] = energy_series.shift(lag)
    
    # Meteorological lags
    meteo_lags = lag_config.get('meteo', [1, 2, 3])
    meteo_vars = ['ssrd_kr_sum', 'tcc_kr', 'tcwv_kr', 'wind10_kr', 't2m_kr']
    
    for var in meteo_vars:
        if var in meteo_features.columns:
            for lag in meteo_lags:
                features[f'{var}_lag{lag}'] = meteo_features[var].shift(lag)
    
    return features


def create_feature_sets(gen_series: pd.Series, ds_korea: xr.Dataset,
                       ds_extended: xr.Dataset, config: dict,
                       train_mask: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create both local and extended feature sets.
    
    Parameters:
    -----------
    gen_series : pd.Series
        Solar generation series
    ds_korea : xr.Dataset
        Korea domain ERA5 data
    ds_extended : xr.Dataset
        Extended domain ERA5 data
    config : dict
        Configuration dictionary
    train_mask : pd.Series
        Training period mask
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Local features, extended features
    """
    # Create base local features
    X_local = create_local_features(ds_korea, config)
    
    # Create extended features
    X_extended = create_extended_features(ds_extended, config, train_mask)
    
    # Add calendar features to both
    X_local = add_calendar_features(X_local, config)
    X_extended = add_calendar_features(X_extended, config)
    
    # Add lag features
    X_local = add_lag_features(X_local, gen_series, X_local, config)
    X_extended = add_lag_features(X_extended, gen_series, X_local, config)
    
    # Ensure same index
    common_index = X_local.index.intersection(X_extended.index)
    X_local = X_local.loc[common_index]
    X_extended = X_extended.loc[common_index]
    
    print(f"Local features shape: {X_local.shape}")
    print(f"Extended features shape: {X_extended.shape}")
    
    return X_local, X_extended


def save_features(X_local: pd.DataFrame, X_extended: pd.DataFrame,
                 output_dir: str = "outputs/features"):
    """
    Save feature sets to parquet files.
    
    Parameters:
    -----------
    X_local : pd.DataFrame
        Local features
    X_extended : pd.DataFrame
        Extended features
    output_dir : str
        Output directory
    """
    import os
    
    # Create output directories
    local_dir = os.path.join(output_dir, "local")
    extended_dir = os.path.join(output_dir, "extended")
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(extended_dir, exist_ok=True)
    
    # Save features
    X_local.to_parquet(os.path.join(local_dir, "features.parquet"))
    X_extended.to_parquet(os.path.join(extended_dir, "features.parquet"))
    
    # Save feature names for reference
    with open(os.path.join(local_dir, "feature_names.txt"), 'w') as f:
        f.write('\n'.join(X_local.columns))
    
    with open(os.path.join(extended_dir, "feature_names.txt"), 'w') as f:
        f.write('\n'.join(X_extended.columns))
    
    print(f"Saved local features: {len(X_local.columns)} features")
    print(f"Saved extended features: {len(X_extended.columns)} features")


def load_features(feature_type: str, output_dir: str = "outputs/features") -> pd.DataFrame:
    """
    Load saved feature sets.
    
    Parameters:
    -----------
    feature_type : str
        'local' or 'extended'
    output_dir : str
        Features directory
        
    Returns:
    --------
    pd.DataFrame
        Loaded features
    """
    import os
    feature_path = os.path.join(output_dir, feature_type, "features.parquet")
    return pd.read_parquet(feature_path)
