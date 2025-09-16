"""
I/O and preprocessing functions for KPX solar energy forecasting.
Handles loading KPX generation data and ERA5 meteorological data.
"""

import pandas as pd
import xarray as xr
import numpy as np
from typing import Union, Tuple, Optional
import warnings


def load_kpx(path: str) -> pd.Series:
    """
    Load KPX daily solar generation data.
    
    Parameters:
    -----------
    path : str
        Path to CSV or Parquet file containing KPX data
        
    Returns:
    --------
    pd.Series
        Daily solar generation in MWh, indexed by date
        Index name: 'date', Series name: 'gen_mwh'
    """
    # Try to load as CSV first, then Parquet
    try:
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        elif path.endswith('.parquet'):
            df = pd.read_parquet(path)
        else:
            # Try both formats
            try:
                df = pd.read_csv(path)
            except:
                df = pd.read_parquet(path)
    except Exception as e:
        raise ValueError(f"Could not load KPX data from {path}: {e}")
    
    # Handle different possible column names
    date_cols = ['date', 'Date', 'DATE', 'time', 'Time', 'TIME', 'datetime']
    gen_cols = ['gen_mwh', 'generation_mwh', 'solar_mwh', 'power_mwh', 'energy_mwh', 'value', 'Value', 'VALUE']
    
    # Find date column
    date_col = None
    for col in date_cols:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        raise ValueError(f"Date column not found. Available columns: {list(df.columns)}")
    
    # Find generation column
    gen_col = None
    for col in gen_cols:
        if col in df.columns:
            gen_col = col
            break
    
    if gen_col is None:
        # If no specific generation column, assume it's the only numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 1:
            gen_col = numeric_cols[0]
        else:
            raise ValueError(f"Generation column not found. Available columns: {list(df.columns)}")
    
    # Convert date column to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Create series with proper index and name
    series = pd.Series(
        df[gen_col].values,
        index=df[date_col].values,
        name='gen_mwh'
    )
    series.index.name = 'date'
    
    # Remove duplicates and sort
    series = series[~series.index.duplicated(keep='first')].sort_index()
    
    # Handle missing values
    if series.isna().any():
        warnings.warn(f"Found {series.isna().sum()} missing values in KPX data")
    
    return series


def load_kpx_multi_file(config: dict) -> tuple:
    """
    Load KPX data from multiple files (original, base anomaly, detrended anomaly).
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with file paths
        
    Returns:
    --------
    tuple
        (gen_series, anom_base_series, anom_detrended_series)
    """
    print("Loading KPX data from multiple files...")
    
    # Load original generation data
    try:
        gen_series = load_kpx(config['data']['kpx_original'])
        print(f"✓ Original data loaded: {len(gen_series)} days")
    except FileNotFoundError:
        print(f"✗ Original data not found: {config['data']['kpx_original']}")
        gen_series = None
    
    # Load base anomaly data
    try:
        anom_base_series = load_kpx(config['data']['kpx_anomaly_base'])
        anom_base_series.name = 'anom_base'
        print(f"✓ Base anomaly data loaded: {len(anom_base_series)} days")
    except FileNotFoundError:
        print(f"✗ Base anomaly data not found: {config['data']['kpx_anomaly_base']}")
        anom_base_series = None
    
    # Load detrended anomaly data
    try:
        anom_detrended_series = load_kpx(config['data']['kpx_anomaly_detrended'])
        anom_detrended_series.name = 'anom_detrended'
        print(f"✓ Detrended anomaly data loaded: {len(anom_detrended_series)} days")
    except FileNotFoundError:
        print(f"✗ Detrended anomaly data not found: {config['data']['kpx_anomaly_detrended']}")
        anom_detrended_series = None
    
    return gen_series, anom_base_series, anom_detrended_series


def load_era5(paths_or_store: Union[str, list]) -> xr.Dataset:
    """
    Load ERA5 meteorological data from NetCDF files.
    
    Parameters:
    -----------
    paths_or_store : str or list
        Path(s) to ERA5 NetCDF files or Zarr store
        
    Returns:
    --------
    xr.Dataset
        ERA5 dataset with normalized coordinates
    """
    try:
        if isinstance(paths_or_store, str):
            if paths_or_store.endswith('.zarr'):
                ds = xr.open_zarr(paths_or_store)
            else:
                # Handle glob patterns or single file
                ds = xr.open_mfdataset(paths_or_store, combine='by_coords')
        else:
            # List of files
            ds = xr.open_mfdataset(paths_or_store, combine='by_coords')
    except Exception as e:
        raise ValueError(f"Could not load ERA5 data: {e}")
    
    # Normalize coordinate names
    coord_mapping = {}
    if 'valid_time' in ds.coords:
        coord_mapping['valid_time'] = 'time'
    if 'latitude' in ds.coords:
        coord_mapping['latitude'] = 'lat'
    if 'longitude' in ds.coords:
        coord_mapping['longitude'] = 'lon'
    
    if coord_mapping:
        ds = ds.rename(coord_mapping)
    
    # Ensure longitude is in 0-360 range
    if ds.lon.max() <= 180:
        ds = ds.assign_coords(lon=(ds.lon + 360) % 360)
    
    # Sort by time
    ds = ds.sortby('time')
    
    # Check required variables
    required_vars = ['ssrd', 'tcc', 'tcwv', 'u10', 'v10', 't2m']
    missing_vars = [var for var in required_vars if var not in ds.data_vars]
    if missing_vars:
        warnings.warn(f"Missing ERA5 variables: {missing_vars}")
    
    return ds


def subset_bbox(ds: xr.Dataset, north: float, south: float, 
                west_e: float, east_e: float) -> xr.Dataset:
    """
    Subset dataset to a bounding box.
    
    Parameters:
    -----------
    ds : xr.Dataset
        Input dataset
    north, south : float
        Northern and southern latitude bounds
    west_e, east_e : float
        Western and eastern longitude bounds (0-360)
        
    Returns:
    --------
    xr.Dataset
        Subset dataset
    """
    # Handle latitude ordering (ERA5 typically has descending latitude)
    lat_slice = slice(max(north, south), min(north, south))
    
    # Handle longitude wrapping
    if east_e > west_e:
        lon_slice = slice(west_e, east_e)
    else:
        # Crosses prime meridian
        lon_slice = slice(west_e, 360)
        ds_west = ds.sel(lon=lon_slice)
        ds_east = ds.sel(lon=slice(0, east_e))
        return xr.concat([ds_west, ds_east], dim='lon')
    
    return ds.sel(lat=lat_slice, lon=lon_slice)


def daily_agg(ds: xr.Dataset) -> xr.Dataset:
    """
    Aggregate hourly ERA5 data to daily values.
    
    Parameters:
    -----------
    ds : xr.Dataset
        ERA5 dataset (hourly or daily)
        
    Returns:
    --------
    xr.Dataset
        Daily aggregated dataset
    """
    # Check if already daily
    if len(ds.time) > 1:
        time_diff = pd.to_datetime(ds.time.values[1]) - pd.to_datetime(ds.time.values[0])
        if time_diff >= pd.Timedelta(days=1):
            # Already daily
            return ds
    
    # Daily aggregation rules
    agg_dict = {}
    
    # Sum for radiation (J/m² to J/m²/day)
    if 'ssrd' in ds.data_vars:
        agg_dict['ssrd'] = 'sum'
    
    # Mean for other variables
    mean_vars = ['tcc', 'tcwv', 'u10', 'v10', 't2m']
    for var in mean_vars:
        if var in ds.data_vars:
            agg_dict[var] = 'mean'
    
    # Aggregate
    ds_daily = ds.resample(time='1D').agg(agg_dict)
    
    # Rename ssrd to ssrd_sum for clarity
    if 'ssrd' in ds_daily.data_vars:
        ds_daily = ds_daily.rename({'ssrd': 'ssrd_sum'})
    
    return ds_daily


def create_korea_and_extended_datasets(era5_path: str, config: dict) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Create both Korea and extended domain datasets from ERA5.
    
    Parameters:
    -----------
    era5_path : str
        Path to ERA5 data
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    Tuple[xr.Dataset, xr.Dataset]
        Korea dataset, extended dataset
    """
    # Load full ERA5 dataset
    ds_full = load_era5(era5_path)
    
    # Create Korea subset
    korea_bbox = config['data']['korea_bbox']
    ds_korea = subset_bbox(
        ds_full, 
        korea_bbox['lat_max'], 
        korea_bbox['lat_min'],
        korea_bbox['lon_min'], 
        korea_bbox['lon_max']
    )
    
    # Create extended subset
    extended_bbox = config['data']['extended_bbox']
    ds_extended = subset_bbox(
        ds_full,
        extended_bbox['lat_max'],
        extended_bbox['lat_min'], 
        extended_bbox['lon_min'],
        extended_bbox['lon_max']
    )
    
    # Convert to daily if needed
    ds_korea = daily_agg(ds_korea)
    ds_extended = daily_agg(ds_extended)
    
    return ds_korea, ds_extended


def check_data_availability(kpx_series: pd.Series, era5_ds: xr.Dataset, 
                          config: dict) -> dict:
    """
    Check data availability and coverage for the specified time period.
    
    Parameters:
    -----------
    kpx_series : pd.Series
        KPX generation data
    era5_ds : xr.Dataset
        ERA5 meteorological data
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    dict
        Data availability summary
    """
    timeframe = config['timeframe']
    start_date = pd.to_datetime(timeframe['start'])
    end_date = pd.to_datetime(timeframe['end'])
    
    # Check KPX data
    kpx_in_period = kpx_series[(kpx_series.index >= start_date) & 
                              (kpx_series.index <= end_date)]
    
    # Check ERA5 data
    era5_in_period = era5_ds.sel(time=slice(start_date, end_date))
    
    summary = {
        'kpx': {
            'total_days': len(kpx_in_period),
            'missing_days': kpx_in_period.isna().sum(),
            'coverage_pct': (1 - kpx_in_period.isna().sum() / len(kpx_in_period)) * 100
        },
        'era5': {
            'total_days': len(era5_in_period.time),
            'variables': list(era5_in_period.data_vars.keys()),
            'spatial_shape': (len(era5_in_period.lat), len(era5_in_period.lon))
        }
    }
    
    return summary
