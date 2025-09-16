"""
Anomaly calculation functions for solar energy forecasting.
Implements DOY-based anomalies with and without detrending.
"""

import pandas as pd
import xarray as xr
import numpy as np
from typing import Tuple, Optional
from sklearn.linear_model import LinearRegression
import warnings


def daily_doy_anom(series: pd.Series, exclude: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
    """
    Calculate daily DOY (Day of Year) anomalies excluding a bad window from baseline fit.
    
    Parameters:
    -----------
    series : pd.Series
        Daily time series (e.g., solar generation)
    exclude : tuple, optional
        (start_date, end_date) to exclude from baseline fitting
        
    Returns:
    --------
    pd.DataFrame
        Columns: 'gen', 'clim_base', 'anom_base', 'anom_pct_base'
    """
    df = pd.DataFrame(index=series.index)
    df['gen'] = series
    
    # Calculate day of year
    df['doy'] = series.index.dayofyear
    
    # Create mask for bad window
    exclude_mask = pd.Series(False, index=series.index)
    if exclude:
        start_exclude = pd.to_datetime(exclude[0])
        end_exclude = pd.to_datetime(exclude[1])
        exclude_mask = (series.index >= start_exclude) & (series.index <= end_exclude)
    
    # Calculate climatology using only non-excluded data
    train_data = series[~exclude_mask]
    train_doy = train_data.index.dayofyear
    
    # Group by DOY and calculate mean
    clim_base = train_data.groupby(train_doy).mean()
    
    # Map climatology to full series
    df['clim_base'] = df['doy'].map(clim_base)
    
    # Calculate anomalies
    df['anom_base'] = df['gen'] - df['clim_base']
    df['anom_pct_base'] = (df['anom_base'] / df['clim_base']) * 100
    
    # Clean up
    df = df.drop('doy', axis=1)
    
    return df


def daily_doy_anom_detrended(series: pd.Series, exclude: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
    """
    Calculate detrended DOY anomalies.
    
    Process:
    1. Linear detrend the series
    2. Calculate DOY climatology on detrended data
    3. Add trend back to climatology
    4. Calculate detrended anomalies
    
    Parameters:
    -----------
    series : pd.Series
        Daily time series
    exclude : tuple, optional
        (start_date, end_date) to exclude from baseline fitting
        
    Returns:
    --------
    pd.DataFrame
        Columns: 'gen', 'clim_with_trend', 'anom_detrended'
    """
    df = pd.DataFrame(index=series.index)
    df['gen'] = series
    
    # Create mask for bad window
    exclude_mask = pd.Series(False, index=series.index)
    if exclude:
        start_exclude = pd.to_datetime(exclude[0])
        end_exclude = pd.to_datetime(exclude[1])
        exclude_mask = (series.index >= start_exclude) & (series.index <= end_exclude)
    
    # Use only non-excluded data for fitting
    train_data = series[~exclude_mask]
    train_time = np.arange(len(train_data))
    
    # Fit linear trend
    if len(train_data) < 2:
        warnings.warn("Insufficient data for trend fitting")
        trend = pd.Series(0, index=series.index)
    else:
        reg = LinearRegression()
        reg.fit(train_time.reshape(-1, 1), train_data.values)
        
        # Apply trend to full series
        full_time = np.arange(len(series))
        trend_values = reg.predict(full_time.reshape(-1, 1))
        trend = pd.Series(trend_values, index=series.index)
    
    # Detrend the data
    detrended = series - trend
    
    # Calculate DOY climatology on detrended data
    train_detrended = detrended[~exclude_mask]
    train_doy = train_detrended.index.dayofyear
    
    # Group by DOY and calculate mean
    clim_detrended = train_detrended.groupby(train_doy).mean()
    
    # Map climatology to full series and add trend back
    df['clim_with_trend'] = df.index.dayofyear.map(clim_detrended) + trend
    
    # Calculate detrended anomalies
    df['anom_detrended'] = df['gen'] - df['clim_with_trend']
    
    return df


def daily_doy_anom_xarray(da: xr.DataArray, exclude: Optional[Tuple[str, str]] = None) -> xr.DataArray:
    """
    Calculate DOY anomalies for gridded data (xarray DataArray).
    
    Parameters:
    -----------
    da : xr.DataArray
        Gridded time series with 'time' dimension
    exclude : tuple, optional
        (start_date, end_date) to exclude from baseline fitting
        
    Returns:
    --------
    xr.DataArray
        DOY anomalies with same dimensions as input
    """
    # Create mask for bad window
    exclude_mask = xr.zeros_like(da.isel(time=0), dtype=bool)
    if exclude:
        start_exclude = pd.to_datetime(exclude[0])
        end_exclude = pd.to_datetime(exclude[1])
        exclude_mask = (da.time >= start_exclude) & (da.time <= end_exclude)
    
    # Calculate day of year
    doy = da.time.dt.dayofyear
    
    # Calculate climatology using only non-excluded data
    da_train = da.where(~exclude_mask, drop=True)
    doy_train = doy.where(~exclude_mask, drop=True)
    
    # Group by DOY and calculate mean
    clim = da_train.groupby(doy_train).mean('time')
    
    # Map climatology back to full time series
    clim_full = clim.sel(dayofyear=doy)
    
    # Calculate anomalies
    anom = da - clim_full
    
    return anom


def daily_doy_anom_detrended_xarray(da: xr.DataArray, exclude: Optional[Tuple[str, str]] = None) -> xr.DataArray:
    """
    Calculate detrended DOY anomalies for gridded data.
    
    Parameters:
    -----------
    da : xr.DataArray
        Gridded time series with 'time' dimension
    exclude : tuple, optional
        (start_date, end_date) to exclude from baseline fitting
        
    Returns:
    --------
    xr.DataArray
        Detrended DOY anomalies
    """
    # Create mask for bad window
    exclude_mask = xr.zeros_like(da.isel(time=0), dtype=bool)
    if exclude:
        start_exclude = pd.to_datetime(exclude[0])
        end_exclude = pd.to_datetime(exclude[1])
        exclude_mask = (da.time >= start_exclude) & (da.time <= end_exclude)
    
    # Use only non-excluded data for fitting
    da_train = da.where(~exclude_mask, drop=True)
    
    # Fit linear trend for each grid point
    time_coord = np.arange(len(da_train.time))
    
    def fit_trend(y):
        """Fit linear trend for a single time series."""
        if np.isnan(y).all():
            return np.zeros_like(time_coord)
        
        # Remove NaN values
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 2:
            return np.zeros_like(time_coord)
        
        y_valid = y[valid_mask]
        t_valid = time_coord[valid_mask]
        
        # Fit linear regression
        coeffs = np.polyfit(t_valid, y_valid, 1)
        trend = np.polyval(coeffs, time_coord)
        
        return trend
    
    # Apply trend fitting across all grid points
    trends = xr.apply_ufunc(
        fit_trend,
        da_train,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )
    
    # Detrend the training data
    detrended_train = da_train - trends
    
    # Calculate DOY climatology on detrended data
    doy_train = da_train.time.dt.dayofyear
    clim_detrended = detrended_train.groupby(doy_train).mean('time')
    
    # For full time series, we need to interpolate/extend the trend
    # and apply climatology
    full_time = np.arange(len(da.time))
    
    def extend_trend(trend_train):
        """Extend trend to full time series."""
        if np.isnan(trend_train).all():
            return np.zeros(len(full_time))
        
        # Simple linear extension
        if len(trend_train) >= 2:
            slope = (trend_train[-1] - trend_train[0]) / (len(trend_train) - 1)
            trend_full = trend_train[0] + slope * full_time
        else:
            trend_full = np.full(len(full_time), trend_train[0] if len(trend_train) == 1 else 0)
        
        return trend_full
    
    # Extend trends to full time series
    trends_full = xr.apply_ufunc(
        extend_trend,
        trends,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        output_sizes={'time': len(da.time)}
    )
    trends_full = trends_full.assign_coords(time=da.time)
    
    # Apply climatology and trend to full series
    doy_full = da.time.dt.dayofyear
    clim_full = clim_detrended.sel(dayofyear=doy_full)
    
    # Calculate detrended anomalies
    anom_detrended = da - (clim_full + trends_full)
    
    return anom_detrended


def create_targets(gen_series: pd.Series, anom_detrended: pd.Series, 
                  config: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Create t+1 targets for both anomaly and total forecasting tracks.
    
    Parameters:
    -----------
    gen_series : pd.Series
        Original generation series
    anom_detrended : pd.Series
        Detrended anomaly series
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    Tuple[pd.Series, pd.Series]
        y_total (t+1 total generation), y_anom (t+1 detrended anomaly)
    """
    # Create t+1 targets (shift by -1 to get next day)
    y_total = gen_series.shift(-1)
    y_anom = anom_detrended.shift(-1)
    
    # Set names for clarity
    y_total.name = 'gen_mwh_t1'
    y_anom.name = 'anom_detrended_t1'
    
    return y_total, y_anom


def apply_train_only_transforms(series: pd.Series, config: dict, 
                              transform_func, *args, **kwargs) -> pd.DataFrame:
    """
    Apply transforms fitted only on training data to avoid leakage.
    
    Parameters:
    -----------
    series : pd.Series
        Full time series
    config : dict
        Configuration dictionary
    transform_func : callable
        Transform function to apply
    *args, **kwargs
        Additional arguments for transform function
        
    Returns:
    --------
    pd.DataFrame
        Transformed data
    """
    timeframe = config['timeframe']
    train_end = pd.to_datetime(timeframe['train_end'])
    
    # Fit transform on training data only
    train_data = series[series.index <= train_end]
    train_result = transform_func(train_data, *args, **kwargs)
    
    # Apply to full series
    full_result = transform_func(series, *args, **kwargs)
    
    return full_result
