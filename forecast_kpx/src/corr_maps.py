"""
Correlation mapping functions for solar energy vs meteorological variables.
Creates Pearson and Spearman correlation maps with proper visualization.
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import Tuple, Optional
import os
from scipy.stats import spearmanr


def corr_map_pearson(da: xr.DataArray, series: pd.Series, dim: str = "time") -> xr.DataArray:
    """
    Calculate Pearson correlation map between gridded data and time series.
    
    Parameters:
    -----------
    da : xr.DataArray
        Gridded meteorological data (e.g., SSRD)
    series : pd.Series
        Energy time series (e.g., solar generation anomalies)
    dim : str
        Dimension to correlate along (typically 'time')
        
    Returns:
    --------
    xr.DataArray
        Pearson correlation map
    """
    # Align time series to match gridded data
    series_da = xr.DataArray(
        series.reindex(da.time.to_pandas()),
        dims=[dim],
        coords={dim: da.time}
    )
    
    # Calculate Pearson correlation
    corr = xr.corr(da, series_da, dim=dim)
    
    return corr


def corr_map_spearman(da: xr.DataArray, series: pd.Series, dim: str = "time") -> xr.DataArray:
    """
    Calculate Spearman correlation map between gridded data and time series.
    
    Parameters:
    -----------
    da : xr.DataArray
        Gridded meteorological data
    series : pd.Series
        Energy time series
    dim : str
        Dimension to correlate along
        
    Returns:
    --------
    xr.DataArray
        Spearman correlation map
    """
    # Align time series to match gridded data
    series_da = xr.DataArray(
        series.reindex(da.time.to_pandas()),
        dims=[dim],
        coords={dim: da.time}
    )
    
    # Rank both datasets
    da_rank = da.rank(dim)
    series_rank = series_da.rank(dim)
    
    # Calculate Pearson correlation of ranks (Spearman)
    corr = xr.corr(da_rank, series_rank, dim=dim)
    
    return corr


def plot_correlation_map(corr_map: xr.DataArray, title: str, 
                        output_path: str, vmin: Optional[float] = None,
                        vmax: Optional[float] = None, cmap: str = 'RdBu_r'):
    """
    Plot correlation map with Cartopy.
    
    Parameters:
    -----------
    corr_map : xr.DataArray
        Correlation map to plot
    title : str
        Plot title
    output_path : str
        Path to save the plot
    vmin, vmax : float, optional
        Color scale limits (auto-determined if None)
    cmap : str
        Colormap name
    """
    # Create figure with Cartopy projection
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set color scale limits
    if vmin is None or vmax is None:
        vmax = max(abs(corr_map.min()), abs(corr_map.max()))
        vmin = -vmax
    
    # Plot correlation map
    im = corr_map.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
        rasterized=True
    )
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    
    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Add Korea outline if within bounds
    korea_bounds = [124, 132, 33, 39.5]  # lon_min, lon_max, lat_min, lat_max
    if (corr_map.lon.min() <= korea_bounds[1] and corr_map.lon.max() >= korea_bounds[0] and
        corr_map.lat.min() <= korea_bounds[3] and corr_map.lat.max() >= korea_bounds[2]):
        
        # Add Korea bounding box
        ax.plot([korea_bounds[0], korea_bounds[1], korea_bounds[1], korea_bounds[0], korea_bounds[0]],
                [korea_bounds[2], korea_bounds[2], korea_bounds[3], korea_bounds[3], korea_bounds[2]],
                color='black', linewidth=2, transform=ccrs.PlateCarree())
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontsize=12)
    
    # Set title and labels
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Set extent to show relevant region
    if 'lon' in corr_map.coords and 'lat' in corr_map.coords:
        ax.set_extent([corr_map.lon.min(), corr_map.lon.max(),
                      corr_map.lat.min(), corr_map.lat.max()],
                     crs=ccrs.PlateCarree())
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_correlation_maps(energy_anomalies: pd.Series, ssrd_data: xr.DataArray,
                           config: dict, output_dir: str = "outputs/correlation_maps"):
    """
    Create correlation maps between energy anomalies and SSRD.
    
    Parameters:
    -----------
    energy_anomalies : pd.Series
        Energy anomaly time series
    ssrd_data : xr.DataArray
        SSRD (surface solar radiation downwards) data
    config : dict
        Configuration dictionary
    output_dir : str
        Output directory for correlation maps
    """
    print("Creating correlation maps...")
    
    # Mask bad window from energy anomalies if specified
    bad_window = config.get('timeframe', {}).get('bad_window')
    if bad_window:
        start_bad = pd.to_datetime(bad_window['start'])
        end_bad = pd.to_datetime(bad_window['end'])
        mask = (energy_anomalies.index >= start_bad) & (energy_anomalies.index <= end_bad)
        energy_anomalies = energy_anomalies.copy()
        energy_anomalies[mask] = np.nan
    
    # Calculate Pearson correlation
    print("Calculating Pearson correlations...")
    pearson_corr = corr_map_pearson(ssrd_data, energy_anomalies)
    
    # Calculate Spearman correlation
    print("Calculating Spearman correlations...")
    spearman_corr = corr_map_spearman(ssrd_data, energy_anomalies)
    
    # Plot Pearson correlation map
    pearson_path = os.path.join(output_dir, "pearson_ssrd_vs_energy.png")
    plot_correlation_map(
        pearson_corr,
        "Pearson Correlation: Energy Anomalies vs SSRD",
        pearson_path
    )
    print(f"Saved Pearson correlation map: {pearson_path}")
    
    # Plot Spearman correlation map
    spearman_path = os.path.join(output_dir, "spearman_ssrd_vs_energy.png")
    plot_correlation_map(
        spearman_corr,
        "Spearman Correlation: Energy Anomalies vs SSRD",
        spearman_path
    )
    print(f"Saved Spearman correlation map: {spearman_path}")
    
    # Save correlation data as NetCDF for further analysis
    corr_ds = xr.Dataset({
        'pearson_correlation': pearson_corr,
        'spearman_correlation': spearman_corr
    })
    
    corr_nc_path = os.path.join(output_dir, "correlation_maps.nc")
    corr_ds.to_netcdf(corr_nc_path)
    print(f"Saved correlation data: {corr_nc_path}")
    
    return pearson_corr, spearman_corr


def correlate_boxed_series(gen_anom: pd.Series, climate_anom_df: pd.DataFrame,
                           output_dir: str = "outputs/correlation_maps") -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations between generation anomaly and
    capacity-weighted climate anomalies (series-to-series, no maps).

    Saves a CSV summary and per-variable scatter plots.

    Output:
      - outputs/correlation_maps/boxed_correlation_summary.csv
      - outputs/plots/boxed_corr_[VAR].png
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join("outputs", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    variables = list(climate_anom_df.columns)
    records = []

    for var in variables:
        x = climate_anom_df[var]
        # Align indices and drop NaN
        common_idx = gen_anom.index.intersection(x.index)
        y = gen_anom.loc[common_idx]
        x = x.loc[common_idx]
        valid = ~(x.isna() | y.isna())
        x_valid = x[valid]
        y_valid = y[valid]
        if len(x_valid) < 3:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = np.corrcoef(x_valid, y_valid)[0, 1]
            spearman = spearmanr(x_valid, y_valid).correlation

        records.append({
            'variable': var,
            'pearson': pearson,
            'spearman': spearman,
            'n': int(valid.sum())
        })

        # Scatter plot
        try:
            plt.figure(figsize=(6, 5))
            plt.scatter(x_valid, y_valid, alpha=0.6, s=12)
            plt.title(f"Boxed Corr: {var}\nPearson={pearson:.3f}, Spearman={spearman:.3f}")
            plt.xlabel(var)
            plt.ylabel('gen_anom')
            plot_path = os.path.join(plots_dir, f"boxed_corr_{var}.png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close()
        except Exception:
            pass

    summary_df = pd.DataFrame.from_records(records)
    csv_path = os.path.join(output_dir, 'boxed_correlation_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved boxed correlation summary: {csv_path}")

    return summary_df


def analyze_correlation_patterns(corr_map: xr.DataArray, threshold: float = 0.3) -> dict:
    """
    Analyze correlation patterns and identify key regions.
    
    Parameters:
    -----------
    corr_map : xr.DataArray
        Correlation map
    threshold : float
        Threshold for significant correlations
        
    Returns:
    --------
    dict
        Analysis results
    """
    # Calculate statistics
    stats = {
        'mean': float(corr_map.mean()),
        'std': float(corr_map.std()),
        'min': float(corr_map.min()),
        'max': float(corr_map.max()),
        'significant_positive': float((corr_map > threshold).sum()),
        'significant_negative': float((corr_map < -threshold).sum()),
        'total_gridpoints': float(corr_map.size)
    }
    
    # Find maximum correlation location
    max_idx = corr_map.argmax()
    min_idx = corr_map.argmin()
    
    stats['max_corr'] = {
        'value': float(corr_map.max()),
        'lat': float(corr_map.lat.isel(lat=max_idx.lat)),
        'lon': float(corr_map.lon.isel(lon=max_idx.lon))
    }
    
    stats['min_corr'] = {
        'value': float(corr_map.min()),
        'lat': float(corr_map.lat.isel(lat=min_idx.lat)),
        'lon': float(corr_map.lon.isel(lon=min_idx.lon))
    }
    
    return stats


def create_correlation_summary(pearson_corr: xr.DataArray, spearman_corr: xr.DataArray,
                              output_path: str = "outputs/correlation_maps/summary.txt"):
    """
    Create a summary of correlation analysis.
    
    Parameters:
    -----------
    pearson_corr : xr.DataArray
        Pearson correlation map
    spearman_corr : xr.DataArray
        Spearman correlation map
    output_path : str
        Path to save summary
    """
    pearson_stats = analyze_correlation_patterns(pearson_corr)
    spearman_stats = analyze_correlation_patterns(spearman_corr)
    
    summary = f"""
Correlation Analysis Summary
============================

Pearson Correlation Statistics:
- Mean: {pearson_stats['mean']:.3f}
- Std: {pearson_stats['std']:.3f}
- Min: {pearson_stats['min']:.3f} (lat: {pearson_stats['min_corr']['lat']:.2f}, lon: {pearson_stats['min_corr']['lon']:.2f})
- Max: {pearson_stats['max']:.3f} (lat: {pearson_stats['max_corr']['lat']:.2f}, lon: {pearson_stats['max_corr']['lon']:.2f})
- Significant positive (>0.3): {pearson_stats['significant_positive']:.0f} gridpoints
- Significant negative (<-0.3): {pearson_stats['significant_negative']:.0f} gridpoints

Spearman Correlation Statistics:
- Mean: {spearman_stats['mean']:.3f}
- Std: {spearman_stats['std']:.3f}
- Min: {spearman_stats['min']:.3f} (lat: {spearman_stats['min_corr']['lat']:.2f}, lon: {spearman_stats['min_corr']['lon']:.2f})
- Max: {spearman_stats['max']:.3f} (lat: {spearman_stats['max_corr']['lat']:.2f}, lon: {spearman_stats['max_corr']['lon']:.2f})
- Significant positive (>0.3): {spearman_stats['significant_positive']:.0f} gridpoints
- Significant negative (<-0.3): {spearman_stats['significant_negative']:.0f} gridpoints

Total gridpoints: {pearson_stats['total_gridpoints']:.0f}
"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(summary)
    
    print(f"Saved correlation summary: {output_path}")
    print(summary)
