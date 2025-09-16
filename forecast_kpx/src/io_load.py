"""
I/O and preprocessing functions for KPX solar energy forecasting.
Handles loading KPX generation data and ERA5 meteorological data.
"""

import pandas as pd
import xarray as xr
import numpy as np
from typing import Union, Tuple, Optional, List, Dict
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


# ===== Three-Site Boxed Climate (Area-weighted per-site, Capacity-weighted combine) =====

def _get_default_site_definitions() -> List[dict]:
    """
    Return default site definitions (approximate coordinates) for three named sites.
    Names (Korean):
      - 안산연성정수장태양광 (Ansan Water Treatment PV)
      - 세종시폐기물내립장태양광 (Sejong Landfill PV)
      - 영암에프원태양광 (Yeongam F1 PV)
    """
    return [
        {
            'name': '안산연성정수장태양광',
            'lat': 37.30,
            'lon': 126.80,
        },
        {
            'name': '세종시폐기물내립장태양광',
            'lat': 36.50,
            'lon': 127.30,
        },
        {
            'name': '영암에프원태양광',
            'lat': 34.70,
            'lon': 126.40,
        },
    ]


def get_site_definitions(config: dict) -> List[dict]:
    """
    Get site definitions from config if provided, otherwise use defaults.

    Expected config structure (optional):
        config['sites']['boxes'] = [
            {'name': '...', 'lat': float, 'lon': float}, ...
        ]
    """
    sites_cfg = config.get('sites', {}).get('boxes')
    if isinstance(sites_cfg, list) and len(sites_cfg) >= 3:
        cleaned: List[dict] = []
        for item in sites_cfg[:3]:
            if all(k in item for k in ['name', 'lat', 'lon']):
                cleaned.append({'name': item['name'], 'lat': float(item['lat']), 'lon': float(item['lon'])})
        if len(cleaned) == 3:
            return cleaned
    return _get_default_site_definitions()


def select_site_box(ds: xr.Dataset, lat_center: float, lon_center: float, half_side_deg: float) -> xr.Dataset:
    """
    Select a square box around a site with a fixed half-side in degrees.
    Uses existing subset_bbox utility to handle lat ordering and lon wrapping.
    """
    north = lat_center + half_side_deg
    south = lat_center - half_side_deg
    west = (lon_center - half_side_deg) % 360
    east = (lon_center + half_side_deg) % 360
    return subset_bbox(ds, north=north, south=south, west_e=west, east_e=east)


def area_weighted_time_series(da: xr.DataArray) -> pd.Series:
    """
    Compute area-weighted mean time series over lat/lon for a DataArray.
    The weighting uses cosine of latitude; intensive variables are averaged (not summed).
    """
    # cosine(lat) weights across lat dimension; broadcast to lon
    weights = np.cos(np.deg2rad(da['lat']))
    ts = da.weighted(weights).mean(dim=['lat', 'lon'], skipna=True)
    return ts.to_pandas()


def compute_per_site_area_weighted_means(ds_korea: xr.Dataset, config: dict) -> Dict[str, pd.DataFrame]:
    """
    Compute per-site, per-variable area-weighted daily means inside fixed boxes.

    Returns a dict mapping variable -> DataFrame with columns = site names, index = date.
    Variables considered if present: ['ssrd_sum', 't2m', 'u10', 'v10', 'tcc'].
    """
    sites = get_site_definitions(config)
    half_side = float(config.get('sites', {}).get('box_half_deg', 0.5))

    var_list = [v for v in ['ssrd_sum', 't2m', 'u10', 'v10', 'tcc'] if v in ds_korea.data_vars]
    per_var: Dict[str, pd.DataFrame] = {}

    for var in var_list:
        site_series: Dict[str, pd.Series] = {}
        for site in sites:
            ds_box = select_site_box(ds_korea[[var]], site['lat'], site['lon'], half_side)
            ts = area_weighted_time_series(ds_box[var])
            site_series[site['name']] = ts
        # Align into a DataFrame
        df = pd.DataFrame(site_series)
        df.index.name = 'date'
        per_var[var] = df

    # Derived wind10 magnitude if u10 and v10 exist
    if 'u10' in per_var and 'v10' in per_var:
        u_df = per_var['u10']
        v_df = per_var['v10']
        wind_df = np.sqrt(u_df.pow(2) + v_df.pow(2))
        wind_df.columns = u_df.columns
        per_var['wind10'] = wind_df

    return per_var


def load_site_capacities_from_kpx(original_path: str, site_names: List[str]) -> pd.Series:
    """
    Attempt to load per-site capacities (MW) from the KPX original CSV.
    Falls back to equal weights if capacities cannot be parsed.
    Heuristics: look for columns like ['site','name'] and ['capacity_mw','capacity','설비용량'].
    """
    capacities = pd.Series(index=site_names, dtype=float)
    try:
        df = pd.read_csv(original_path)
        cols_lower = {c: c.lower() for c in df.columns}
        # Candidate name columns
        name_cols = [c for c in df.columns if any(tok in cols_lower[c] for tok in ['site', 'name', '발전소', '설비명'])]
        cap_cols = [c for c in df.columns if any(tok in cols_lower[c] for tok in ['capacity_mw', 'capacity', '설비용량', '용량'])]
        if name_cols and cap_cols:
            # Use the first matching pair; group by name and take max capacity
            tmp = df[[name_cols[0], cap_cols[0]]].copy()
            tmp.columns = ['name', 'capacity']
            tmp['name'] = tmp['name'].astype(str)
            # Aggregate capacities per site name
            cap_map = tmp.groupby('name')['capacity'].max()
            for sn in site_names:
                # Exact match or contains
                match = cap_map.filter(like=sn)
                if sn in cap_map.index:
                    capacities.loc[sn] = float(cap_map.loc[sn])
                elif len(match) > 0:
                    capacities.loc[sn] = float(match.iloc[0])
        
        # If still missing or invalid, fill with equal weights
        if capacities.isna().any() or (capacities <= 0).any():
            raise ValueError("Parsed capacities are incomplete or non-positive")
    except Exception:
        # Fallback: equal weights (1.0 MW each)
        capacities = pd.Series(1.0, index=site_names)
    capacities.name = 'capacity_mw'
    return capacities


def capacity_weighted_combine(per_site: Dict[str, pd.DataFrame], capacities_mw: pd.Series) -> Dict[str, pd.Series]:
    """
    Combine per-site series into capacity-weighted means per variable.
    Returns dict var -> pd.Series.
    """
    combined: Dict[str, pd.Series] = {}
    # Normalize weights to sum to 1
    w = capacities_mw.reindex(capacities_mw.index).astype(float)
    w = w / w.sum() if w.sum() != 0 else pd.Series(1.0 / len(w), index=w.index)

    for var, df in per_site.items():
        # Align columns to capacities index order
        cols = [c for c in w.index if c in df.columns]
        if len(cols) == 0:
            continue
        X = df[cols]
        weights = w.loc[cols]
        combined[var] = (X * weights.values).sum(axis=1)
        combined[var].name = f"{var}_boxed"
    return combined


def build_boxed_climate_series(ds_korea: xr.Dataset, config: dict, kpx_original_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build capacity-weighted combined climate series (raw and anomaly variants).
    - Per-site area-weighted daily means inside fixed boxes
    - Capacity-weighted combine across three sites
    - Return two DataFrames (raw_total, anom_detrended) with aligned daily index
    """
    # Per-site area-weighted means
    per_site = compute_per_site_area_weighted_means(ds_korea, config)

    # Capacities from KPX original CSV or config
    site_names = list(next(iter(per_site.values())).columns)
    if kpx_original_path is None:
        kpx_original_path = config.get('data', {}).get('kpx_original', '')
    capacities = load_site_capacities_from_kpx(kpx_original_path, site_names)

    # Capacity-weighted combine
    combined = capacity_weighted_combine(per_site, capacities)

    # Assemble raw total DataFrame
    raw_df = pd.DataFrame(combined)
    raw_df.index.name = 'date'

    # Create anomalies using same settings as generation anomalies
    from anomalies import daily_doy_anom_detrended
    bad_window_cfg = config.get('timeframe', {}).get('bad_window', None)
    exclude = None
    if bad_window_cfg:
        exclude = (bad_window_cfg.get('start'), bad_window_cfg.get('end'))

    anom_cols: Dict[str, pd.Series] = {}
    for col in raw_df.columns:
        res = daily_doy_anom_detrended(raw_df[col], exclude=exclude)
        anom = res['anom_detrended']
        anom.name = f"{col}_anom"
        anom_cols[anom.name] = anom
    anom_df = pd.DataFrame(anom_cols)
    anom_df.index.name = 'date'

    return raw_df, anom_df
