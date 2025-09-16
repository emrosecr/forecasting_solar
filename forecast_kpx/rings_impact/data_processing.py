"""
Data processing utilities for concentric climate rings and PV anomalies.

This module builds features by aggregating climate variables within concentric
rings around a site location. It also provides helpers to load/prepare targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr


EARTH_RADIUS_KM = 6371.0


def _to_0360(lon_deg: float) -> float:
    """Normalize longitude to [0, 360)."""
    return float((lon_deg + 360.0) % 360.0)


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """
    Compute haversine distance (km) from (lat2, lon2) to each (lat1, lon1) pair.
    lat1, lon1 may be 2D arrays broadcastable across grid; lat2, lon2 are scalars.
    """
    phi1 = np.deg2rad(lat1)
    phi2 = math.radians(lat2)
    dphi = phi1 - phi2

    # Handle wrapped longitude difference across dateline
    lam1 = np.deg2rad(lon1)
    lam2 = math.radians(lon2)
    dlam = lam1 - lam2
    dlam = (dlam + np.pi) % (2 * np.pi) - np.pi

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * math.cos(phi2) * np.sin(dlam / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


@dataclass
class RingSpec:
    center_lat: float
    center_lon: float  # degrees, 0-360 preferred
    ring_width_km: float
    num_rings: int


def build_ring_masks(ds: xr.Dataset, spec: RingSpec) -> List[xr.DataArray]:
    """
    Build boolean masks for concentric rings over the dataset grid.
    Returns list of DataArrays with dims ['lat','lon'] of dtype=bool.
    """
    if 'lat' not in ds.coords or 'lon' not in ds.coords:
        raise ValueError("Dataset must have 'lat' and 'lon' coordinates")

    # Create 2D lat/lon grids
    lat2d, lon2d = xr.broadcast(ds['lat'], ds['lon'])
    lat_grid = lat2d.values
    lon_grid = lon2d.values

    # Normalize center lon to dataset convention (assume 0-360 in this codebase)
    center_lon = _to_0360(spec.center_lon)

    # Compute distance grid
    dist_km = _haversine_km(lat_grid, lon_grid, spec.center_lat, center_lon)

    masks: List[xr.DataArray] = []
    for r in range(spec.num_rings):
        inner = spec.ring_width_km * r
        outer = spec.ring_width_km * (r + 1)
        ring_mask = (dist_km >= inner) & (dist_km < outer)
        masks.append(xr.DataArray(ring_mask, coords={'lat': ds['lat'], 'lon': ds['lon']}, dims=['lat', 'lon']))

    return masks


def area_weighted_mean_masked(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """
    Compute area-weighted mean across lat/lon for a masked grid.
    Returns a 1D DataArray over time.
    """
    if not set(['lat', 'lon']).issubset(set(da.dims)):
        raise ValueError("DataArray must have 'lat' and 'lon' dimensions")

    # Expand mask to include time, broadcast automatically
    masked = da.where(mask, drop=False)
    # Cosine latitude weights
    weights = np.cos(np.deg2rad(masked['lat']))
    # If a ring has no valid cells, result becomes NaN automatically
    ts = masked.weighted(weights).mean(dim=['lat', 'lon'], skipna=True)
    return ts


def aggregate_concentric_rings(
    ds: xr.Dataset,
    spec: RingSpec,
    variables: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate climate variables in concentric rings to build features.

    - ds: xarray Dataset with daily-aggregated variables (e.g., ssrd_sum, tcc, t2m, u10, v10)
    - spec: RingSpec describing center and rings
    - variables: optional subset of variable names to include

    Returns a pandas DataFrame indexed by date with columns like:
        ssrd_sum_ring1, ssrd_sum_ring2, ..., t2m_ring1, ...
    """
    if variables is None:
        variables = [v for v in ['ssrd_sum', 'tcc', 't2m', 'u10', 'v10'] if v in ds.data_vars]

    masks = build_ring_masks(ds, spec)
    features = pd.DataFrame(index=pd.to_datetime(ds['time'].values))
    features.index.name = 'date'

    # Derived wind speed magnitude if u10,v10 available
    computed_wind = False
    if 'u10' in ds.data_vars and 'v10' in ds.data_vars and 'wind10' not in variables:
        ds = ds.copy()
        ds['wind10'] = xr.apply_ufunc(lambda u, v: np.sqrt(u ** 2 + v ** 2), ds['u10'], ds['v10'])
        variables = variables + ['wind10']
        computed_wind = True

    for var in variables:
        if var not in ds.data_vars:
            continue
        for ring_idx, mask in enumerate(masks, start=1):
            ts = area_weighted_mean_masked(ds[var], mask)
            features[f"{var}_ring{ring_idx}"] = ts.to_pandas()

    return features


def load_generation_anomalies(gen_series: pd.Series, bad_window: Optional[Tuple[str, str]] = None,
                              standardize: bool = True) -> pd.Series:
    """
    Convert generation series to detrended DOY anomalies aligned with project pipeline,
    then optionally standardize to zero mean / unit variance.
    """
    from ..src.anomalies import daily_doy_anom_detrended

    exclude = bad_window if bad_window else None
    res = daily_doy_anom_detrended(gen_series, exclude=exclude)
    anom = res['anom_detrended'].copy()
    if standardize:
        mu = anom.mean()
        sigma = anom.std(ddof=0)
        if sigma and sigma > 0:
            anom = (anom - mu) / sigma
    anom.name = 'pv_anom_norm'
    return anom


def align_features_and_target(features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align feature and target indices, drop rows with all-NaN features or NaN target.
    """
    common_idx = features.index.intersection(target.index)
    X = features.loc[common_idx]
    y = target.loc[common_idx]
    valid = ~(y.isna()) & ~(X.isna().all(axis=1))
    return X[valid], y[valid]

