import xarray as xr
import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

def make_feature_matrix(
    sat_ds: xr.Dataset,
    era_ds: xr.Dataset,
    grid: xr.Dataset,
    static_features: Dict[str, Any] = None,
    target_variable: str = "pm25"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine satellite, meteorological, static, and time features per grid cell.

    Args:
        sat_ds (xr.Dataset): Satellite data (e.g., NO2, QA)
        era_ds (xr.Dataset): ERA5 meteorological data (temp, wind, humidity)
        grid (xr.Dataset): Target grid definition (lat, lon)
        static_features (dict, optional): Static features (elevation, landcover)
        target_variable (str): Target variable (default: 'pm25')

    Returns:
        X (np.ndarray): Feature matrix [n_cells, n_features]
        y (np.ndarray): Target values [n_cells]
        coords (np.ndarray): Cell coordinates [n_cells, 2]

    Example:
        >>> X, y, coords = make_feature_matrix(sat_ds, era_ds, grid)
        # X: (n_cells, n_features), y: (n_cells,), coords: (n_cells,2)
    """
    # Flatten grid: lat/lon mesh
    lat = grid.lat.values if "lat" in grid else np.linspace(-90, 90, 4)
    lon = grid.lon.values if "lon" in grid else np.linspace(-180, 180, 4)
    lat_grid, lon_grid = np.meshgrid(lat, lon)
    n_cells = lat_grid.size

    # Features: satellite + meteorology + static
    sat_features = sat_ds.to_array().values.reshape(-1, n_cells).T
    era_features = era_ds.to_array().values.reshape(-1, n_cells).T
    static = []
    if static_features:
        for k, v in static_features.items():
            static.append(np.full(n_cells, v))
    static_features_arr = np.stack(static, axis=1) if static else np.zeros((n_cells,0))

    # Time features (e.g., month, day)
    time = sat_ds.time.values[0] if "time" in sat_ds else None
    if time is not None:
        month = np.full(n_cells, np.datetime64(time, 'M').astype(int) % 12 + 1)
        day = np.full(n_cells, np.datetime64(time, 'D').astype(int) % 31 + 1)
        time_features = np.stack([month, day], axis=1)
    else:
        time_features = np.zeros((n_cells,0))

    X = np.hstack([sat_features, era_features, static_features_arr, time_features])
    y = np.random.rand(n_cells)  # Placeholder: use ground truth mapping here

    coords = np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T
    return X, y, coords