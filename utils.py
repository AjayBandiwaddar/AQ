import xarray as xr
import numpy as np

def make_synthetic_satellite(shape=(2,2), variable='no2'):
    """
    Create a small synthetic satellite dataset for testing.

    Args:
        shape (tuple): (lat, lon) shape
        variable (str): variable name

    Returns:
        xr.Dataset
    """
    data = np.arange(1, shape[0]*shape[1]+1).reshape(shape)
    ds = xr.Dataset({variable: (["lat","lon"], data)})
    return ds