import pytest
import xarray as xr
import numpy as np
from src.features import make_feature_matrix
from tests.utils import make_synthetic_satellite

def test_make_feature_matrix_shapes():
    sat_ds = make_synthetic_satellite((2,2), variable='no2')
    era_ds = make_synthetic_satellite((2,2), variable='temp')
    grid = sat_ds
    X, y, coords = make_feature_matrix(sat_ds, era_ds, grid)
    assert X.shape[0] == 4
    assert y.shape[0] == 4
    assert coords.shape == (4,2)

def test_make_feature_matrix_static_features():
    sat_ds = make_synthetic_satellite((2,2), variable='no2')
    era_ds = make_synthetic_satellite((2,2), variable='temp')
    grid = sat_ds
    static_features = {'elevation': 100, 'landcover': 2}
    X, y, coords = make_feature_matrix(sat_ds, era_ds, grid, static_features)
    assert X.shape[1] >= 4  # at least some features