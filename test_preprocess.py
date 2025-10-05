import pytest
from src.preprocess import resample_to_target
from tests.utils import make_synthetic_satellite
import numpy as np

def test_resample_to_target_basic():
    ds = make_synthetic_satellite((2,2))
    profile = {"shape": (4,4)}
    out = resample_to_target(ds, profile)
    assert out.no2.shape == (4,4)
    # Center interpolation check (approximate)
    center = out.no2.values[2,2]
    assert np.isclose(center, 2.5, atol=0.5)

def test_resample_to_target_deterministic():
    ds = make_synthetic_satellite((2,2))
    profile = {"shape": (4,4)}
    out1 = resample_to_target(ds, profile)
    out2 = resample_to_target(ds, profile)
    assert np.allclose(out1.no2.values, out2.no2.values)