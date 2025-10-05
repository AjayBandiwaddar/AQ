import pytest
import numpy as np
import torch
from src.models.sr_unet import UNetSR, set_seed, SRDataset

def test_unet_forward_shape():
    set_seed(42)
    model = UNetSR(in_channels=3, out_channels=1)
    x = torch.randn(2,3,16,16)
    y = model(x)
    assert y.shape == (2,1,16,16)

def test_sr_dataset_shapes():
    set_seed(42)
    low = np.random.rand(5, 3, 16, 16)
    high = np.random.rand(5, 1, 32, 32)
    ds = SRDataset(low, high, patch_size=16, augment=False)
    x, y = ds[0]
    assert x.shape == (3,16,16)
    assert y.shape == (1,32,32)
    # Check determinism
    ds2 = SRDataset(low, high, patch_size=16, augment=False)
    x2, y2 = ds2[0]
    assert torch.allclose(x, x2)
    assert torch.allclose(y, y2)