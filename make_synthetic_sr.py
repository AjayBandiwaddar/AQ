import numpy as np
from src.models.sr_unet import set_seed

def make_synthetic_sr_data(num_samples=8, in_channels=3, lr_shape=(16,16), hr_shape=(32,32)):
    """
    Create synthetic low-res and high-res arrays for SR training.

    Returns:
        low_res (np.ndarray): [N, C, H, W]
        high_res (np.ndarray): [N, 1, h, w]
    """
    set_seed(42)
    low_res = np.random.rand(num_samples, in_channels, *lr_shape)
    high_res = np.random.rand(num_samples, 1, *hr_shape)
    return low_res, high_res