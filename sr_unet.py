import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class UNetSR(nn.Module):
    """
    U-Net style super-resolution network for remote sensing downscaling.

    Args:
        in_channels (int): Number of input channels (e.g., satellite + covariates)
        out_channels (int): Number of output channels (e.g., PM2.5 map)
        base_filters (int): Base number of filters

    Example:
        >>> model = UNetSR(in_channels=3, out_channels=1)
        >>> y = model(torch.randn(2,3,16,16))
    """
    def __init__(self, in_channels: int, out_channels: int, base_filters: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, base_filters, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(base_filters, base_filters*2, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.dec1 = nn.Sequential(nn.Conv2d(base_filters*2, base_filters, 3, padding=1), nn.ReLU())
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.outc = nn.Conv2d(base_filters, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.enc2(x2)
        x4 = self.up(x3)
        x5 = self.dec1(x4)
        x6 = self.outc(x5)
        return x6

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed

    Example:
        >>> set_seed(123)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

class SRDataset(torch.utils.data.Dataset):
    """
    Dataset for SR training with on-the-fly patching and augmentations.

    Args:
        low_res (np.ndarray): Low-res input, shape [N, C, H, W]
        high_res (np.ndarray): High-res target, shape [N, 1, h, w]
        patch_size (int): Crop size
        augment (bool): Whether to apply random flips/rotations

    Example:
        >>> ds = SRDataset(np.random.rand(10,3,16,16), np.random.rand(10,1,32,32), patch_size=16)
        >>> x, y = ds[0]
    """
    def __init__(self, low_res: np.ndarray, high_res: np.ndarray, patch_size: int = 16, augment: bool = True):
        self.low_res = low_res
        self.high_res = high_res
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self) -> int:
        return self.low_res.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.low_res[idx]
        y = self.high_res[idx]
        # Random crop (for demo, center crop)
        x = torch.tensor(x[:, :self.patch_size, :self.patch_size], dtype=torch.float32)
        y = torch.tensor(y[:, :self.patch_size*2, :self.patch_size*2], dtype=torch.float32)
        # Augmentations
        if self.augment:
            if np.random.rand() > 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[2])
            if np.random.rand() > 0.5:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[1])
        return x, y