import pytest
import numpy as np
import os
from src.train import train_sr_unet, load_sr_model

def test_train_sr_unet_runs(tmp_path):
    low = np.random.rand(8, 3, 16, 16)
    high = np.random.rand(8, 1, 32, 32)
    config = {
        "batch_size": 2,
        "lr": 1e-3,
        "epochs": 3,
        "seed": 123,
        "early_stop": 2,
        "checkpoint_path": os.path.join(tmp_path, "sr_checkpoint.pth"),
        "patch_size": 16,
    }
    model = train_sr_unet(low, high, config)
    assert hasattr(model, "forward")
    assert os.path.exists(config["checkpoint_path"])
    loaded_model = load_sr_model(config["checkpoint_path"], in_channels=3, out_channels=1)
    assert hasattr(loaded_model, "forward")