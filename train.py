import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.sr_unet import UNetSR, SRDataset, set_seed
import logging
import numpy as np

logger = logging.getLogger(__name__)

def train_sr_unet(
    low_res: np.ndarray,
    high_res: np.ndarray,
    config: dict
) -> UNetSR:
    """
    Train the U-Net SR model with early stopping and checkpointing.

    Args:
        low_res (np.ndarray): Low-res input [N, C, H, W]
        high_res (np.ndarray): High-res target [N, 1, h, w]
        config (dict): Training config (batch_size, lr, epochs, seed, early_stop, checkpoint_path)

    Returns:
        UNetSR: Trained model

    Example:
        >>> model = train_sr_unet(low_res, high_res, config)
    """
    set_seed(config.get("seed",42))
    model = UNetSR(in_channels=low_res.shape[1], out_channels=1)
    ds = SRDataset(low_res, high_res, patch_size=config.get("patch_size",16), augment=True)
    dl = DataLoader(ds, batch_size=config.get("batch_size",4), shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=config.get("lr",1e-3))
    criterion = torch.nn.MSELoss()
    best_loss = float("inf")
    patience = config.get("early_stop",5)
    wait = 0
    for epoch in range(config.get("epochs",20)):
        model.train()
        epoch_loss = 0
        for x, y in dl:
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dl)
        logger.info(f"Epoch {epoch+1}: Loss {epoch_loss:.4f}")
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            wait = 0
            if "checkpoint_path" in config:
                torch.save(model.state_dict(), config["checkpoint_path"])
                logger.info(f"Checkpoint saved to {config['checkpoint_path']}")
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping triggered.")
                break
    return model

def load_sr_model(model_path: str, in_channels: int = 3, out_channels: int = 1) -> UNetSR:
    """
    Load trained U-Net SR model from checkpoint.

    Args:
        model_path (str): Path to checkpoint
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels

    Returns:
        UNetSR: Loaded model

    Example:
        >>> model = load_sr_model('checkpoint.pth')
    """
    model = UNetSR(in_channels, out_channels)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model