import argparse
import numpy as np
from src.train import train_sr_unet, load_sr_model
from src.models.sr_unet import set_seed
import os

def make_synthetic_sr_data(num_samples=8, in_channels=3, lr_shape=(16,16), hr_shape=(32,32)):
    """
    Make synthetic low-res and high-res arrays for SR training.
    """
    set_seed(42)
    low_res = np.random.rand(num_samples, in_channels, *lr_shape)
    high_res = np.random.rand(num_samples, 1, *hr_shape)
    return low_res, high_res

def main():
    parser = argparse.ArgumentParser(description="Train U-Net SR model.")
    parser.add_argument("--output", type=str, default="models/sr_unet_checkpoint.pth", help="Checkpoint path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop", type=int, default=3)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for demo")
    args = parser.parse_args()

    if args.synthetic:
        low_res, high_res = make_synthetic_sr_data(num_samples=10)
    else:
        # Load real arrays from disk (implement loading logic here)
        raise NotImplementedError("Non-synthetic mode: implement data loading.")

    config = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "early_stop": args.early_stop,
        "checkpoint_path": args.output,
        "patch_size": args.patch_size,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model = train_sr_unet(low_res, high_res, config)
    print(f"SR U-Net model trained and checkpoint saved to {args.output}")

if __name__ == "__main__":
    main()