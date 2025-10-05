import argparse
import numpy as np
from src.models.baseline import train_baseline, save_baseline_model
from src.features import make_feature_matrix
from tests.utils import make_synthetic_satellite

def main():
    parser = argparse.ArgumentParser(description="Train baseline RandomForest model.")
    parser.add_argument("--sat_path", type=str, default=None, help="Path to satellite data (npz/nc)")
    parser.add_argument("--era_path", type=str, default=None, help="Path to ERA5 data (npz/nc)")
    parser.add_argument("--output", type=str, default="models/baseline_rf.joblib", help="Path to save model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for demo")
    args = parser.parse_args()

    if args.synthetic:
        sat_ds = make_synthetic_satellite((4,4), variable='no2')
        era_ds = make_synthetic_satellite((4,4), variable='temp')
        grid = sat_ds
        X, y, coords = make_feature_matrix(sat_ds, era_ds, grid)
    else:
        # Load real datasets: implement loading logic here
        raise NotImplementedError("Non-synthetic mode: implement data loading.")

    model = train_baseline(X, y, seed=args.seed, n_estimators=args.n_estimators)
    save_baseline_model(model, args.output)
    print(f"Baseline RandomForest model saved to {args.output}")

if __name__ == "__main__":
    main()