## Baseline Model Training (RandomForest)

Run on synthetic data:

```bash
python scripts/train_baseline.py --synthetic --output models/baseline_rf.joblib
```

## Super-Resolution U-Net Training

Run on synthetic data:

```bash
python scripts/train_sr.py --synthetic --output models/sr_unet_checkpoint.pth --epochs 10 --batch_size 4
```

## What Happens?

- Both scripts generate synthetic data, train the respective models, and save a model checkpoint.
- To use real data, replace the synthetic flag and implement data loading logic.