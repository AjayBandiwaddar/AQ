import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import logging
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

def evaluate_against_stations(
    pred_grid: np.ndarray,
    station_df: pd.DataFrame,
    grid_coords: np.ndarray,
    variable: str = "pm25"
) -> Dict[str, float]:
    """
    Evaluate predicted grid against ground truth stations using standard metrics.

    Args:
        pred_grid (np.ndarray): Predicted grid [H, W]
        station_df (pd.DataFrame): DataFrame with columns ['lat','lon',variable]
        grid_coords (np.ndarray): Grid coordinates [n_cells,2]
        variable (str): Variable to compare (default: 'pm25')

    Returns:
        dict: Metrics (RMSE, MAE, R2, Pearson r)

    Example:
        >>> metrics = evaluate_against_stations(pred_grid, station_df, grid_coords)
    """
    if len(station_df) == 0:
        raise ValueError("No ground stations found for bbox/date range.")
    # Find nearest grid cell for each station
    preds, obs = [], []
    for _, row in station_df.iterrows():
        dists = np.linalg.norm(grid_coords - np.array([row['lat'], row['lon']]), axis=1)
        idx = np.argmin(dists)
        pred_value = pred_grid.flat[idx]
        preds.append(pred_value)
        obs.append(row[variable])
    preds = np.array(preds)
    obs = np.array(obs)
    rmse = np.sqrt(np.mean((preds - obs)**2))
    mae = np.mean(np.abs(preds - obs))
    r2 = 1 - np.sum((obs - preds)**2) / np.sum((obs - np.mean(obs))**2)
    r, _ = pearsonr(preds, obs)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Pearson_r": r}

def aggregate(pred_grid: np.ndarray, factor: int) -> np.ndarray:
    """
    Aggregate high-resolution grid back to coarse grid by block averaging.

    Args:
        pred_grid (np.ndarray): Fine grid [H, W]
        factor (int): Aggregation factor (e.g., 4 for 16x16 -> 4x4)

    Returns:
        np.ndarray: Aggregated coarse grid

    Example:
        >>> coarse = aggregate(pred_grid, 4)
    """
    h, w = pred_grid.shape
    assert h % factor == 0 and w % factor == 0
    coarse = pred_grid.reshape(h//factor, factor, w//factor, factor).mean(axis=(1,3))
    return coarse

def bootstrap_metrics(
    pred_grid: np.ndarray,
    station_df: pd.DataFrame,
    grid_coords: np.ndarray,
    variable: str = "pm25",
    n_boot: int = 100
) -> pd.DataFrame:
    """
    Compute bootstrapped metrics for uncertainty estimation.

    Args:
        pred_grid (np.ndarray): Predicted grid [H, W]
        station_df (pd.DataFrame): DataFrame with columns ['lat','lon',variable]
        grid_coords (np.ndarray): Grid coordinates [n_cells,2]
        variable (str): Variable to compare
        n_boot (int): Number of bootstrap samples

    Returns:
        pd.DataFrame: Bootstrapped metrics

    Example:
        >>> df = bootstrap_metrics(pred_grid, station_df, grid_coords)
    """
    metrics = []
    n = len(station_df)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        df_sample = station_df.iloc[idx]
        metr = evaluate_against_stations(pred_grid, df_sample, grid_coords, variable)
        metrics.append(metr)
    return pd.DataFrame(metrics)

def plot_pred_vs_obs(preds: np.ndarray, obs: np.ndarray, out_path: str):
    """
    Simple scatter plot of predictions vs observations.

    Args:
        preds (np.ndarray): Predicted values
        obs (np.ndarray): Observed values
        out_path (str): Path to save PNG

    Example:
        >>> plot_pred_vs_obs(preds, obs, "results/scatter.png")
    """
    plt.figure(figsize=(6,6))
    plt.scatter(obs, preds, alpha=0.7)
    plt.plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'r--')
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title("Prediction vs Observation")
    plt.savefig(out_path)
    plt.close()

def save_metrics_csv(metrics: Dict[str, float], out_path: str):
    """
    Save evaluation metrics to CSV.

    Args:
        metrics (dict): Metrics dictionary
        out_path (str): File path

    Example:
        >>> save_metrics_csv({"RMSE":1.2, "MAE":0.7}, "results/metrics.csv")
    """
    df = pd.DataFrame([metrics])
    df.to_csv(out_path, index=False)