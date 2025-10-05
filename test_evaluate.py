import pytest
import numpy as np
import pandas as pd
from src.evaluate import evaluate_against_stations, aggregate, bootstrap_metrics

def test_evaluate_against_stations_metrics():
    pred_grid = np.array([[1, 2], [3, 4]])
    station_df = pd.DataFrame({"lat":[0,0.5,1,1.5], "lon":[0,0.5,1,1.5], "pm25":[1,2,3,4]})
    grid_coords = np.array([[0,0],[0,1],[1,0],[1,1]])
    metrics = evaluate_against_stations(pred_grid, station_df, grid_coords)
    assert "RMSE" in metrics and "R2" in metrics

def test_aggregate_shape_and_values():
    pred_grid = np.arange(16).reshape(4,4)
    coarse = aggregate(pred_grid, 2)
    assert coarse.shape == (2,2)
    assert np.isclose(coarse[0,0], pred_grid[:2,:2].mean())

def test_bootstrap_metrics_runs():
    pred_grid = np.array([[1,2],[3,4]])
    station_df = pd.DataFrame({"lat":[0,0.5,1,1.5], "lon":[0,0.5,1,1.5], "pm25":[1,2,3,4]})
    grid_coords = np.array([[0,0],[0,1],[1,0],[1,1]])
    df = bootstrap_metrics(pred_grid, station_df, grid_coords, n_boot=5)
    assert df.shape[0] == 5