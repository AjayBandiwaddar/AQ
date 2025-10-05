import pytest
import numpy as np
from src.models.baseline import train_baseline, predict_baseline

def test_train_baseline_reproducibility():
    X = np.random.rand(10, 4)
    y = np.random.rand(10)
    model1 = train_baseline(X, y, seed=123)
    model2 = train_baseline(X, y, seed=123)
    y_pred1 = predict_baseline(model1, X)
    y_pred2 = predict_baseline(model2, X)
    assert np.allclose(y_pred1, y_pred2)

def test_predict_baseline_shape():
    X = np.random.rand(10, 3)
    y = np.random.rand(10)
    model = train_baseline(X, y)
    y_pred = predict_baseline(model, X)
    assert y_pred.shape == (10,)