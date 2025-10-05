import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import joblib
import logging

logger = logging.getLogger(__name__)

def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    n_estimators: int = 100
) -> RandomForestRegressor:
    """
    Train a baseline RandomForest model on feature matrix.

    Args:
        X_train (np.ndarray): Training features [n_samples, n_features]
        y_train (np.ndarray): Training targets [n_samples]
        seed (int): Random seed for reproducibility
        n_estimators (int): Number of trees

    Returns:
        RandomForestRegressor: Trained model

    Example:
        >>> model = train_baseline(X_train, y_train)
    """
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    logger.info("Baseline RandomForest trained.")
    return rf

def predict_baseline(
    model: RandomForestRegressor,
    X_val: np.ndarray
) -> np.ndarray:
    """
    Predict using trained baseline model.

    Args:
        model (RandomForestRegressor): Fitted model
        X_val (np.ndarray): Validation features [n_samples, n_features]

    Returns:
        np.ndarray: Predicted values [n_samples]

    Example:
        >>> y_pred = predict_baseline(model, X_val)
    """
    y_pred = model.predict(X_val)
    return y_pred

def save_baseline_model(model: RandomForestRegressor, path: str):
    """
    Save trained RandomForest model using joblib.

    Args:
        model (RandomForestRegressor): Trained model
        path (str): File path

    Example:
        >>> save_baseline_model(model, "rf_model.joblib")
    """
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")