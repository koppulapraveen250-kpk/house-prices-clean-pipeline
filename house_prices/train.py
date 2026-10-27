"""Training module for building and evaluating house price model."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

from house_prices import FEATURE_COLUMNS, LABEL_COLUMN, MODELS_DIR
from house_prices.preprocess import preprocess


def _load_training_data(filepath: str) -> pd.DataFrame:
    """Load training data from CSV file path."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found at: {filepath}")
    return pd.read_csv(path)


def _split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into training and validation sets."""
    x = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]
    return train_test_split(x, y, test_size=0.2, random_state=42)


def _train_and_save_model(
    x_train: pd.DataFrame, y_train: pd.Series
) -> LinearRegression:
    """Fit linear regression model and persist it."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = LinearRegression()
    model.fit(x_train, y_train)
    joblib.dump(model, MODELS_DIR / "model.joblib")
    return model


def _compute_rmsle(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Compute root mean squared logarithmic error score."""
    clipped_pred = np.clip(y_pred, a_min=1, a_max=None)
    rmsle = np.sqrt(mean_squared_log_error(y_true.values, clipped_pred))
    return round(float(rmsle), 2)


def _evaluate_model(
    model: LinearRegression, x_val: pd.DataFrame, y_val: pd.Series
) -> float:
    """Evaluate trained model on validation set."""
    predictions = model.predict(x_val)
    return _compute_rmsle(y_val, predictions)


def build_model(filepath: str) -> dict[str, float]:
    """Train model on data and return evaluation metrics.

    Args:
        filepath: Path to training CSV file.

    Returns:
        Dictionary with metric names as keys and values as floats.
    """
    df = _load_training_data(filepath)
    x_tr, x_val, y_tr, y_val = _split_data(df)
    x_tr_proc = preprocess(x_tr, is_training=True)
    model = _train_and_save_model(x_tr_proc, y_tr)
    x_val_proc = preprocess(x_val, is_training=False)
    score = _evaluate_model(model, x_val_proc, y_val)
    return {"rmsle": score}
