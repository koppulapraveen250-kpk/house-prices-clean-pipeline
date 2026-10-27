"""Inference module for house price predictions."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from house_prices import FEATURE_COLUMNS, MODELS_DIR
from house_prices.preprocess import preprocess


def _load_inference_data(filepath: str) -> pd.DataFrame:
    """Load inference input CSV file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Inference data file not found: {filepath}")
    return pd.read_csv(path)


def _load_model() -> LinearRegression:
    """Load persisted linear regression model artifact."""
    model_path = MODELS_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run training first."
        )
    return joblib.load(model_path)


def make_predictions(filepath: str) -> np.ndarray:
    """Load model and return predictions for input data.

    Args:
        filepath: Path to input CSV file.

    Returns:
        Array of predicted house prices.
    """
    df = _load_inference_data(filepath)
    model = _load_model()
    processed_features = preprocess(df[FEATURE_COLUMNS], is_training=False)
    predictions: np.ndarray = model.predict(processed_features)
    return predictions
