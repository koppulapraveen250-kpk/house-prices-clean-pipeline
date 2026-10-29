"""Training and evaluation module for AIS Competition v3 (BeatsPerMinute)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from bpm_pipeline import MODELS_DIR, RAW_FEATURES, TARGET_COLUMN
from bpm_pipeline.preprocess import preprocess


def _load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV file and verify its existence."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Training file not found at: {filepath}")
    return pd.read_csv(path)


def _split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split raw dataset into training and validation subsets before fitting."""
    x = df[RAW_FEATURES]
    y = df[TARGET_COLUMN]
    return train_test_split(x, y, test_size=0.2, random_state=42)


def _compute_metrics(
    y_true: pd.Series | np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute RMSE, MAE, and R2 evaluation metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 6),
    }


def build_model(
    filepath: str, model_type: str = "hgbr"
) -> dict[str, float]:
    """Train model on preprocessed features and return validation metrics.

    Args:
        filepath: Path to training CSV file.
        model_type: Either 'hgbr' (HistGradientBoostingRegressor) or 'ridge'.

    Returns:
        Dictionary of validation metrics (rmse, mae, r2).
    """
    df = _load_csv(filepath)
    x_train, x_val, y_train, y_val = _split_dataset(df)

    x_train_proc = preprocess(x_train, is_training=True)
    x_val_proc = preprocess(x_val, is_training=False)

    if model_type == "ridge":
        model = Ridge(alpha=10.0, random_state=42)
    else:
        model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=42,
        )

    model.fit(x_train_proc, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.joblib")

    val_preds = model.predict(x_val_proc)
    return _compute_metrics(y_val, val_preds)
