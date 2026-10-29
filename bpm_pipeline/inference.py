"""Inference module for generating BeatsPerMinute predictions."""
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from bpm_pipeline import ID_COLUMN, MODELS_DIR, RAW_FEATURES, TARGET_COLUMN
from bpm_pipeline.preprocess import preprocess


def _load_test_data(filepath: str) -> pd.DataFrame:
    """Load test CSV file and verify its existence."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Inference data file not found: {filepath}")
    return pd.read_csv(path)


def _load_model() -> Any:
    """Load persisted regression model from MODELS_DIR."""
    model_path = MODELS_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run training first."
        )
    return joblib.load(model_path)


def make_predictions(
    filepath: str, output_csv: str | None = None
) -> np.ndarray:
    """Load saved artifacts, preprocess test data, and predict BeatsPerMinute.

    Args:
        filepath: Path to input test.csv file.
        output_csv: Optional path to save Kaggle submission CSV (id, BeatsPerMinute).

    Returns:
        NumPy array of predicted BeatsPerMinute values.
    """
    df = _load_test_data(filepath)
    model = _load_model()
    x_test_proc = preprocess(df[RAW_FEATURES], is_training=False)
    predictions = np.asarray(model.predict(x_test_proc), dtype=float)

    if output_csv is not None and ID_COLUMN in df.columns:
        sub = pd.DataFrame(
            {ID_COLUMN: df[ID_COLUMN], TARGET_COLUMN: predictions}
        )
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(output_csv, index=False)

    return predictions
