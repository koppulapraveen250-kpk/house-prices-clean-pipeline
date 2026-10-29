"""Standalone clean-code preprocessing, training, and inference pipeline.

Implements both the modular `bpm_pipeline` workflow and a scikit-learn
`Pipeline` using `FunctionTransformer`, `SimpleImputer`, and `StandardScaler`
for the AIS Competition v3 (BeatsPerMinute prediction) dataset.
"""
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from bpm_pipeline import (
    FEATURE_COLUMNS,
    RAW_FEATURES,
)
from bpm_pipeline.inference import make_predictions
from bpm_pipeline.preprocess import engineer_audio_features
from bpm_pipeline.train import build_model


def _engineer_from_array(x_arr: np.ndarray) -> pd.DataFrame:
    """Convert imputed 2D array to DataFrame and compute engineered features."""
    df = pd.DataFrame(x_arr, columns=RAW_FEATURES)
    return engineer_audio_features(df)


def create_sklearn_preprocessing_pipeline() -> Pipeline:
    """Create a clean scikit-learn preprocessing Pipeline.

    Steps:
        1. Median imputation (`SimpleImputer`) on the 9 raw audio columns.
        2. Domain feature engineering (`FunctionTransformer`) producing 16 cols.
        3. Z-score standardization (`StandardScaler`).

    Returns:
        Unfitted scikit-learn preprocessing Pipeline.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "feature_engineer",
                FunctionTransformer(_engineer_from_array, validate=False),
            ),
            ("scaler", StandardScaler()),
        ]
    )


def create_full_pipeline() -> Pipeline:
    """Create a complete end-to-end Pipeline with preprocessing and regressor.

    Returns:
        Unfitted scikit-learn Pipeline.
    """
    return Pipeline(
        steps=[
            ("preprocessor", create_sklearn_preprocessing_pipeline()),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=150,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    min_samples_leaf=50,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def run_pipeline(
    train_filepath: str = "data/train.csv",
    test_filepath: str = "data/test.csv",
    output_filepath: str = "submission.csv",
) -> dict[str, Any]:
    """Execute the full training, evaluation, and submission generation flow.

    Args:
        train_filepath: Path to training CSV dataset.
        test_filepath: Path to test CSV dataset.
        output_filepath: Path where submission CSV will be written.

    Returns:
        Dictionary containing validation metrics and submission shape.
    """
    metrics = build_model(train_filepath, model_type="hgbr")
    preds = make_predictions(test_filepath, output_csv=output_filepath)
    return {
        "metrics": metrics,
        "submission_rows": int(preds.shape[0]),
        "num_features": len(FEATURE_COLUMNS),
    }


if __name__ == "__main__":
    results = run_pipeline()
    print("Pipeline execution complete:", results)
