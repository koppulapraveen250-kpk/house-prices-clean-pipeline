"""Standalone clean-code preprocessing, training, and inference pipeline.

This script provides a complete, self-contained sklearn Pipeline and
ColumnTransformer implementation alongside the modular `house_prices` package.
"""
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from house_prices import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    MODELS_DIR,
)


def create_preprocessing_pipeline(
    continuous_features: list[str] = CONTINUOUS_FEATURES,
    categorical_features: list[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Create a clean scikit-learn ColumnTransformer for preprocessing.

    Args:
        continuous_features: List of numerical column names.
        categorical_features: List of categorical column names.

    Returns:
        Unfitted ColumnTransformer combining imputation, scaling, and encoding.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, continuous_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def create_full_pipeline(
    continuous_features: list[str] = CONTINUOUS_FEATURES,
    categorical_features: list[str] = CATEGORICAL_FEATURES,
) -> Pipeline:
    """Create a complete end-to-end Pipeline with preprocessing and regressor.

    Args:
        continuous_features: List of numerical column names.
        categorical_features: List of categorical column names.

    Returns:
        Unfitted scikit-learn Pipeline.
    """
    preprocessor = create_preprocessing_pipeline(
        continuous_features, categorical_features
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def evaluate_predictions(
    y_true: pd.Series | np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute regression evaluation metrics (RMSLE, RMSE, MAE, R2).

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary of rounded evaluation metrics.
    """
    clipped = np.clip(y_pred, a_min=1.0, a_max=None)
    rmsle = float(np.sqrt(mean_squared_log_error(y_true, clipped)))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "rmsle": round(rmsle, 4),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
    }


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
    train_df = pd.read_csv(train_filepath)
    test_df = pd.read_csv(test_filepath)

    x = train_df[FEATURE_COLUMNS]
    y = train_df[LABEL_COLUMN]
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    pipeline = create_full_pipeline()
    pipeline.fit(x_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "full_pipeline.joblib")

    val_preds = pipeline.predict(x_val)
    metrics = evaluate_predictions(y_val, val_preds)

    test_preds = np.clip(
        pipeline.predict(test_df[FEATURE_COLUMNS]), a_min=0.0, a_max=None
    )
    submission = pd.DataFrame({"Id": test_df["Id"], "SalePrice": test_preds})
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_filepath, index=False)

    return {"metrics": metrics, "submission_rows": len(submission)}


if __name__ == "__main__":
    results = run_pipeline()
    print("Pipeline execution complete:", results)
