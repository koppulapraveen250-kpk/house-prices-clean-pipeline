"""Preprocessing module for feature imputation, scaling, and encoding."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from house_prices import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    MODELS_DIR,
)


def _check_models_exist(models_dir: Path) -> None:
    """Verify saved imputer, scaler, and encoder artifacts exist."""
    artifacts = [
        "num_imputer.joblib",
        "cat_imputer.joblib",
        "scaler.joblib",
        "encoder.joblib",
    ]
    if not all((models_dir / name).exists() for name in artifacts):
        raise FileNotFoundError(
            f"Transformers not found in {models_dir}. Run training first."
        )


def _fit_and_save_imputers(
    df: pd.DataFrame,
) -> tuple[SimpleImputer, SimpleImputer]:
    """Fit numeric and categorical imputers on training data and save them."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    num_imp = SimpleImputer(strategy="median").fit(df[CONTINUOUS_FEATURES])
    cat_imp = SimpleImputer(strategy="most_frequent").fit(
        df[CATEGORICAL_FEATURES]
    )
    joblib.dump(num_imp, MODELS_DIR / "num_imputer.joblib")
    joblib.dump(cat_imp, MODELS_DIR / "cat_imputer.joblib")
    return num_imp, cat_imp


def _load_imputers() -> tuple[SimpleImputer, SimpleImputer]:
    """Load persisted numeric and categorical imputer artifacts."""
    _check_models_exist(MODELS_DIR)
    num_imp: SimpleImputer = joblib.load(MODELS_DIR / "num_imputer.joblib")
    cat_imp: SimpleImputer = joblib.load(MODELS_DIR / "cat_imputer.joblib")
    return num_imp, cat_imp


def _apply_imputers(
    df: pd.DataFrame, num_imp: SimpleImputer, cat_imp: SimpleImputer
) -> pd.DataFrame:
    """Apply fitted imputers to continuous and categorical columns."""
    df_clean = df.copy()
    df_clean[CONTINUOUS_FEATURES] = num_imp.transform(df[CONTINUOUS_FEATURES])
    df_clean[CATEGORICAL_FEATURES] = cat_imp.transform(
        df[CATEGORICAL_FEATURES]
    )
    return df_clean


def _fit_and_save_transformers(
    df: pd.DataFrame,
) -> tuple[StandardScaler, OneHotEncoder]:
    """Fit scaler and encoder on training data and persist them."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler().fit(df[CONTINUOUS_FEATURES])
    encoder = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    ).fit(df[CATEGORICAL_FEATURES])
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(encoder, MODELS_DIR / "encoder.joblib")
    return scaler, encoder


def _load_transformers() -> tuple[StandardScaler, OneHotEncoder]:
    """Load persisted scaler and encoder artifacts."""
    _check_models_exist(MODELS_DIR)
    scaler: StandardScaler = joblib.load(MODELS_DIR / "scaler.joblib")
    encoder: OneHotEncoder = joblib.load(MODELS_DIR / "encoder.joblib")
    return scaler, encoder


def _transform_features(
    df: pd.DataFrame, scaler: StandardScaler, encoder: OneHotEncoder
) -> pd.DataFrame:
    """Transform continuous and categorical features into a DataFrame."""
    x_cont = scaler.transform(df[CONTINUOUS_FEATURES])
    x_cat = encoder.transform(df[CATEGORICAL_FEATURES])
    col_names = CONTINUOUS_FEATURES + list(
        encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    )
    stacked = np.hstack([x_cont, x_cat])
    return pd.DataFrame(stacked, columns=col_names, index=df.index)


def preprocess(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """Preprocess features for training or inference.

    Args:
        df: Raw dataframe with feature columns.
        is_training: If True, fit and save transformers.
            If False, load saved transformers.

    Returns:
        Preprocessed dataframe ready for model.
    """
    if is_training:
        num_imp, cat_imp = _fit_and_save_imputers(df)
        df_clean = _apply_imputers(df, num_imp, cat_imp)
        scaler, encoder = _fit_and_save_transformers(df_clean)
    else:
        num_imp, cat_imp = _load_imputers()
        df_clean = _apply_imputers(df, num_imp, cat_imp)
        scaler, encoder = _load_transformers()
    return _transform_features(df_clean, scaler, encoder)
