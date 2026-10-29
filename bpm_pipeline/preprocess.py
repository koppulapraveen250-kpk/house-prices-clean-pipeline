"""Preprocessing and feature engineering module for AIS Competition v3."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from bpm_pipeline import (
    FEATURE_COLUMNS,
    MODELS_DIR,
    RAW_FEATURES,
)


def validate_input_columns(df: pd.DataFrame) -> None:
    """Verify that all required raw audio feature columns exist in DataFrame."""
    missing = [col for col in RAW_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required audio feature columns: {missing}")


def engineer_audio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific interaction and ratio features without mutation.

    Args:
        df: DataFrame containing imputed RAW_FEATURES.

    Returns:
        New DataFrame containing both RAW_FEATURES and ENGINEERED_FEATURES.
    """
    out = df[RAW_FEATURES].copy()
    eps = 1e-5
    out["Rhythm_Energy_Product"] = out["RhythmScore"] * out["Energy"]
    out["Mood_Rhythm_Product"] = out["MoodScore"] * out["RhythmScore"]
    out["Vocal_to_Instrumental_Ratio"] = out["VocalContent"] / (
        out["InstrumentalScore"] + eps
    )
    out["Acoustic_Energy_Ratio"] = out["AcousticQuality"] / (
        out["Energy"] + eps
    )
    out["TrackDurationMinutes"] = out["TrackDurationMs"] / 60000.0
    out["Loudness_Linear"] = np.power(10.0, out["AudioLoudness"] / 20.0)
    out["Energy_Mood_Sum"] = out["Energy"] + out["MoodScore"]
    return out[FEATURE_COLUMNS]


def clip_outliers(
    df: pd.DataFrame, bounds: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    """Clip extreme values in each column using fitted training quantile bounds."""
    clipped = df.copy()
    for col, (low, high) in bounds.items():
        if col in clipped.columns:
            clipped[col] = clipped[col].clip(lower=low, upper=high)
    return clipped


def compute_quantile_bounds(
    df: pd.DataFrame, lower_q: float = 0.005, upper_q: float = 0.995
) -> dict[str, tuple[float, float]]:
    """Compute per-column quantile clipping bounds on training data."""
    bounds: dict[str, tuple[float, float]] = {}
    for col in df.columns:
        low = float(df[col].quantile(lower_q))
        high = float(df[col].quantile(upper_q))
        bounds[col] = (low, high)
    return bounds


def _check_artifacts_exist(models_dir: Path) -> None:
    """Verify that saved preprocessing artifacts exist on disk."""
    required = ["imputer.joblib", "bounds.joblib", "scaler.joblib"]
    if not all((models_dir / name).exists() for name in required):
        raise FileNotFoundError(
            f"Preprocessing artifacts not found in {models_dir}. "
            "Run preprocess(..., is_training=True) first."
        )


def _fit_and_save_preprocessors(
    df_raw: pd.DataFrame,
) -> tuple[SimpleImputer, dict[str, tuple[float, float]], StandardScaler]:
    """Fit imputer, outlier bounds, and scaler on training data and save them."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    imputer = SimpleImputer(strategy="median").fit(df_raw[RAW_FEATURES])
    imputed_df = pd.DataFrame(
        imputer.transform(df_raw[RAW_FEATURES]),
        columns=RAW_FEATURES,
        index=df_raw.index,
    )
    engineered_df = engineer_audio_features(imputed_df)
    bounds = compute_quantile_bounds(engineered_df)
    clipped_df = clip_outliers(engineered_df, bounds)
    scaler = StandardScaler().fit(clipped_df[FEATURE_COLUMNS])

    joblib.dump(imputer, MODELS_DIR / "imputer.joblib")
    joblib.dump(bounds, MODELS_DIR / "bounds.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    return imputer, bounds, scaler


def _load_preprocessors() -> (
    tuple[SimpleImputer, dict[str, tuple[float, float]], StandardScaler]
):
    """Load persisted imputer, outlier bounds, and scaler artifacts."""
    _check_artifacts_exist(MODELS_DIR)
    imputer: SimpleImputer = joblib.load(MODELS_DIR / "imputer.joblib")
    bounds: dict[str, tuple[float, float]] = joblib.load(
        MODELS_DIR / "bounds.joblib"
    )
    scaler: StandardScaler = joblib.load(MODELS_DIR / "scaler.joblib")
    return imputer, bounds, scaler


def preprocess(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """Preprocess raw audio features for training or inference.

    Steps:
        1. Validate that all 9 raw audio columns are present.
        2. Impute missing values using training medians (SimpleImputer).
        3. Engineer 7 domain interaction/ratio audio features (16 total).
        4. Clip extreme outliers using training [0.5%, 99.5%] quantile bounds.
        5. Standardize all 16 features to zero mean and unit variance.

    Args:
        df: Raw input DataFrame containing at least RAW_FEATURES.
        is_training: If True, fit and persist transformers on df.
            If False, load persisted transformers and transform df.

    Returns:
        Clean, standardized DataFrame with 16 columns and preserved index.
    """
    validate_input_columns(df)
    if is_training:
        imputer, bounds, scaler = _fit_and_save_preprocessors(df)
    else:
        imputer, bounds, scaler = _load_preprocessors()

    imputed_df = pd.DataFrame(
        imputer.transform(df[RAW_FEATURES]),
        columns=RAW_FEATURES,
        index=df.index,
    )
    engineered_df = engineer_audio_features(imputed_df)
    clipped_df = clip_outliers(engineered_df, bounds)
    scaled_arr = scaler.transform(clipped_df[FEATURE_COLUMNS])
    return pd.DataFrame(scaled_arr, columns=FEATURE_COLUMNS, index=df.index)
