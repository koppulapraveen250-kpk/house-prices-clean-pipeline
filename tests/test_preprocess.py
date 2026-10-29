"""Unit tests for the BeatsPerMinute (AIS Competition v3) preprocessing pipeline."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import bpm_pipeline
import bpm_pipeline.inference as inference_mod
import bpm_pipeline.preprocess as preprocess_mod
import bpm_pipeline.train as train_mod
from bpm_pipeline import (
    ENGINEERED_FEATURES,
    FEATURE_COLUMNS,
    RAW_FEATURES,
    TARGET_COLUMN,
)
from bpm_pipeline.inference import make_predictions
from bpm_pipeline.preprocess import (
    clip_outliers,
    compute_quantile_bounds,
    engineer_audio_features,
    preprocess,
    validate_input_columns,
)
from bpm_pipeline.train import build_model
from pipeline import (
    create_full_pipeline,
    create_sklearn_preprocessing_pipeline,
    run_pipeline,
)


class TestBPMPreprocessingPipeline(unittest.TestCase):
    """Test suite verifying all invariants of the audio preprocessing pipeline."""

    def setUp(self) -> None:
        """Create an isolated temporary models directory and synthetic audio data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.temp_dir.name) / "models"

        self.patcher_init = patch.object(
            bpm_pipeline, "MODELS_DIR", self.models_dir
        )
        self.patcher_prep = patch.object(
            preprocess_mod, "MODELS_DIR", self.models_dir
        )
        self.patcher_train = patch.object(
            train_mod, "MODELS_DIR", self.models_dir
        )
        self.patcher_inf = patch.object(
            inference_mod, "MODELS_DIR", self.models_dir
        )

        self.patcher_init.start()
        self.patcher_prep.start()
        self.patcher_train.start()
        self.patcher_inf.start()

        rng = np.random.default_rng(42)
        n_train = 120
        self.train_df = pd.DataFrame(
            {
                "id": np.arange(n_train),
                "RhythmScore": rng.uniform(0.1, 0.95, n_train),
                "AudioLoudness": rng.uniform(-25.0, -1.5, n_train),
                "VocalContent": rng.uniform(0.02, 0.25, n_train),
                "AcousticQuality": rng.uniform(0.001, 0.95, n_train),
                "InstrumentalScore": rng.uniform(0.001, 0.85, n_train),
                "LivePerformanceLikelihood": rng.uniform(0.02, 0.58, n_train),
                "MoodScore": rng.uniform(0.05, 0.95, n_train),
                "TrackDurationMs": rng.uniform(70000.0, 420000.0, n_train),
                "Energy": rng.uniform(0.01, 0.99, n_train),
                "BeatsPerMinute": rng.uniform(60.0, 180.0, n_train),
            }
        )

        n_test = 25
        self.test_df = pd.DataFrame(
            {
                "id": np.arange(1000, 1000 + n_test),
                "RhythmScore": rng.uniform(0.1, 0.95, n_test),
                "AudioLoudness": rng.uniform(-25.0, -1.5, n_test),
                "VocalContent": rng.uniform(0.02, 0.25, n_test),
                "AcousticQuality": rng.uniform(0.001, 0.95, n_test),
                "InstrumentalScore": rng.uniform(0.001, 0.85, n_test),
                "LivePerformanceLikelihood": rng.uniform(0.02, 0.58, n_test),
                "MoodScore": rng.uniform(0.05, 0.95, n_test),
                "TrackDurationMs": rng.uniform(70000.0, 420000.0, n_test),
                "Energy": rng.uniform(0.01, 0.99, n_test),
            }
        )

    def tearDown(self) -> None:
        """Stop patches and clean up temporary directory."""
        self.patcher_init.stop()
        self.patcher_prep.stop()
        self.patcher_train.stop()
        self.patcher_inf.stop()
        self.temp_dir.cleanup()

    def test_validate_input_columns_raises_on_missing_feature(self) -> None:
        """Verify ValueError is raised when required raw columns are missing."""
        incomplete_df = self.train_df.drop(columns=["RhythmScore"])
        with self.assertRaises(ValueError):
            validate_input_columns(incomplete_df)

    def test_engineer_audio_features_math_and_columns(self) -> None:
        """Verify domain feature engineering formulas and output schema."""
        engineered = engineer_audio_features(self.train_df[RAW_FEATURES])
        self.assertEqual(list(engineered.columns), FEATURE_COLUMNS)
        self.assertEqual(engineered.shape, (len(self.train_df), 16))
        for col in ENGINEERED_FEATURES:
            self.assertIn(col, engineered.columns)

        # Check exact formula for TrackDurationMinutes and Rhythm_Energy_Product
        np.testing.assert_allclose(
            engineered["TrackDurationMinutes"].values,
            self.train_df["TrackDurationMs"].values / 60000.0,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            engineered["Rhythm_Energy_Product"].values,
            (self.train_df["RhythmScore"] * self.train_df["Energy"]).values,
            rtol=1e-6,
        )

    def test_preprocess_training_creates_artifacts_and_standardizes(
        self,
    ) -> None:
        """Verify training preprocess saves artifacts and standardizes to N(0,1)."""
        processed = preprocess(
            self.train_df[RAW_FEATURES], is_training=True
        )

        for artifact in ["imputer.joblib", "bounds.joblib", "scaler.joblib"]:
            self.assertTrue(
                (self.models_dir / artifact).exists(),
                f"Expected artifact {artifact} was not created.",
            )

        self.assertEqual(processed.shape, (len(self.train_df), 16))
        self.assertEqual(list(processed.columns), FEATURE_COLUMNS)
        self.assertFalse(processed.isnull().values.any())

        # Every standardized column should have ~0 mean and ~1 std
        for col in FEATURE_COLUMNS:
            self.assertAlmostEqual(float(processed[col].mean()), 0.0, places=5)
            self.assertAlmostEqual(
                float(processed[col].std(ddof=0)), 1.0, places=5
            )

    def test_preprocess_imputes_missing_nan_values(self) -> None:
        """Verify NaN values in raw features are imputed without errors."""
        preprocess(self.train_df[RAW_FEATURES], is_training=True)

        dirty_test = self.test_df[RAW_FEATURES].copy()
        dirty_test.iloc[0, 0] = np.nan  # RhythmScore NaN
        dirty_test.iloc[1, 3] = np.nan  # AcousticQuality NaN
        dirty_test.iloc[2, 7] = np.nan  # TrackDurationMs NaN

        processed = preprocess(dirty_test, is_training=False)
        self.assertFalse(processed.isnull().values.any())
        self.assertTrue(np.isfinite(processed.values).all())
        self.assertEqual(processed.shape, (len(dirty_test), 16))

    def test_outlier_clipping_bounds_extreme_test_spikes(self) -> None:
        """Verify extreme out-of-distribution test values are clipped safely."""
        engineered = engineer_audio_features(self.train_df[RAW_FEATURES])
        bounds = compute_quantile_bounds(
            engineered, lower_q=0.05, upper_q=0.95
        )

        extreme_df = engineered.iloc[:2].copy()
        extreme_df.loc[extreme_df.index[0], "TrackDurationMs"] = 99999999.0
        extreme_df.loc[extreme_df.index[1], "AudioLoudness"] = -999.0

        clipped = clip_outliers(extreme_df, bounds)
        low_dur, high_dur = bounds["TrackDurationMs"]
        low_loud, high_loud = bounds["AudioLoudness"]

        self.assertLessEqual(
            float(clipped.iloc[0]["TrackDurationMs"]), high_dur
        )
        self.assertGreaterEqual(
            float(clipped.iloc[1]["AudioLoudness"]), low_loud
        )

    def test_preprocess_does_not_mutate_input_dataframe(self) -> None:
        """Verify input DataFrame is never modified in place."""
        original = self.train_df[RAW_FEATURES].copy(deep=True)
        _ = preprocess(self.train_df[RAW_FEATURES], is_training=True)
        pd.testing.assert_frame_equal(self.train_df[RAW_FEATURES], original)

    def test_preprocess_inference_without_artifacts_raises_error(self) -> None:
        """Verify FileNotFoundError is raised if inference runs before training."""
        with self.assertRaises(FileNotFoundError):
            preprocess(self.test_df[RAW_FEATURES], is_training=False)

    def test_end_to_end_build_model_and_make_predictions(self) -> None:
        """Verify full training, evaluation, and submission CSV generation."""
        train_csv = Path(self.temp_dir.name) / "train.csv"
        test_csv = Path(self.temp_dir.name) / "test.csv"
        sub_csv = Path(self.temp_dir.name) / "submission.csv"

        self.train_df.to_csv(train_csv, index=False)
        self.test_df.to_csv(test_csv, index=False)

        metrics = build_model(str(train_csv), model_type="hgbr")
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)
        self.assertTrue((self.models_dir / "model.joblib").exists())

        preds = make_predictions(str(test_csv), output_csv=str(sub_csv))
        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(preds.shape, (len(self.test_df),))
        self.assertTrue(sub_csv.exists())

        sub_df = pd.read_csv(sub_csv)
        self.assertEqual(list(sub_df.columns), ["id", TARGET_COLUMN])
        self.assertEqual(len(sub_df), len(self.test_df))

    def test_sklearn_pipeline_and_run_pipeline(self) -> None:
        """Verify standalone scikit-learn Pipeline and run_pipeline wrapper."""
        sk_prep = create_sklearn_preprocessing_pipeline()
        x_tr = sk_prep.fit_transform(self.train_df[RAW_FEATURES])
        x_te = sk_prep.transform(self.test_df[RAW_FEATURES])
        self.assertEqual(x_tr.shape, (len(self.train_df), 16))
        self.assertEqual(x_te.shape, (len(self.test_df), 16))

        full_pipe = create_full_pipeline()
        full_pipe.fit(
            self.train_df[RAW_FEATURES], self.train_df[TARGET_COLUMN]
        )
        preds = full_pipe.predict(self.test_df[RAW_FEATURES])
        self.assertEqual(preds.shape, (len(self.test_df),))

        train_csv = Path(self.temp_dir.name) / "train2.csv"
        test_csv = Path(self.temp_dir.name) / "test2.csv"
        sub_csv = Path(self.temp_dir.name) / "sub2.csv"
        self.train_df.to_csv(train_csv, index=False)
        self.test_df.to_csv(test_csv, index=False)

        res = run_pipeline(str(train_csv), str(test_csv), str(sub_csv))
        self.assertEqual(res["submission_rows"], len(self.test_df))
        self.assertEqual(res["num_features"], 16)


if __name__ == "__main__":
    unittest.main()
