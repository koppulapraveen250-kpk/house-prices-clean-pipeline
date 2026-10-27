"""Unit tests for the house_prices preprocessing, training, and inference pipeline."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import house_prices
import house_prices.inference as inference_mod
import house_prices.preprocess as preprocess_mod
import house_prices.train as train_mod
from house_prices import CONTINUOUS_FEATURES, FEATURE_COLUMNS
from house_prices.inference import make_predictions
from house_prices.preprocess import preprocess
from house_prices.train import build_model
from pipeline import create_full_pipeline, create_preprocessing_pipeline, run_pipeline


class TestPreprocessingPipeline(unittest.TestCase):
    """Test suite for validating the preprocessing and modeling pipeline."""

    def setUp(self) -> None:
        """Create a temporary models directory and synthetic housing data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.temp_dir.name) / "models"

        self.patcher_init = patch.object(
            house_prices, "MODELS_DIR", self.models_dir
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

        self.train_df = pd.DataFrame(
            {
                "Id": [1, 2, 3, 4, 5],
                "GrLivArea": [1500.0, 2000.0, 1200.0, 1800.0, 2500.0],
                "LotArea": [8000.0, 10000.0, 7500.0, 9500.0, 12000.0],
                "Neighborhood": [
                    "CollgCr",
                    "Veenker",
                    "CollgCr",
                    "Crawfor",
                    "NoRidge",
                ],
                "HouseStyle": [
                    "2Story",
                    "1Story",
                    "2Story",
                    "1Story",
                    "2Story",
                ],
                "SalePrice": [200000, 250000, 160000, 220000, 310000],
            }
        )

        self.test_df = pd.DataFrame(
            {
                "Id": [1461, 1462],
                "GrLivArea": [1600.0, 2100.0],
                "LotArea": [8500.0, 11000.0],
                "Neighborhood": ["CollgCr", "UnseenNeighborhood"],
                "HouseStyle": ["2Story", "UnseenStyle"],
            }
        )

    def tearDown(self) -> None:
        """Stop patches and clean up temporary directory."""
        self.patcher_init.stop()
        self.patcher_prep.stop()
        self.patcher_train.stop()
        self.patcher_inf.stop()
        self.temp_dir.cleanup()

    def test_preprocess_training_creates_artifacts_and_scales(self) -> None:
        """Verify training preprocess fits, saves artifacts, and standardizes."""
        processed = preprocess(
            self.train_df[FEATURE_COLUMNS], is_training=True
        )

        expected_artifacts = [
            "num_imputer.joblib",
            "cat_imputer.joblib",
            "scaler.joblib",
            "encoder.joblib",
        ]
        for artifact in expected_artifacts:
            self.assertTrue(
                (self.models_dir / artifact).exists(),
                f"Expected artifact {artifact} was not persisted.",
            )

        self.assertIsInstance(processed, pd.DataFrame)
        self.assertEqual(processed.shape[0], len(self.train_df))
        # 2 continuous + 4 unique neighborhoods + 2 unique house styles = 8 cols
        self.assertEqual(processed.shape[1], 8)
        self.assertFalse(processed.isnull().values.any())

        # Standardized continuous features should have ~0 mean and ~1 std
        for col in CONTINUOUS_FEATURES:
            self.assertAlmostEqual(float(processed[col].mean()), 0.0, places=5)
            self.assertAlmostEqual(
                float(processed[col].std(ddof=0)), 1.0, places=5
            )

    def test_preprocess_inference_handles_unseen_categories(self) -> None:
        """Verify inference mode loads artifacts and handles unseen categories."""
        train_proc = preprocess(
            self.train_df[FEATURE_COLUMNS], is_training=True
        )
        test_proc = preprocess(
            self.test_df[FEATURE_COLUMNS], is_training=False
        )

        self.assertEqual(test_proc.shape[0], len(self.test_df))
        self.assertEqual(test_proc.shape[1], train_proc.shape[1])
        self.assertEqual(list(test_proc.columns), list(train_proc.columns))
        self.assertFalse(test_proc.isnull().values.any())

        # Second row has unseen Neighborhood and HouseStyle -> one-hot all 0.0
        ohe_cols = [
            c for c in test_proc.columns if c not in CONTINUOUS_FEATURES
        ]
        self.assertTrue(np.allclose(test_proc.iloc[1][ohe_cols].values, 0.0))

    def test_preprocess_imputes_missing_values(self) -> None:
        """Verify missing numeric and categorical values are imputed cleanly."""
        preprocess(self.train_df[FEATURE_COLUMNS], is_training=True)

        df_with_nans = pd.DataFrame(
            {
                "GrLivArea": [np.nan, 1800.0],
                "LotArea": [9000.0, np.nan],
                "Neighborhood": [None, "CollgCr"],
                "HouseStyle": ["2Story", None],
            }
        )
        processed = preprocess(df_with_nans, is_training=False)
        self.assertFalse(processed.isnull().values.any())
        self.assertEqual(processed.shape[0], 2)

    def test_preprocess_does_not_mutate_input_dataframe(self) -> None:
        """Verify input DataFrame is not mutated during preprocessing."""
        raw_copy = self.train_df[FEATURE_COLUMNS].copy(deep=True)
        _ = preprocess(self.train_df[FEATURE_COLUMNS], is_training=True)
        pd.testing.assert_frame_equal(
            self.train_df[FEATURE_COLUMNS], raw_copy
        )

    def test_preprocess_inference_without_artifacts_raises_error(self) -> None:
        """Verify FileNotFoundError is raised when artifacts are missing."""
        with self.assertRaises(FileNotFoundError):
            preprocess(self.test_df[FEATURE_COLUMNS], is_training=False)

    def test_end_to_end_build_model_and_make_predictions(self) -> None:
        """Verify full training and inference pipeline on CSV files."""
        train_csv = Path(self.temp_dir.name) / "train.csv"
        test_csv = Path(self.temp_dir.name) / "test.csv"
        larger_train = pd.concat([self.train_df] * 4, ignore_index=True)
        larger_train.to_csv(train_csv, index=False)
        self.test_df.to_csv(test_csv, index=False)

        metrics = build_model(str(train_csv))
        self.assertIn("rmsle", metrics)
        self.assertIsInstance(metrics["rmsle"], float)
        self.assertGreaterEqual(metrics["rmsle"], 0.0)
        self.assertTrue((self.models_dir / "model.joblib").exists())

        preds = make_predictions(str(test_csv))
        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(preds.shape, (len(self.test_df),))

    def test_sklearn_column_transformer_and_full_pipeline(self) -> None:
        """Verify standalone sklearn ColumnTransformer and Pipeline match."""
        ct = create_preprocessing_pipeline()
        x_tr = ct.fit_transform(self.train_df[FEATURE_COLUMNS])
        x_te = ct.transform(self.test_df[FEATURE_COLUMNS])
        self.assertEqual(x_tr.shape, (len(self.train_df), 8))
        self.assertEqual(x_te.shape, (len(self.test_df), 8))

        full_pipe = create_full_pipeline()
        full_pipe.fit(
            self.train_df[FEATURE_COLUMNS], self.train_df["SalePrice"]
        )
        preds = full_pipe.predict(self.test_df[FEATURE_COLUMNS])
        self.assertEqual(preds.shape, (len(self.test_df),))

    def test_run_pipeline_generates_submission(self) -> None:
        """Verify run_pipeline writes valid submission CSV and returns metrics."""
        train_csv = Path(self.temp_dir.name) / "train.csv"
        test_csv = Path(self.temp_dir.name) / "test.csv"
        sub_csv = Path(self.temp_dir.name) / "submission.csv"

        larger_train = pd.concat([self.train_df] * 4, ignore_index=True)
        larger_train.to_csv(train_csv, index=False)
        self.test_df.to_csv(test_csv, index=False)

        with patch("pipeline.MODELS_DIR", self.models_dir):
            res = run_pipeline(str(train_csv), str(test_csv), str(sub_csv))

        self.assertIn("metrics", res)
        self.assertEqual(res["submission_rows"], len(self.test_df))
        self.assertTrue(sub_csv.exists())
        sub_df = pd.read_csv(sub_csv)
        self.assertEqual(list(sub_df.columns), ["Id", "SalePrice"])


if __name__ == "__main__":
    unittest.main()
