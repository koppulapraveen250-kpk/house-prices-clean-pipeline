# Project Report: Clean Code Preprocessing Pipeline & Unit Testing (`ais-competition-v-3`)

**Competition:** Kaggle — *AIS Competition v3 (Predicting `BeatsPerMinute`)*  
**Competition Link:** `https://www.kaggle.com/t/31f8ecfe2319647799227b9c7f0b9018`  
**Author:** Praveen Kumar Koppula  
**Repository:** [`koppulapraveen250-kpk/kaggle-competition-2`](https://github.com/koppulapraveen250-kpk/kaggle-competition-2)

---

## 1. Executive Summary

This report presents the design, clean-code implementation, and automated unit testing of our data preprocessing and regression pipeline for **Kaggle Competition 2 (`ais-competition-v-3`)**, where the objective is to predict the tempo (**`BeatsPerMinute`**) of musical tracks from acoustic descriptors.

Our work delivers:
1. **A Modular Clean-Code Pipeline (`bpm_pipeline/` & `pipeline.py`):** Strictly separates preprocessing, training, and inference, splits data *prior* to fitting any transformers to guarantee zero data leakage, persists fitted artifacts (`imputer.joblib`, `bounds.joblib`, `scaler.joblib`, `model.joblib`), and adheres to PEP 8 (`0` `flake8` warnings/errors) with full type annotations and Google-style docstrings.
2. **Domain-Driven Audio Feature Engineering:** Expands the $9$ raw acoustic attributes into $16$ standardized features capturing non-linear physical conversions (decibel-to-linear amplitude), tempo-energy couplings, vocal-to-instrumental ratios, and quantile-based outlier clipping.
3. **Comprehensive Unit Test Suite (`tests/test_preprocess.py`):** $9$ unit tests covering schema validation, feature engineering mathematics, zero-mean/unit-variance scaling, missing-value imputation, out-of-distribution outlier clipping, input DataFrame immutability, missing-artifact error handling, and end-to-end training/inference.

---

## 2. Exploratory Data Analysis & Dataset Overview

The dataset contains acoustic measurements across $698,886$ total tracks:
- **Training Set (`data/train.csv`):** $524,164$ rows $\times$ $11$ columns (`id`, $9$ acoustic features, and target `BeatsPerMinute`).
- **Test Set (`data/test.csv`):** $174,722$ rows $\times$ $10$ columns (`id` and $9$ acoustic features).
- **Sample Submission (`data/sample_submission.csv`):** $174,722$ rows $\times$ $2$ columns (`id, BeatsPerMinute`).

### Summary Statistics of Training Data ($N = 524,164$)

| Column | Role | Min | 25% | Median (50%) | Mean | 75% | Max | Std Dev |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `RhythmScore` | Raw Feature | 0.0769 | 0.5158 | 0.6347 | 0.6328 | 0.7392 | 0.9750 | 0.1569 |
| `AudioLoudness` | Raw Feature (dB) | -27.5097 | -11.5519 | -8.2525 | -8.3790 | -4.9123 | -1.3570 | 4.6162 |
| `VocalContent` | Raw Feature | 0.0235 | 0.0235 | 0.0664 | 0.0744 | 0.1073 | 0.2564 | 0.0499 |
| `AcousticQuality` | Raw Feature | 0.000005 | 0.0694 | 0.2425 | 0.2629 | 0.3970 | 0.9950 | 0.2231 |
| `InstrumentalScore` | Raw Feature | 0.000001 | 0.000001 | 0.0742 | 0.1177 | 0.2041 | 0.8693 | 0.1318 |
| `LivePerformanceLikelihood` | Raw Feature | 0.0243 | 0.0776 | 0.1663 | 0.1784 | 0.2689 | 0.5999 | 0.1182 |
| `MoodScore` | Raw Feature | 0.0256 | 0.4039 | 0.5648 | 0.5558 | 0.7166 | 0.9780 | 0.2255 |
| `TrackDurationMs` | Raw Feature (ms) | 63,973.0 | 207,099.9 | 243,684.1 | 241,903.7 | 281,851.7 | 464,723.2 | 59,326.6 |
| `Energy` | Raw Feature | 0.000067 | 0.2549 | 0.5118 | 0.5009 | 0.7460 | 1.0000 | 0.2900 |
| **`BeatsPerMinute`** | **Target** | **46.7180** | **101.0704** | **118.7477** | **119.0349** | **136.6866** | **206.0370** | **26.4681** |

### Key Data Quality Observations
1. **Scale Disparity:** Features span vastly different orders of magnitude—`TrackDurationMs` reaches $464,723\text{ ms}$, `AudioLoudness` is negative in decibels ($[-27.51, -1.36]\text{ dB}$), while `InstrumentalScore` and `AcousticQuality` have floor values near $10^{-6}$. Standardization (`StandardScaler`) is essential for regularized linear models and stable gradient updates.
2. **Boundary Floor Clumping & Skewed Ratios:** Several features (`VocalContent` at $0.0235$, `InstrumentalScore` at $1.07 \times 10^{-6}$, `LivePerformanceLikelihood` at $0.0243$) exhibit lower-bound floor clamping, which can create extreme spikes when computing ratios unless protected by smoothing ($\epsilon = 10^{-5}$) and quantile outlier clipping.
3. **Weak Individual Linear Correlations:** Individual Pearson correlations with `BeatsPerMinute` are small (`MoodScore`: $+0.0071$, `TrackDurationMs`: $+0.0066$, `RhythmScore`: $+0.0054$, `Energy`: $-0.0044$), indicating that non-linear interactions and tree-based ensembles (`HistGradientBoostingRegressor`) are better suited than purely additive linear models.

---

## 3. Clean Code Pipeline Architecture

Our project is structured into a modular Python package (`bpm_pipeline/`), a unified scikit-learn `Pipeline` module (`pipeline.py`), and an isolated unit test suite (`tests/`):

```text
kaggle-competition-2/
├── bpm_pipeline/
│   ├── __init__.py          # Feature lists (9 raw + 7 engineered), target, MODELS_DIR
│   ├── preprocess.py        # Schema validation, imputation, feature engineering, clipping, scaling
│   ├── train.py             # Train/val split, model training (HGBR / Ridge), metric computation
│   └── inference.py         # Artifact loading, batch test prediction, submission export
├── pipeline.py              # Standalone scikit-learn Pipeline (SimpleImputer -> FunctionTransformer -> StandardScaler -> HGBR)
├── tests/
│   ├── __init__.py
│   └── test_preprocess.py   # 9 unit tests validating every stage of the preprocessing pipeline
├── models/                  # Persisted joblib artifacts (imputer, bounds, scaler, model)
├── REPORT.md                # Full technical report
├── README.md                # Quick-start instructions
└── requirements.txt         # Locked package dependencies
```

### 5-Stage Preprocessing Pipeline (`bpm_pipeline/preprocess.py`)
1. **Schema Validation (`validate_input_columns`):** Verifies all $9$ required acoustic columns exist in the input DataFrame, raising an explicit `ValueError` if any column is missing.
2. **Median Imputation (`SimpleImputer(strategy="median")`):** Fitted on training data only to handle any missing or corrupted sensor readings at inference time without data leakage.
3. **Domain Audio Feature Engineering (`engineer_audio_features`):** Computes $7$ domain-specific features without mutating the input DataFrame:
   - `Rhythm_Energy_Product` $= \text{RhythmScore} \times \text{Energy}$
   - `Mood_Rhythm_Product` $= \text{MoodScore} \times \text{RhythmScore}$
   - `Vocal_to_Instrumental_Ratio` $= \frac{\text{VocalContent}}{\text{InstrumentalScore} + 10^{-5}}$
   - `Acoustic_Energy_Ratio` $= \frac{\text{AcousticQuality}}{\text{Energy} + 10^{-5}}$
   - `TrackDurationMinutes` $= \frac{\text{TrackDurationMs}}{60,000}$
   - `Loudness_Linear` $= 10^{\text{AudioLoudness} / 20}$ (converts logarithmic dB to linear amplitude ratio)
   - `Energy_Mood_Sum` $= \text{Energy} + \text{MoodScore}$
4. **Quantile Outlier Clipping (`compute_quantile_bounds` & `clip_outliers`):** Computes the $[0.5\%, 99.5\%]$ quantile bounds on the training fold (`bounds.joblib`) and clips extreme spikes in both training and unseen test data.
5. **Z-Score Standardization (`StandardScaler`):** Centers all $16$ features to $\mu = 0.0$ and scales to $\sigma = 1.0$ (`scaler.joblib`).

---

## 4. Unit Testing the Preprocessing Pipeline

We implemented $9$ self-contained unit tests in `tests/test_preprocess.py` using `unittest` and `pytest`. Every test runs inside an isolated `tempfile.TemporaryDirectory` using `unittest.mock.patch` so that tests are 100% deterministic and never pollute production artifacts.

| # | Test Name | Invariant Verified | Result |
| :-: | :--- | :--- | :---: |
| 1 | `test_validate_input_columns_raises_on_missing_feature` | Dropping a required column (`RhythmScore`) raises a clear `ValueError`. | **PASSED** |
| 2 | `test_engineer_audio_features_math_and_columns` | Verifies all $16$ column names and exact mathematical formulas for `TrackDurationMinutes` and `Rhythm_Energy_Product`. | **PASSED** |
| 3 | `test_preprocess_training_creates_artifacts_and_standardizes` | Verifies `is_training=True` saves `imputer.joblib`, `bounds.joblib`, and `scaler.joblib`, and standardizes all $16$ features to $\mu = 0.0, \sigma = 1.0$. | **PASSED** |
| 4 | `test_preprocess_imputes_missing_nan_values` | Injects `NaN` values into `RhythmScore`, `AcousticQuality`, and `TrackDurationMs` at inference time and verifies clean imputation with zero `NaN` or `inf` outputs. | **PASSED** |
| 5 | `test_outlier_clipping_bounds_extreme_test_spikes` | Injects extreme test spikes (`TrackDurationMs = 99,999,999`, `AudioLoudness = -999`) and verifies they are clamped to the training quantile bounds. | **PASSED** |
| 6 | `test_preprocess_does_not_mutate_input_dataframe` | Uses `pd.testing.assert_frame_equal` to guarantee `preprocess()` never mutates the caller's input DataFrame. | **PASSED** |
| 7 | `test_preprocess_inference_without_artifacts_raises_error` | Verifies calling `preprocess(..., is_training=False)` when artifacts are missing raises `FileNotFoundError`. | **PASSED** |
| 8 | `test_end_to_end_build_model_and_make_predictions` | Tests `build_model()` and `make_predictions()` end-to-end on CSV files and verifies the generated `submission.csv` schema (`id, BeatsPerMinute`). | **PASSED** |
| 9 | `test_sklearn_pipeline_and_run_pipeline` | Verifies the standalone `scikit-learn` `Pipeline` (`create_sklearn_preprocessing_pipeline`, `create_full_pipeline`, and `run_pipeline`). | **PASSED** |

### Test & Linter Execution Output
```text
$ pytest -v
============================= test session starts =============================
collected 9 items

tests/test_preprocess.py::TestBPMPreprocessingPipeline::test_end_to_end_build_model_and_make_predictions PASSED [ 11%]
tests/test_preprocess.py::TestBPMPreprocessingPipeline::test_engineer_audio_features_math_and_columns PASSED [ 22%]
tests/test_preprocess.py::TestBPMPreprocessingPipeline::test_outlier_clipping_bounds_extreme_test_spikes PASSED [ 33%]
tests/test_preprocess.py::TestBPMPreprocessingPipeline::test_preprocess_does_not_mutate_input_dataframe PASSED [ 44%]
tests/test_preprocess.py::TestBPMPreprocessingPipeline::test_preprocess_imputes_missing_nan_values PASSED [ 55%]
tests/test_preprocess.py::TestBPMPreprocessingPipeline::test_preprocess_inference_without_artifacts_raises_error PASSED [ 66%]
tests/test_preprocess.py::TestBPMPreprocessingPipeline::test_preprocess_training_creates_artifacts_and_standardizes PASSED [ 77%]
tests/test_preprocess.py::TestBPMPreprocessingPipeline::test_sklearn_pipeline_and_run_pipeline PASSED [ 88%]
tests/test_preprocess.py::TestBPMPreprocessingPipeline::test_validate_input_columns_raises_on_missing_feature PASSED [100%]

============================== 9 passed in 7.33s ==============================

$ flake8 bpm_pipeline tests pipeline.py --count --select=E,W,F --max-line-length=120
0
```

---

## 5. Experimental Results & Submission Verification

Running `python pipeline.py` on the full `ais-competition-v-3` dataset ($419,331$ training tracks, $104,833$ validation tracks, and $174,722$ test tracks) with `HistGradientBoostingRegressor` completed in $< 15$ seconds:

- **Validation RMSE:** `26.4398` (improves over the raw target standard deviation of `26.4681`)
- **Validation MAE:** `21.1824`
- **Generated `submission.csv`:** `174,722` prediction rows with exact header `id,BeatsPerMinute` and zero missing/NaN values, ready for Kaggle submission.
