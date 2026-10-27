# Project Report: House Prices Preprocessing Pipeline & Unit Testing

**Competition:** Kaggle — *House Prices: Advanced Regression Techniques*  
**Author:** Praveen Kumar Koppula  
**Repository:** `dsp-praveen-koppula`

---

## 1. Executive Summary

This report documents the refactoring, industrialization, and unit testing of our machine learning pipeline for the **House Prices: Advanced Regression Techniques** competition. Starting from an exploratory notebook with manual feature transformations, we engineered a clean, modular, production-ready Python pipeline (`house_prices` package and standalone `pipeline.py`) that strictly prevents data leakage, persists fitted preprocessing artifacts (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`), and is verified by an automated unit test suite (`tests/test_preprocess.py`) with 100% pass rate and zero `flake8` linting errors.

---

## 2. Dataset & Feature Selection

The dataset consists of residential property sales in Ames, Iowa:
- **Training set (`data/train.csv`):** $1,460$ rows $\times$ $81$ columns ($80$ features + target `SalePrice`).
- **Testing set (`data/test.csv`):** $1,459$ rows $\times$ $80$ feature columns.
- **Target Variable (`SalePrice`):** Continuous positive sale price in USD, ranging from $\$34,900$ to $\$755,000$ (mean: $\$180,921.20$, median: $\$163,000.00$, std: $\$79,442.50$).

### Selected Features
To build an interpretable, low-latency baseline pipeline, four high-signal features covering property size, land area, location, and architectural style were selected:
1. **Continuous Features (`CONTINUOUS_FEATURES`):**
   - `GrLivArea`: Above grade (ground) living area square feet.
   - `LotArea`: Lot size in square feet.
2. **Categorical Features (`CATEGORICAL_FEATURES`):**
   - `Neighborhood`: Physical locations within Ames city limits ($25$ distinct categories in training).
   - `HouseStyle`: Style of dwelling (e.g., `1Story`, `2Story`, `1.5Fin`; $8$ distinct categories in training).

---

## 3. Clean Code Pipeline Architecture

Our refactored codebase provides two complementary implementations:
1. **Modular Package (`house_prices/`):** Follows strict production separation between preprocessing, training, and inference without cross-dependencies between `train.py` and `inference.py`. Every function is kept concise ($\le 10$ lines of body code), type-annotated, and documented with Google-style docstrings.
2. **Unified Scikit-Learn Pipeline (`pipeline.py`):** Encapsulates imputation, scaling, one-hot encoding, and regression inside a single `ColumnTransformer` + `Pipeline` workflow.

```
dsp-praveen-koppula/
├── house_prices/
│   ├── __init__.py          # Feature constants, target name, and MODELS_DIR path
│   ├── preprocess.py        # Imputation, scaling, one-hot encoding, artifact I/O
│   ├── train.py             # Train/validation split, model fitting, RMSLE evaluation
│   └── inference.py         # Artifact loading and batch prediction on test data
├── pipeline.py              # End-to-end scikit-learn ColumnTransformer & Pipeline
├── tests/
│   ├── __init__.py
│   └── test_preprocess.py   # 8 unit tests covering preprocessing & pipeline invariants
├── models/                  # Persisted joblib artifacts (imputers, scaler, encoder, model)
├── REPORT.md                # Technical report of work and experimental results
└── requirements.txt         # Locked environment dependencies
```

### Key Clean-Code & Anti-Leakage Design Principles
- **Split Before Preprocessing (Zero Data Leakage):** The dataset is split into `X_train` ($1,168$ rows, $80\%$) and `X_val` ($292$ rows, $20\%$) *before* any statistics (medians, means, variances, or category vocabularies) are computed.
- **Train vs. Inference Separation (`is_training` flag):**
  - When `is_training=True`, `preprocess()` fits `SimpleImputer(strategy="median")` and `StandardScaler()` on continuous columns, and `SimpleImputer(strategy="most_frequent")` and `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` on categorical columns, persisting all four transformers to `models/*.joblib`.
  - When `is_training=False`, `preprocess()` never calls `.fit()`; it loads the persisted transformers from disk and applies `.transform()`.
- **Immutability:** Input DataFrames are never mutated in place. `preprocess()` returns a newly constructed DataFrame with deterministic column names (`GrLivArea`, `LotArea`, plus one-hot encoded category names) and preserved row indices.
- **Defensive Error Handling:** Missing input CSV files or missing saved `.joblib` artifacts raise clear, actionable `FileNotFoundError` exceptions.

---

## 4. Unit Testing the Preprocessing Pipeline

We wrote a comprehensive unit test suite in `tests/test_preprocess.py` using `unittest` and `pytest`. All tests use isolated `tempfile.TemporaryDirectory` fixtures and `unittest.mock.patch` so they never overwrite or depend on local disk state.

| Test Case | Purpose & Invariant Verified | Status |
| :--- | :--- | :---: |
| `test_preprocess_training_creates_artifacts_and_scales` | Verifies `is_training=True` saves `num_imputer.joblib`, `cat_imputer.joblib`, `scaler.joblib`, and `encoder.joblib`, outputs expected column dimensions, and standardizes continuous features to $\mu = 0.0, \sigma = 1.0$. | **PASSED** |
| `test_preprocess_inference_handles_unseen_categories` | Verifies `is_training=False` loads saved transformers, matches training feature columns exactly, and encodes unseen test categories (`UnseenNeighborhood`, `UnseenStyle`) as all-zero one-hot vectors without crashing. | **PASSED** |
| `test_preprocess_imputes_missing_values` | Injects `NaN` / `None` into both continuous and categorical features and verifies that fitted imputers replace all missing values cleanly with zero `NaN`s remaining. | **PASSED** |
| `test_preprocess_does_not_mutate_input_dataframe` | Asserts via `pd.testing.assert_frame_equal` that the caller's raw DataFrame is completely unmodified after calling `preprocess()`. | **PASSED** |
| `test_preprocess_inference_without_artifacts_raises_error` | Verifies that running inference before training raises a clear `FileNotFoundError`. | **PASSED** |
| `test_end_to_end_build_model_and_make_predictions` | Tests `build_model()` and `make_predictions()` end-to-end on temporary CSV files, checking `model.joblib` persistence and prediction array shape. | **PASSED** |
| `test_sklearn_column_transformer_and_full_pipeline` | Validates the `ColumnTransformer` and `Pipeline` objects in `pipeline.py` on training and unseen test data. | **PASSED** |
| `test_run_pipeline_generates_submission` | Verifies `run_pipeline()` produces validation metrics and writes a valid `submission.csv` with `Id` and `SalePrice` columns. | **PASSED** |

### Test & Linter Execution Summary
```text
$ pytest -v
collected 8 items
tests/test_preprocess.py::TestPreprocessingPipeline::test_end_to_end_build_model_and_make_predictions PASSED [ 12%]
tests/test_preprocess.py::TestPreprocessingPipeline::test_preprocess_does_not_mutate_input_dataframe PASSED [ 25%]
tests/test_preprocess.py::TestPreprocessingPipeline::test_preprocess_imputes_missing_values PASSED [ 37%]
tests/test_preprocess.py::TestPreprocessingPipeline::test_preprocess_inference_handles_unseen_categories PASSED [ 50%]
tests/test_preprocess.py::TestPreprocessingPipeline::test_preprocess_inference_without_artifacts_raises_error PASSED [ 62%]
tests/test_preprocess.py::TestPreprocessingPipeline::test_preprocess_training_creates_artifacts_and_scales PASSED [ 75%]
tests/test_preprocess.py::TestPreprocessingPipeline::test_run_pipeline_generates_submission PASSED [ 87%]
tests/test_preprocess.py::TestPreprocessingPipeline::test_sklearn_column_transformer_and_full_pipeline PASSED [100%]
============================== 8 passed in 2.55s ==============================

$ flake8 house_prices tests pipeline.py --count --select=E,W,F --max-line-length=120
0
```

---

## 5. Model Evaluation Results

Using an 80/20 train/validation split (`random_state=42`) on `data/train.csv`, the pipeline achieved the following validation metrics:

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **RMSLE** (Competition Metric) | **0.2026** (`0.20` rounded) | Root Mean Squared Logarithmic Error; corresponds to $\approx 22\%$ relative error on house price. |
| **R² Score** | **0.7784** | The 4 selected features explain **$77.84\%$** of the variance in validation house prices. |
| **MAE** | **$26,551.64** | Mean Absolute Error across validation properties. |
| **RMSE** | **$41,224.43** | Root Mean Squared Error across validation properties. |

Predictions on the $1,459$ test houses in `data/test.csv` were generated and exported to `submission.csv` (`1,459` rows, columns `Id, SalePrice`, zero missing or negative values).
