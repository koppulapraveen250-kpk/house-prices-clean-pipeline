# House Prices Clean Preprocessing Pipeline & Unit Tests

Clean-code machine learning preprocessing, training, and inference pipeline with unit tests (`pytest` / `unittest`) for the Kaggle **House Prices: Advanced Regression Techniques** competition (`https://www.kaggle.com/t/31f8ecfe2319647799227b9c7f0b9018`).

## Project Structure

```text
house-prices-clean-pipeline/
├── house_prices/
│   ├── __init__.py          # Feature constants, target name, and MODELS_DIR path
│   ├── preprocess.py        # Imputation, scaling, one-hot encoding, artifact I/O
│   ├── train.py             # Train/validation split, model fitting, RMSLE evaluation
│   └── inference.py         # Artifact loading and batch prediction on test data
├── pipeline.py              # Standalone scikit-learn ColumnTransformer & Pipeline
├── tests/
│   ├── __init__.py
│   └── test_preprocess.py   # 8 unit tests covering preprocessing & pipeline invariants
├── models/                  # Persisted joblib artifacts (imputers, scaler, encoder, model)
├── REPORT.md                # Full technical report of work and experimental results
├── submission.csv           # Generated test set predictions (1,459 rows)
└── requirements.txt         # Locked environment dependencies
```

## Quick Start

### 1. Run Unit Tests
```bash
pytest -v
```

### 2. Run Code Linter (`flake8`)
```bash
flake8 house_prices tests pipeline.py
```

### 3. Run the Clean Pipeline
```bash
python pipeline.py
```

## Report
See [`REPORT.md`](./REPORT.md) for the complete write-up of our preprocessing design, anti-leakage architecture, unit tests, and evaluation results.
