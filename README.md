# Kaggle Competition 2 (`ais-competition-v-3`): Clean Code Pipeline & Unit Tests

Clean-code preprocessing, feature engineering, training, and inference pipeline with automated unit tests (`pytest` / `unittest`) for **Kaggle Competition 2 (`ais-competition-v-3` — Predicting `BeatsPerMinute`)** (`https://www.kaggle.com/t/31f8ecfe2319647799227b9c7f0b9018`).

## Project Structure

```text
kaggle-competition-2/
├── bpm_pipeline/
│   ├── __init__.py          # Raw & engineered feature constants, target, MODELS_DIR
│   ├── preprocess.py        # Validation, imputation, audio feature engineering, clipping, scaling
│   ├── train.py             # Train/validation split, HistGradientBoosting/Ridge training, metrics
│   └── inference.py         # Artifact loading, batch test prediction, submission CSV generation
├── pipeline.py              # Standalone scikit-learn Pipeline & end-to-end runner
├── tests/
│   ├── __init__.py
│   └── test_preprocess.py   # 9 unit tests verifying preprocessing & pipeline invariants
├── models/                  # Persisted joblib artifacts (imputer, bounds, scaler, model)
├── REPORT.md                # Full technical report of work and experimental results
└── requirements.txt         # Locked environment dependencies
```

## Quick Start

### 1. Run Unit Tests
```bash
pytest -v
```

### 2. Run Code Linter (`flake8`)
```bash
flake8 bpm_pipeline tests pipeline.py
```

### 3. Run the Full Pipeline
```bash
python pipeline.py
```

## Report
See [`REPORT.md`](./REPORT.md) for the complete technical report covering dataset exploration, the 5-stage preprocessing pipeline, unit test suite, and evaluation metrics.
