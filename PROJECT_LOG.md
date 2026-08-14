# Project Log

## Phase 1: Project Setup

Date: 2026-07-26

Completed the initial repository setup for the NFL predictor project.

- Created the required project folders for raw data, processed data, notebooks, source code, app code, and documentation.
- Confirmed the project uses Poetry metadata in `pyproject.toml`.
- Updated `.gitignore` so generated data and model artifacts stay out of version control.
- Added starter documentation files for project progress and model results.

## Upcoming Work

- Phase 8: Train the PyTorch MLP using the same feature set and split.

## Phase 2: Data Loading

Date: 2026-07-26

Implemented `src/data/load_raw_data.py` to load completed NFL seasons 2000-2025 with `nflreadpy`.

- Data source: `nflreadpy`, backed by nflverse data.
- Seasons loaded by default: 2000-2025.
- Raw schedules/games output: `data/raw/games.parquet`.
- Raw weekly team stats output: `data/raw/team_stats.parquet`.
- Verified output from the first successful run:
  - `data/raw/games.parquet`: 7,017 rows and 46 columns.
  - `data/raw/team_stats.parquet`: 14,014 rows and 133 columns.
- Run command: `.venv/bin/python src/data/load_raw_data.py`.

## Phase 3: Dataset Creation

Date: 2026-07-26

Implemented `src/data/make_dataset.py` to create the first cleaned game-level modeling dataset from `data/raw/games.parquet`.

- Output: `data/processed/modeling_dataset.parquet`.
- Target: `home_win`, where `1` means the home team scored more points than the away team and `0` means the away team won.
- Tie handling: removed tied games so the first modeling target stays binary.
- Sorting: ordered games by `season`, `week`, parsed `gameday`, and `game_id`.
- Verified output from the first successful run:
  - Raw schedule rows: 7,017.
  - Removed non-regular-season rows: 298.
  - Removed rows missing final scores: 0.
  - Removed tied games: 15.
  - Final modeling rows: 6,704.
- Run command: `.venv/bin/python src/data/make_dataset.py`.

## Phase 6: Logistic Regression Baseline

Date: 2026-08-02

Implemented `src/models/train_logistic_regression.py` as the first baseline model.

- Model: scikit-learn `LogisticRegression`.
- Training split: 2000-2020 only.
- Validation split: 2021-2023.
- Test split: 2024-2025.
- Numeric preprocessing: median imputation and standard scaling for season, week, division flag, weather, and rolling team stats.
- Categorical preprocessing: most-frequent imputation and one-hot encoding for home team, away team, roof, and surface.
- Saved artifacts:
  - `artifacts/logistic_regression/model.joblib`
  - `artifacts/logistic_regression/metrics.json`
  - `artifacts/logistic_regression/validation_predictions.parquet`
  - `artifacts/logistic_regression/test_predictions.parquet`
- Validation metrics: 0.578 accuracy, 0.612 ROC-AUC, 0.679 log loss.
- Test metrics: 0.617 accuracy, 0.672 ROC-AUC, 0.649 log loss.
- Run command: `.venv/bin/python src/models/train_logistic_regression.py`.

## Phase 7: XGBoost Model

Date: 2026-08-14

Implemented `src/models/train_xgboost.py` as the main traditional ML model.

- Model: `xgboost.XGBClassifier`.
- Training split: 2000-2020 only.
- Validation split: 2021-2023.
- Test split: 2024-2025.
- Feature set: same numeric and categorical predictors used by logistic regression.
- Preprocessing: median imputation for numeric features and one-hot encoding for categorical features.
- Basic configuration: 300 trees, 0.05 learning rate, max depth 3, 0.9 subsampling, 0.9 column sampling.
- Saved artifacts:
  - `artifacts/xgboost/model.joblib`
  - `artifacts/xgboost/metrics.json`
  - `artifacts/xgboost/feature_importance.parquet`
  - `artifacts/xgboost/validation_predictions.parquet`
  - `artifacts/xgboost/test_predictions.parquet`
- Validation metrics: 0.605 accuracy, 0.625 ROC-AUC, 0.671 log loss.
- Test metrics: 0.637 accuracy, 0.691 ROC-AUC, 0.635 log loss.
- Top features: `home_rolling_win_pct_5`, `away_rolling_win_pct_5`, `home_rolling_points_scored_5`, `away_rolling_win_pct_3`, and `away_rolling_points_scored_5`.
- Run command: `.venv/bin/python src/models/train_xgboost.py`.
