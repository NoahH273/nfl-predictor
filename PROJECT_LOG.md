# Project Log

## Phase 1: Project Setup

Date: 2026-07-26

Completed the initial repository setup for the NFL predictor project.

- Created the required project folders for raw data, processed data, notebooks, source code, app code, and documentation.
- Confirmed the project uses Poetry metadata in `pyproject.toml`.
- Updated `.gitignore` so generated data and model artifacts stay out of version control.
- Added starter documentation files for project progress and model results.

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

## Phase 4: Feature Engineering

Date: 2026-07-26

Implemented `src/features/build_features.py` to create matchup features and leakage-safe rolling team features.

- Output: `data/processed/features.parquet`.
- Basic matchup features: season, week, home team, away team, matchup label, team codes, division-game flag, roof, surface, temperature, and wind.
- Rolling features: 3-game and 5-game rolling win percentage, points scored, and points allowed for both home and away teams.
- Leakage prevention: each team's rolling values are shifted so the current game is excluded before rolling averages are calculated.
- Verified output:
  - Input rows: 6,704.
  - Output rows: 6,704.
  - Output columns: 29.
- Run command: `.venv/bin/python src/features/build_features.py`.

## Phase 5: Train, Validation, and Test Split

Date: 2026-07-26

Implemented `src/data/split_dataset.py` to create chronological season-based splits from the feature dataset.

- Train seasons: 2000-2020.
- Validation seasons: 2021-2023.
- Test seasons: 2024-2025.
- Output files:
  - `data/processed/features_with_splits.parquet`
  - `data/processed/train_features.parquet`
  - `data/processed/validation_features.parquet`
  - `data/processed/test_features.parquet`
- Verified output:
  - All rows: 6,704.
  - Train rows: 5,349.
  - Validation rows: 812.
  - Test rows: 543.
  - Unassigned rows: 0.
- Decision: used a season-based split instead of a random split because the real prediction task is forward-looking.
- Run command: `.venv/bin/python src/data/split_dataset.py`.

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

## Phase 8: PyTorch MLP

Date: 2026-08-22

Implemented and ran `src/models/train_pytorch.py` as the neural network comparison model.

- Model: PyTorch multilayer perceptron.
- Architecture: input layer, one hidden linear layer, ReLU, dropout, and one linear output logit.
- Loss: `BCEWithLogitsLoss`.
- Optimizer: Adam.
- Training split: 2000-2020 only.
- Validation split: 2021-2023.
- Test split: 2024-2025.
- Feature set: same numeric and categorical predictors used by logistic regression and XGBoost.
- Preprocessing: median imputation and standard scaling for numeric features, plus most-frequent imputation and one-hot encoding for categorical features.
- Saved artifacts:
  - `artifacts/pytorch/preprocessor.joblib`
  - `artifacts/pytorch/model.pt`
  - `artifacts/pytorch/model_config.json`
  - `artifacts/pytorch/training_history.json`
  - `artifacts/pytorch/metrics.json`
  - `artifacts/pytorch/validation_predictions.parquet`
  - `artifacts/pytorch/test_predictions.parquet`
- Validation metrics: 0.586 accuracy, 0.613 ROC-AUC, 0.746 log loss.
- Test metrics: 0.587 accuracy, 0.656 ROC-AUC, 0.699 log loss.
- The MLP did not beat XGBoost, which is reasonable for a small tabular dataset. Training loss decreased while validation loss increased, suggesting the network began to overfit.
- Run command: `.venv/bin/python src/models/train_pytorch.py`.

## Phase 9: Evaluation

Date: 2026-08-22

Implemented `src/evaluation/evaluate_models.py` to standardize metrics across saved model prediction files.

- Metrics: accuracy, ROC-AUC, log loss, row counts.
- Optional reported metrics: test precision and recall in `docs/model_results.md`.
- Output: `artifacts/model_comparison_metrics.json`.
- Final model comparison table added to `docs/model_results.md`.
- Best model: XGBoost, with 0.637 test accuracy, 0.691 test ROC-AUC, and 0.635 test log loss.
- Decision: kept the final comparison based on held-out 2024-2025 test seasons.
- Run command: `.venv/bin/python src/evaluation/evaluate_models.py`.

## Phase 10: Streamlit Demo

Date: 2026-08-22

Implemented `app/streamlit_app.py` as a simple local and deployed prediction demo.

- Loads the models from `artifacts/`.
- Lets the user select home and away teams.
- Builds a prediction row using latest rolling team form from the processed game dataset.
- Displays predicted winner and home team win probability.
- Local run command: `streamlit run app/streamlit_app.py`.
- Deployed app: `https://noahh273-nfl-predictor-appstreamlit-app-arwef9.streamlit.app/`.

## Phase 11: Notebooks

Date: 2026-08-23

Added lightweight notebooks that explain and inspect the project without replacing the source-code pipeline.

- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_baseline_models.ipynb`
- `notebooks/03_feature_engineering.ipynb`
- `notebooks/04_pytorch_model.ipynb`
- Each notebook includes the source-script commands needed to reproduce the artifacts it inspects.
- Verified notebooks as valid JSON and checked that code cells compile.

## Phase 12: README

Date: 2026-08-24

Rewrote `README.md` into the resume-ready project structure.

- Added sections: Overview, Motivation, Tech Stack, Data, Feature Engineering, Modeling Approach, Results, How to Run, Limitations, and Future Work.
- Explained the `home_win` target, nflreadpy/nflverse data source, season-based split, rolling feature leakage prevention, model comparison, limitations, and resume bullet.
- Added the Streamlit Community Cloud deployment link.

## Key Design Decisions

- Used `nflreadpy` / nflverse as the data source to keep the data pipeline reproducible.
- Removed tied games so `home_win` stays a clean binary target.
- Used only regular season games for the main model.
- Used chronological season-based splits instead of random splits to simulate predicting future seasons from past seasons.
- Shifted rolling features before merging into matchups to prevent current-game leakage.
- Compared Logistic Regression, XGBoost, and PyTorch MLP on the same split and feature set.

## Final Results and Limitations

Final held-out test results:

| Model               | Test Accuracy | Test ROC-AUC | Test Log Loss |
| ------------------- | ------------: | -----------: | ------------: |
| Logistic Regression |         0.617 |        0.672 |         0.649 |
| XGBoost             |         0.637 |        0.691 |         0.635 |
| PyTorch MLP         |         0.587 |        0.656 |         0.699 |

Limitations:

- Injuries are not fully modeled.
- Roster changes and quarterback changes are difficult to capture.
- NFL games have high randomness and limited sample size.
- The model predicts straight-up home wins, not point spreads or betting value.

## Final Reflection

This project now demonstrates an end-to-end machine learning workflow: data loading, cleaning, feature engineering, season-based evaluation, three model types, standardized metrics, notebooks, documentation, and a Streamlit demo.

The strongest engineering lesson was that leakage prevention matters more than raw model complexity. The rolling features had to be calculated from prior games only, and the model split had to respect time. XGBoost performed best, which fits expectations for a small tabular dataset with engineered rolling features. The PyTorch model was still valuable because it showed basic neural network training, validation tracking, and an honest comparison against stronger tabular baselines.
