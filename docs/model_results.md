# Model Results

This file summarizes model evaluation as training phases are completed.

## Planned Metrics

- Accuracy
- ROC-AUC
- Log loss

## Planned Model Comparison

| Model | Validation Accuracy | Validation ROC-AUC | Validation Log Loss | Test Accuracy | Test ROC-AUC | Test Log Loss |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.578 | 0.612 | 0.679 | 0.617 | 0.672 | 0.649 |
| XGBoost | 0.605 | 0.625 | 0.671 | 0.637 | 0.691 | 0.635 |
| PyTorch MLP | TBD | TBD | TBD | TBD | TBD | TBD |

## Logistic Regression Baseline

The logistic regression baseline uses the season-based split:

- Train: 2000-2020
- Validation: 2021-2023
- Test: 2024-2025

The model uses numeric preprocessing with median imputation and standard scaling, plus categorical preprocessing with most-frequent imputation and one-hot encoding. Numeric features include season, week, division flag, weather fields, and 3-game/5-game rolling team statistics. Categorical features include home team, away team, roof, and surface.

Artifacts are saved under `artifacts/logistic_regression/`.

## XGBoost

The XGBoost model uses the same season-based split and feature set as logistic regression. It uses median imputation for numeric features and one-hot encoding for categorical features, then trains a basic `XGBClassifier` with a conservative tree depth and learning rate.

Artifacts are saved under `artifacts/xgboost/`.

Top predictive features by XGBoost feature importance:

| Feature | Importance |
| --- | ---: |
| home_rolling_win_pct_5 | 0.046 |
| away_rolling_win_pct_5 | 0.041 |
| home_rolling_points_scored_5 | 0.028 |
| away_rolling_win_pct_3 | 0.022 |
| away_rolling_points_scored_5 | 0.022 |

## Leakage Prevention

Rolling team statistics are shifted before merging into each matchup so that the model only uses information available before kickoff. For each team, the feature builder sorts prior games chronologically, applies a one-game shift, then calculates 3-game and 5-game rolling averages for win percentage, points scored, and points allowed.

The current game's result and scores are kept out of the rolling windows. This prevents the model from learning from the result or statistics of the game it is trying to predict.
