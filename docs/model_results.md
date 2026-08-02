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
| XGBoost | TBD | TBD | TBD | TBD | TBD | TBD |
| PyTorch MLP | TBD | TBD | TBD | TBD | TBD | TBD |

## Logistic Regression Baseline

The logistic regression baseline uses the season-based split:

- Train: 2000-2020
- Validation: 2021-2023
- Test: 2024-2025

The model uses numeric preprocessing with median imputation and standard scaling, plus categorical preprocessing with most-frequent imputation and one-hot encoding. Numeric features include season, week, division flag, weather fields, and 3-game/5-game rolling team statistics. Categorical features include home team, away team, roof, and surface.

Artifacts are saved under `artifacts/logistic_regression/`.

## Leakage Prevention

Rolling team statistics are shifted before merging into each matchup so that the model only uses information available before kickoff. For each team, the feature builder sorts prior games chronologically, applies a one-game shift, then calculates 3-game and 5-game rolling averages for win percentage, points scored, and points allowed.

The current game's result and scores are kept out of the rolling windows. This prevents the model from learning from the result or statistics of the game it is trying to predict.
