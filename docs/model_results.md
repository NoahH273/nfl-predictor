# Model Results

This file summarizes model evaluation for the season-based held-out splits.

## Metrics

- Accuracy
- ROC-AUC
- Log loss

## Model Comparison

| Model | Validation Accuracy | Validation ROC-AUC | Validation Log Loss | Test Accuracy | Test ROC-AUC | Test Log Loss |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.578 | 0.612 | 0.679 | 0.617 | 0.672 | 0.649 |
| XGBoost | 0.605 | 0.625 | 0.671 | 0.637 | 0.691 | 0.635 |
| PyTorch MLP | 0.586 | 0.613 | 0.746 | 0.587 | 0.656 | 0.699 |

## Test Precision And Recall

These optional metrics use the same held-out 2024-2025 test split and a 0.50 probability threshold.

| Model | Test Precision | Test Recall |
| --- | ---: | ---: |
| Logistic Regression | 0.626 | 0.708 |
| XGBoost | 0.657 | 0.677 |
| PyTorch MLP | 0.612 | 0.629 |

All three models were evaluated on the same chronological split:

- Train: 2000-2020
- Validation: 2021-2023
- Test: 2024-2025

The test split is the main held-out comparison because it represents the most recent unseen seasons.

## Summary

XGBoost performed best overall, with the highest validation and test accuracy, highest validation and test ROC-AUC, and lowest validation and test log loss. On the held-out test seasons, XGBoost reached 0.637 accuracy, 0.691 ROC-AUC, and 0.635 log loss.

The logistic regression baseline was competitive but weaker than XGBoost, which suggests the engineered rolling features contain useful signal even for a simple linear baseline.

The PyTorch MLP did not outperform XGBoost. That is reasonable for this project because the dataset is relatively small and tabular, while tree-based boosted models often perform strongly on structured features with limited tuning. The PyTorch training history also shows validation loss increasing while training loss decreases, which suggests the MLP began to overfit the training seasons.

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

## PyTorch MLP

The PyTorch model uses the same season-based split and feature set as the logistic regression and XGBoost models. It uses median imputation and standard scaling for numeric features, one-hot encoding for categorical features, and a simple multilayer perceptron with one hidden layer, ReLU activation, dropout, and a single logit output trained with `BCEWithLogitsLoss`.

Artifacts are saved under `artifacts/pytorch/`.

This model is useful as a neural-network comparison even though it is not the best-performing model.

## Leakage Prevention

Rolling team statistics are shifted before merging into each matchup so that the model only uses information available before kickoff. For each team, the feature builder sorts prior games chronologically, applies a one-game shift, then calculates 3-game and 5-game rolling averages for win percentage, points scored, and points allowed.

The current game's result and scores are kept out of the rolling windows. This prevents the model from learning from the result or statistics of the game it is trying to predict.
