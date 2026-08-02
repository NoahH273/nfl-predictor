# Model Results

This file will summarize model evaluation after the training phases are complete.

## Planned Metrics

- Accuracy
- ROC-AUC
- Log loss

## Planned Model Comparison

| Model | Validation Accuracy | Validation ROC-AUC | Validation Log Loss | Test Accuracy | Test ROC-AUC | Test Log Loss |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | TBD | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD | TBD |
| PyTorch MLP | TBD | TBD | TBD | TBD | TBD | TBD |

## Leakage Prevention

Rolling team statistics are shifted before merging into each matchup so that the model only uses information available before kickoff. For each team, the feature builder sorts prior games chronologically, applies a one-game shift, then calculates 3-game and 5-game rolling averages for win percentage, points scored, and points allowed.

The current game's result and scores are kept out of the rolling windows. This prevents the model from learning from the result or statistics of the game it is trying to predict.
