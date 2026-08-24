# NFL Game Predictor

An end-to-end machine learning project that predicts whether the home team wins an NFL game using historical nflverse data loaded with `nflreadpy`.

Live demo: [Streamlit Community Cloud](https://noahh273-nfl-predictor-appstreamlit-app-arwef9.streamlit.app/)

## Overview

This project builds a reproducible NFL game prediction pipeline:

- load historical schedules and team statistics
- create a cleaned game-level modeling dataset
- engineer matchup and rolling team-performance features
- train Logistic Regression, XGBoost, and PyTorch MLP models
- compare models on held-out seasons
- serve a simple Streamlit prediction demo

The target is `home_win`, where `1` means the home team won and `0` means the away team won.

## Motivation

The goal is to demonstrate practical Python data engineering, leakage-aware feature engineering, basic machine learning evaluation, and PyTorch experience in a project that is easy to understand and run locally.

This is a learning project, not a betting model.

## Tech Stack

- Python 3.12
- Poetry
- `nflreadpy` / nflverse
- pandas, Polars, NumPy, PyArrow
- scikit-learn
- XGBoost
- PyTorch
- Streamlit

## Data

Data comes from `nflreadpy`, which provides access to nflverse datasets.

Raw files:

- `data/raw/games.parquet`
- `data/raw/team_stats.parquet`

Processed files:

- `data/processed/modeling_dataset.parquet`
- `data/processed/features.parquet`
- `data/processed/train_features.parquet`
- `data/processed/validation_features.parquet`
- `data/processed/test_features.parquet`

The cleaned dataset keeps regular season games only, removes games without final scores, removes ties, and sorts games chronologically.

## Feature Engineering

The model uses basic matchup features and rolling team-performance features.

Basic features include:

- `season`
- `week`
- `home_team`
- `away_team`
- division-game indicator
- roof, surface, temperature, and wind

Rolling features include 3-game and 5-game averages for both teams:

- win percentage
- points scored
- points allowed

Rolling statistics are shifted before being merged into each matchup. The current game is excluded from its own features, so the model only sees information that would have been available before kickoff.

## Modeling Approach

All models use the same season-based split:

- Train: 2000-2020
- Validation: 2021-2023
- Test: 2024-2025

A season-based split is used instead of a random split because the real prediction task is forward-looking: train on past seasons and predict future games.

Models compared:

- Logistic Regression baseline
- XGBoost traditional ML model
- PyTorch MLP neural network

## Results

| Model | Validation Accuracy | Validation ROC-AUC | Validation Log Loss | Test Accuracy | Test ROC-AUC | Test Log Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.578 | 0.612 | 0.679 | 0.617 | 0.672 | 0.649 |
| XGBoost | 0.605 | 0.625 | 0.671 | 0.637 | 0.691 | 0.635 |
| PyTorch MLP | 0.586 | 0.613 | 0.746 | 0.587 | 0.656 | 0.699 |

XGBoost performed best overall on the held-out test seasons. This is reasonable because boosted tree models often perform well on smaller tabular datasets with engineered features.

More detail is available in `docs/model_results.md`.

## How to Run

Install dependencies:

```bash
poetry install
```

Load raw data:

```bash
.venv/bin/python src/data/load_raw_data.py
```

Create the modeling dataset and features:

```bash
.venv/bin/python src/data/make_dataset.py
.venv/bin/python src/features/build_features.py
.venv/bin/python src/data/split_dataset.py
```

Train models:

```bash
.venv/bin/python src/models/train_logistic_regression.py
.venv/bin/python src/models/train_xgboost.py
.venv/bin/python src/models/train_pytorch.py
```

Recompute standardized metrics:

```bash
.venv/bin/python src/evaluation/evaluate_models.py
```

Run the local Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Deployment:

- Streamlit Community Cloud: [NFL Game Predictor](https://noahh273-nfl-predictor-appstreamlit-app-arwef9.streamlit.app/)

## Limitations

- Injuries are not fully modeled.
- Roster changes and quarterback changes are difficult to capture.
- NFL games have high randomness and a small sample size compared with many ML problems.
- The model predicts straight-up home wins, not betting spreads or betting value.
- The project is intended as a learning project, not as betting advice.

## Future Work

- Add model calibration to improve probability quality.
- Add richer team-strength features such as ELO ratings.
- Improve the Streamlit UI with clearer matchup context and recent team form.
- Add automated notebook execution checks.

## Resume Bullet

Built an end-to-end NFL game prediction system using Python, nflreadpy, scikit-learn, XGBoost, and PyTorch, engineering rolling team-performance features and evaluating models with season-based validation.
