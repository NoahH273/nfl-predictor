# NFL Predictor

NFL game outcome prediction project using `nflreadpy`, scikit-learn, XGBoost, and PyTorch.

## Overview

This project predicts whether the home team wins an NFL game. The final project will include a reproducible data pipeline, engineered rolling team features, model comparisons, and a small prediction demo.

## Project Status

Phase 1 is complete: the repository structure, starter documentation, dependency metadata, and generated-file ignore rules are in place.

## Planned Tech Stack

- Python 3.12
- Poetry
- `nflreadpy` for NFL data access
- pandas, Polars, NumPy, and PyArrow for data processing
- scikit-learn for baseline modeling and metrics
- XGBoost for tree-based modeling
- PyTorch for an MLP model
- Streamlit for the demo app

## Planned Data

The project will use historical NFL data loaded with `nflreadpy`, including schedules and team-level weekly statistics for completed seasons. Raw Parquet files will be written under `data/raw/`, and the processed modeling dataset will be written under `data/processed/`.

## Target

The first model target will be `home_win`, defined as:

```text
home_win = 1 if home_score > away_score else 0
```

Tie handling will be documented when the processed dataset is created.

## Evaluation Plan

The main evaluation will use a season-based split instead of a random split. This better simulates the real use case: training on past seasons and predicting future games.

## Data Leakage Prevention

Rolling team statistics will be shifted before being used as matchup features so that each prediction only uses information available before that game.

## How to Run

Install dependencies:

```bash
poetry install
```

Phase-specific commands will be added as the data pipeline, models, and Streamlit app are implemented.
