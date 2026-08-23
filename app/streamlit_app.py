"""Streamlit demo for NFL home win predictions."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn


ROOT_DIR = Path(__file__).resolve().parents[1]
GAMES_PATH = ROOT_DIR / "data/processed/modeling_dataset.parquet"
FEATURES_PATH = ROOT_DIR / "data/processed/features.parquet"

MODEL_OPTIONS = {
    "Logistic Regression": {
        "kind": "sklearn",
        "path": ROOT_DIR / "artifacts/logistic_regression/model.joblib",
    },
    "XGBoost": {
        "kind": "sklearn",
        "path": ROOT_DIR / "artifacts/xgboost/model.joblib",
    },
    "PyTorch MLP": {
        "kind": "pytorch",
        "model_path": ROOT_DIR / "artifacts/pytorch/model.pt",
        "preprocessor_path": ROOT_DIR / "artifacts/pytorch/preprocessor.joblib",
        "config_path": ROOT_DIR / "artifacts/pytorch/model_config.json",
    },
}

ROLLING_FEATURES = [
    "rolling_win_pct_3",
    "rolling_points_scored_3",
    "rolling_points_allowed_3",
    "rolling_win_pct_5",
    "rolling_points_scored_5",
    "rolling_points_allowed_5",
]

DIVISIONS = {
    "ARI": "NFC West",
    "ATL": "NFC South",
    "BAL": "AFC North",
    "BUF": "AFC East",
    "CAR": "NFC South",
    "CHI": "NFC North",
    "CIN": "AFC North",
    "CLE": "AFC North",
    "DAL": "NFC East",
    "DEN": "AFC West",
    "DET": "NFC North",
    "GB": "NFC North",
    "HOU": "AFC South",
    "IND": "AFC South",
    "JAX": "AFC South",
    "KC": "AFC West",
    "LA": "NFC West",
    "LAC": "AFC West",
    "LV": "AFC West",
    "MIA": "AFC East",
    "MIN": "NFC North",
    "NE": "AFC East",
    "NO": "NFC South",
    "NYG": "NFC East",
    "NYJ": "AFC East",
    "PHI": "NFC East",
    "PIT": "AFC North",
    "SEA": "NFC West",
    "SF": "NFC West",
    "TB": "NFC South",
    "TEN": "AFC South",
    "WAS": "NFC East",
}


class GameOutcomeMLP(nn.Module):
    """Simple multilayer perceptron for binary game outcome prediction."""

    def __init__(self, input_dim: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


class PyTorchPredictionModel:
    """Prediction wrapper matching sklearn's predict_proba output shape."""

    def __init__(
        self,
        preprocessor,
        model: GameOutcomeMLP,
    ) -> None:
        self.preprocessor = preprocessor
        self.model = model

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        features = self.preprocessor.transform(rows).astype(np.float32)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(torch.from_numpy(features))
            probabilities = torch.sigmoid(logits).numpy()

        return np.column_stack([1 - probabilities, probabilities])


@st.cache_resource
def load_model(model_name: str):
    model_info = MODEL_OPTIONS[model_name]

    if model_info["kind"] == "sklearn":
        return joblib.load(model_info["path"])

    with open(model_info["config_path"], encoding="utf-8") as file:
        config = json.load(file)

    preprocessor = joblib.load(model_info["preprocessor_path"])
    model = GameOutcomeMLP(
        input_dim=int(config["input_dim"]),
        hidden_size=int(config["hidden_size"]),
        dropout=float(config["dropout"]),
    )
    state_dict = torch.load(model_info["model_path"], map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return PyTorchPredictionModel(preprocessor, model)


def validate_model_artifacts(model_name: str) -> list[Path]:
    model_info = MODEL_OPTIONS[model_name]
    paths = [
        value
        for key, value in model_info.items()
        if key.endswith("path") and isinstance(value, Path)
    ]
    return [path for path in paths if not path.exists()]


@st.cache_data
def load_games() -> pd.DataFrame:
    games = pd.read_parquet(GAMES_PATH)
    games["gameday_sort"] = pd.to_datetime(games["gameday"], errors="coerce")
    return games.sort_values(["season", "week", "gameday_sort", "game_id"])


@st.cache_data
def load_feature_context() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


def available_teams(games: pd.DataFrame) -> list[str]:
    teams = set(games["home_team"]).union(set(games["away_team"]))
    return sorted(teams)


def build_team_history(games: pd.DataFrame) -> pd.DataFrame:
    home_rows = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "season": games["season"],
            "week": games["week"],
            "gameday_sort": games["gameday_sort"],
            "team": games["home_team"],
            "win": games["home_win"],
            "points_scored": games["home_score"],
            "points_allowed": games["away_score"],
        }
    )
    away_rows = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "season": games["season"],
            "week": games["week"],
            "gameday_sort": games["gameday_sort"],
            "team": games["away_team"],
            "win": 1 - games["home_win"],
            "points_scored": games["away_score"],
            "points_allowed": games["home_score"],
        }
    )
    return (
        pd.concat([home_rows, away_rows], ignore_index=True)
        .sort_values(["team", "season", "week", "gameday_sort", "game_id"])
        .reset_index(drop=True)
    )


@st.cache_data
def latest_team_form(games: pd.DataFrame) -> pd.DataFrame:
    team_history = build_team_history(games)
    records = []

    for team, team_games in team_history.groupby("team"):
        team_record = {"team": team}
        for window in (3, 5):
            recent_games = team_games.tail(window)
            team_record[f"rolling_win_pct_{window}"] = recent_games["win"].mean()
            team_record[f"rolling_points_scored_{window}"] = recent_games[
                "points_scored"
            ].mean()
            team_record[f"rolling_points_allowed_{window}"] = recent_games[
                "points_allowed"
            ].mean()
        records.append(team_record)

    return pd.DataFrame(records).set_index("team")


def default_game_context(features: pd.DataFrame) -> dict[str, object]:
    latest_season = int(features["season"].max())
    latest_week = int(features.loc[features["season"] == latest_season, "week"].max())

    return {
        "season": latest_season,
        "week": latest_week,
        "roof": features["roof"].mode(dropna=True).iloc[0],
        "surface": features["surface"].mode(dropna=True).iloc[0],
        "temp": float(features["temp"].median()),
        "wind": float(features["wind"].median()),
    }


def is_division_game(home_team: str, away_team: str) -> int:
    return int(DIVISIONS.get(home_team) == DIVISIONS.get(away_team))


def build_prediction_row(
    home_team: str,
    away_team: str,
    form: pd.DataFrame,
    context: dict[str, object],
) -> pd.DataFrame:
    row = {
        "season": context["season"],
        "week": context["week"],
        "home_team": home_team,
        "away_team": away_team,
        "is_division_game": is_division_game(home_team, away_team),
        "roof": context["roof"],
        "surface": context["surface"],
        "temp": context["temp"],
        "wind": context["wind"],
    }

    for feature in ROLLING_FEATURES:
        row[f"home_{feature}"] = form.loc[home_team, feature]
        row[f"away_{feature}"] = form.loc[away_team, feature]

    return pd.DataFrame([row])


def display_prediction(home_team: str, away_team: str, probability: float) -> None:
    predicted_winner = home_team if probability >= 0.5 else away_team
    away_probability = 1 - probability

    st.metric("Predicted winner", predicted_winner)
    st.metric("Home team win probability", f"{probability:.1%}")
    st.progress(float(probability))

    col_home, col_away = st.columns(2)
    col_home.metric(home_team, f"{probability:.1%}")
    col_away.metric(away_team, f"{away_probability:.1%}")


def main() -> None:
    st.set_page_config(page_title="NFL Game Predictor")
    st.title("NFL Game Predictor")

    games = load_games()
    features = load_feature_context()
    form = latest_team_form(games)
    teams = available_teams(games)
    context = default_game_context(features)

    selected_model = st.selectbox(
        "Model",
        list(MODEL_OPTIONS),
        index=list(MODEL_OPTIONS).index("XGBoost"),
    )
    missing_artifacts = validate_model_artifacts(selected_model)
    if missing_artifacts:
        st.error(
            "Missing model artifact(s): "
            + ", ".join(str(path.relative_to(ROOT_DIR)) for path in missing_artifacts)
        )
        st.stop()
    model = load_model(selected_model)

    col_home, col_away = st.columns(2)
    home_team = col_home.selectbox("Home team", teams, index=teams.index("ARI"))
    away_default = "ATL" if home_team != "ATL" else "BAL"
    away_team = col_away.selectbox("Away team", teams, index=teams.index(away_default))

    if home_team == away_team:
        st.warning("Choose two different teams.")
        return

    with st.expander("Game context", expanded=False):
        context["season"] = st.number_input(
            "Season",
            min_value=2000,
            max_value=2100,
            value=int(context["season"]),
            step=1,
        )
        context["week"] = st.number_input(
            "Week",
            min_value=1,
            max_value=22,
            value=int(context["week"]),
            step=1,
        )
        context["roof"] = st.selectbox(
            "Roof",
            sorted(features["roof"].dropna().unique()),
            index=sorted(features["roof"].dropna().unique()).index(context["roof"]),
        )
        context["surface"] = st.selectbox(
            "Surface",
            sorted(features["surface"].dropna().unique()),
            index=sorted(features["surface"].dropna().unique()).index(
                context["surface"]
            ),
        )
        context["temp"] = st.number_input(
            "Temperature",
            min_value=-20.0,
            max_value=130.0,
            value=float(context["temp"]),
            step=1.0,
        )
        context["wind"] = st.number_input(
            "Wind",
            min_value=0.0,
            max_value=80.0,
            value=float(context["wind"]),
            step=1.0,
        )

    prediction_row = build_prediction_row(home_team, away_team, form, context)
    probability = float(model.predict_proba(prediction_row)[0, 1])
    display_prediction(home_team, away_team, probability)

    st.dataframe(prediction_row, hide_index=True)


if __name__ == "__main__":
    main()
