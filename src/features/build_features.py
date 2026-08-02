"""Build model-ready matchup and rolling team features."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl


MODELING_DATASET_PATH = Path("data/processed/modeling_dataset.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")
ROLLING_WINDOWS = (3, 5)

REQUIRED_COLUMNS = {
    "game_id",
    "season",
    "week",
    "gameday",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_win",
    "div_game",
    "roof",
    "surface",
    "temp",
    "wind",
}

BASIC_OUTPUT_COLUMNS = [
    "game_id",
    "season",
    "week",
    "gameday",
    "home_team",
    "away_team",
    "matchup",
    "home_team_code",
    "away_team_code",
    "is_division_game",
    "roof",
    "surface",
    "temp",
    "wind",
    "home_score",
    "away_score",
    "home_win",
]


def rolling_feature_columns() -> list[str]:
    columns = []
    for side in ("home", "away"):
        for window in ROLLING_WINDOWS:
            columns.extend(
                [
                    f"{side}_rolling_win_pct_{window}",
                    f"{side}_rolling_points_scored_{window}",
                    f"{side}_rolling_points_allowed_{window}",
                ]
            )
    return columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build matchup and rolling team features for NFL game prediction."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=MODELING_DATASET_PATH,
        help=f"Cleaned game dataset. Defaults to {MODELING_DATASET_PATH}.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=FEATURES_PATH,
        help=f"Feature dataset output path. Defaults to {FEATURES_PATH}.",
    )
    return parser.parse_args()


def validate_columns(df: pl.DataFrame) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Modeling dataset is missing required columns: {missing}")


def build_basic_features(games: pl.DataFrame) -> pl.DataFrame:
    """Create basic matchup features."""
    validate_columns(games)

    teams = (
        games.select(pl.col("home_team").alias("team"))
        .vstack(games.select(pl.col("away_team").alias("team")))
        .unique()
        .sort("team")
        .with_row_index("team_code")
    )

    home_team_codes = teams.rename(
        {"team": "home_team", "team_code": "home_team_code"}
    )
    away_team_codes = teams.rename(
        {"team": "away_team", "team_code": "away_team_code"}
    )

    features = (
        games.join(home_team_codes, on="home_team", how="left")
        .join(away_team_codes, on="away_team", how="left")
        .with_columns(
            (pl.col("home_team") + "_vs_" + pl.col("away_team")).alias("matchup"),
            pl.col("div_game").cast(pl.Int8).alias("is_division_game"),
            pl.col("gameday").str.to_date(strict=False).alias("gameday_sort"),
        )
        .select(BASIC_OUTPUT_COLUMNS + ["gameday_sort"])
        .sort(["season", "week", "gameday_sort", "game_id"])
        .drop("gameday_sort")
    )

    return features


def build_rolling_features(games: pl.DataFrame) -> pl.DataFrame:
    """Create prior-game rolling team features and return one row per matchup."""
    validate_columns(games)

    games_df = games.to_pandas()
    games_df["gameday_sort"] = pd.to_datetime(games_df["gameday"], errors="coerce")
    games_df = games_df.sort_values(
        ["season", "week", "gameday_sort", "game_id"], kind="mergesort"
    )

    home_rows = pd.DataFrame(
        {
            "game_id": games_df["game_id"],
            "season": games_df["season"],
            "week": games_df["week"],
            "gameday_sort": games_df["gameday_sort"],
            "team": games_df["home_team"],
            "is_home": True,
            "win": games_df["home_win"],
            "points_scored": games_df["home_score"],
            "points_allowed": games_df["away_score"],
        }
    )
    away_rows = pd.DataFrame(
        {
            "game_id": games_df["game_id"],
            "season": games_df["season"],
            "week": games_df["week"],
            "gameday_sort": games_df["gameday_sort"],
            "team": games_df["away_team"],
            "is_home": False,
            "win": 1 - games_df["home_win"],
            "points_scored": games_df["away_score"],
            "points_allowed": games_df["home_score"],
        }
    )

    team_games = (
        pd.concat([home_rows, away_rows], ignore_index=True)
        .sort_values(
            ["team", "season", "week", "gameday_sort", "game_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    grouped = team_games.groupby("team", group_keys=False)
    source_columns = ["win", "points_scored", "points_allowed"]
    for window in ROLLING_WINDOWS:
        for source_column in source_columns:
            output_column = f"rolling_{source_column}_{window}"
            team_games[output_column] = grouped[source_column].transform(
                lambda values: values.shift(1).rolling(window, min_periods=1).mean()
            )

    rename_map = {
        f"rolling_win_{window}": f"rolling_win_pct_{window}"
        for window in ROLLING_WINDOWS
    }
    team_games = team_games.rename(columns=rename_map)

    rolling_columns = [
        f"rolling_win_pct_{window}" for window in ROLLING_WINDOWS
    ] + [
        f"rolling_points_scored_{window}" for window in ROLLING_WINDOWS
    ] + [
        f"rolling_points_allowed_{window}" for window in ROLLING_WINDOWS
    ]

    home_rolling = team_games[team_games["is_home"]][
        ["game_id", *rolling_columns]
    ].rename(columns={column: f"home_{column}" for column in rolling_columns})
    away_rolling = team_games[~team_games["is_home"]][
        ["game_id", *rolling_columns]
    ].rename(columns={column: f"away_{column}" for column in rolling_columns})

    matchup_rolling = games_df[["game_id"]].merge(home_rolling, on="game_id", how="left")
    matchup_rolling = matchup_rolling.merge(away_rolling, on="game_id", how="left")

    return pl.from_pandas(matchup_rolling)


def build_features(games: pl.DataFrame) -> pl.DataFrame:
    """Create basic matchup features plus prior-game rolling team features."""
    basic_features = build_basic_features(games)
    rolling_features = build_rolling_features(games)

    features = basic_features.join(rolling_features, on="game_id", how="left")
    return features.select(BASIC_OUTPUT_COLUMNS + rolling_feature_columns())


def make_features(
    input_path: Path = MODELING_DATASET_PATH,
    output_path: Path = FEATURES_PATH,
) -> dict[str, int]:
    games = pl.read_parquet(input_path)
    features = build_features(games)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(output_path)

    return {
        "input_rows": games.height,
        "output_rows": features.height,
        "output_columns": features.width,
    }


def main() -> None:
    args = parse_args()
    counts = make_features(args.input_path, args.output_path)

    print("Built matchup and rolling feature dataset")
    print(f"Input rows: {counts['input_rows']:,}")
    print(f"Output rows: {counts['output_rows']:,}")
    print(f"Output columns: {counts['output_columns']:,}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
