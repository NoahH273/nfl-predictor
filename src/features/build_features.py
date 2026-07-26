"""Build model-ready matchup features from the cleaned game dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


MODELING_DATASET_PATH = Path("data/processed/modeling_dataset.parquet")
BASIC_FEATURES_PATH = Path("data/processed/basic_features.parquet")

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

OUTPUT_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build basic matchup features for NFL game prediction."
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
        default=BASIC_FEATURES_PATH,
        help=f"Feature dataset output path. Defaults to {BASIC_FEATURES_PATH}.",
    )
    return parser.parse_args()


def validate_columns(df: pl.DataFrame) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Modeling dataset is missing required columns: {missing}")


def build_basic_features(games: pl.DataFrame) -> pl.DataFrame:
    """Create basic matchup features without rolling performance stats."""
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
        .select(OUTPUT_COLUMNS + ["gameday_sort"])
        .sort(["season", "week", "gameday_sort", "game_id"])
        .drop("gameday_sort")
    )

    return features


def make_basic_features(
    input_path: Path = MODELING_DATASET_PATH,
    output_path: Path = BASIC_FEATURES_PATH,
) -> dict[str, int]:
    games = pl.read_parquet(input_path)
    features = build_basic_features(games)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(output_path)

    return {
        "input_rows": games.height,
        "output_rows": features.height,
        "output_columns": features.width,
    }


def main() -> None:
    args = parse_args()
    counts = make_basic_features(args.input_path, args.output_path)

    print("Built basic matchup feature dataset")
    print(f"Input rows: {counts['input_rows']:,}")
    print(f"Output rows: {counts['output_rows']:,}")
    print(f"Output columns: {counts['output_columns']:,}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
