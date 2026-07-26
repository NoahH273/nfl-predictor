"""Create the cleaned game-level modeling dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


RAW_GAMES_PATH = Path("data/raw/games.parquet")
PROCESSED_DATA_DIR = Path("data/processed")
MODELING_DATASET_PATH = PROCESSED_DATA_DIR / "modeling_dataset.parquet"

OUTPUT_COLUMNS = [
    "game_id",
    "season",
    "week",
    "gameday",
    "weekday",
    "gametime",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_win",
    "overtime",
    "div_game",
    "roof",
    "surface",
    "temp",
    "wind",
    "stadium",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean raw NFL games into a game-level modeling dataset."
    )
    parser.add_argument(
        "--games-path",
        type=Path,
        default=RAW_GAMES_PATH,
        help=f"Raw games Parquet file. Defaults to {RAW_GAMES_PATH}.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=MODELING_DATASET_PATH,
        help=f"Processed dataset output path. Defaults to {MODELING_DATASET_PATH}.",
    )
    return parser.parse_args()


def clean_games(games: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Filter completed regular-season games, remove ties, and add home_win."""
    required_columns = {
        "game_id",
        "season",
        "game_type",
        "week",
        "gameday",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    }
    missing_columns = sorted(required_columns - set(games.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Raw games data is missing required columns: {missing}")

    initial_rows = games.height

    regular_season_games = games.filter(pl.col("game_type") == "REG")
    completed_games = regular_season_games.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    non_tie_games = completed_games.filter(pl.col("home_score") != pl.col("away_score"))

    cleaned_games = (
        non_tie_games.with_columns(
            (pl.col("home_score") > pl.col("away_score"))
            .cast(pl.Int8)
            .alias("home_win"),
            pl.col("gameday").str.to_date(strict=False).alias("gameday_sort"),
        )
        .select(OUTPUT_COLUMNS + ["gameday_sort"])
        .sort(["season", "week", "gameday_sort", "game_id"])
        .drop("gameday_sort")
    )

    counts = {
        "initial_rows": initial_rows,
        "non_regular_season_rows_removed": initial_rows - regular_season_games.height,
        "missing_score_rows_removed": regular_season_games.height - completed_games.height,
        "tie_rows_removed": completed_games.height - non_tie_games.height,
        "final_rows": cleaned_games.height,
    }

    return cleaned_games, counts


def make_dataset(
    games_path: Path = RAW_GAMES_PATH,
    output_path: Path = MODELING_DATASET_PATH,
) -> dict[str, int]:
    games = pl.read_parquet(games_path)
    cleaned_games, counts = clean_games(games)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_games.write_parquet(output_path)

    return counts


def main() -> None:
    args = parse_args()
    counts = make_dataset(args.games_path, args.output_path)

    print("Created cleaned game-level modeling dataset")
    print(f"Raw rows: {counts['initial_rows']:,}")
    print(
        "Removed non-regular-season rows: "
        f"{counts['non_regular_season_rows_removed']:,}"
    )
    print(f"Removed rows missing scores: {counts['missing_score_rows_removed']:,}")
    print(f"Removed tied games: {counts['tie_rows_removed']:,}")
    print(f"Saved rows: {counts['final_rows']:,}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
