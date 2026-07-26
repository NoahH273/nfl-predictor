"""Load raw NFL data with nflreadpy and save local Parquet files."""

from __future__ import annotations

import argparse
from pathlib import Path

import nflreadpy as nfl


DEFAULT_START_SEASON = 2000
DEFAULT_END_SEASON = 2025
RAW_DATA_DIR = Path("data/raw")
GAMES_OUTPUT_PATH = RAW_DATA_DIR / "games.parquet"
TEAM_STATS_OUTPUT_PATH = RAW_DATA_DIR / "team_stats.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load NFL schedules and weekly team stats with nflreadpy."
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=DEFAULT_START_SEASON,
        help=f"First season to load. Defaults to {DEFAULT_START_SEASON}.",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=DEFAULT_END_SEASON,
        help=f"Last season to load. Defaults to {DEFAULT_END_SEASON}.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help=f"Directory for raw Parquet outputs. Defaults to {RAW_DATA_DIR}.",
    )
    return parser.parse_args()


def season_range(start_season: int, end_season: int) -> list[int]:
    if start_season > end_season:
        raise ValueError(
            "start-season must be less than or equal to end-season")
    return list(range(start_season, end_season + 1))


def load_raw_data(seasons: list[int], raw_data_dir: Path = RAW_DATA_DIR) -> dict[str, int]:
    """Load schedules and weekly team stats, then save them as Parquet files."""
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    games = nfl.load_schedules(seasons)
    team_stats = nfl.load_team_stats(seasons, summary_level="week")

    games_path = raw_data_dir / GAMES_OUTPUT_PATH.name
    team_stats_path = raw_data_dir / TEAM_STATS_OUTPUT_PATH.name

    games.write_parquet(games_path)
    team_stats.write_parquet(team_stats_path)

    return {
        "games_rows": games.height,
        "games_columns": games.width,
        "team_stats_rows": team_stats.height,
        "team_stats_columns": team_stats.width,
    }


def main() -> None:
    args = parse_args()
    seasons = season_range(args.start_season, args.end_season)
    counts = load_raw_data(seasons, args.raw_data_dir)

    print("Loaded raw NFL data with nflreadpy")
    print(f"Seasons: {seasons[0]}-{seasons[-1]}")
    print(
        f"Saved {counts['games_rows']:,} schedule rows "
        f"({counts['games_columns']:,} columns) to {args.raw_data_dir / GAMES_OUTPUT_PATH.name}"
    )
    print(
        f"Saved {counts['team_stats_rows']:,} team-stat rows "
        f"({counts['team_stats_columns']:,} columns) to "
        f"{args.raw_data_dir / TEAM_STATS_OUTPUT_PATH.name}"
    )


if __name__ == "__main__":
    main()
