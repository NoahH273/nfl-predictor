"""Create season-based train, validation, and test splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


FEATURES_PATH = Path("data/processed/features.parquet")
PROCESSED_DATA_DIR = Path("data/processed")
FEATURES_WITH_SPLITS_PATH = PROCESSED_DATA_DIR / "features_with_splits.parquet"
TRAIN_FEATURES_PATH = PROCESSED_DATA_DIR / "train_features.parquet"
VALIDATION_FEATURES_PATH = PROCESSED_DATA_DIR / "validation_features.parquet"
TEST_FEATURES_PATH = PROCESSED_DATA_DIR / "test_features.parquet"

TRAIN_SEASONS = range(2000, 2021)
VALIDATION_SEASONS = range(2021, 2024)
TEST_SEASONS = range(2024, 2026)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create chronological season-based modeling splits."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=FEATURES_PATH,
        help=f"Feature dataset path. Defaults to {FEATURES_PATH}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help=f"Directory for split outputs. Defaults to {PROCESSED_DATA_DIR}.",
    )
    return parser.parse_args()


def add_split_column(features: pl.DataFrame) -> pl.DataFrame:
    if "season" not in features.columns:
        raise ValueError("Feature dataset is missing required column: season")

    return features.with_columns(
        pl.when(pl.col("season").is_in(TRAIN_SEASONS))
        .then(pl.lit("train"))
        .when(pl.col("season").is_in(VALIDATION_SEASONS))
        .then(pl.lit("validation"))
        .when(pl.col("season").is_in(TEST_SEASONS))
        .then(pl.lit("test"))
        .otherwise(None)
        .alias("split")
    )


def write_split_files(
    features_with_splits: pl.DataFrame,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": output_dir / TRAIN_FEATURES_PATH.name,
        "validation": output_dir / VALIDATION_FEATURES_PATH.name,
        "test": output_dir / TEST_FEATURES_PATH.name,
    }

    features_with_splits.write_parquet(output_dir / FEATURES_WITH_SPLITS_PATH.name)

    counts = {"all": features_with_splits.height}
    for split_name, split_path in split_paths.items():
        split_df = features_with_splits.filter(pl.col("split") == split_name)
        split_df.write_parquet(split_path)
        counts[split_name] = split_df.height

    unassigned_rows = features_with_splits.filter(pl.col("split").is_null()).height
    counts["unassigned"] = unassigned_rows
    return counts


def split_dataset(
    input_path: Path = FEATURES_PATH,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, int]:
    features = pl.read_parquet(input_path)
    features_with_splits = add_split_column(features)
    return write_split_files(features_with_splits, output_dir)


def main() -> None:
    args = parse_args()
    counts = split_dataset(args.input_path, args.output_dir)

    print("Created season-based modeling splits")
    print("Train seasons: 2000-2020")
    print("Validation seasons: 2021-2023")
    print("Test seasons: 2024-2025")
    print(f"All rows: {counts['all']:,}")
    print(f"Train rows: {counts['train']:,}")
    print(f"Validation rows: {counts['validation']:,}")
    print(f"Test rows: {counts['test']:,}")
    print(f"Unassigned rows: {counts['unassigned']:,}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
