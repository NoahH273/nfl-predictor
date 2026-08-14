"""Train and evaluate the logistic regression baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TRAIN_PATH = Path("data/processed/train_features.parquet")
VALIDATION_PATH = Path("data/processed/validation_features.parquet")
TEST_PATH = Path("data/processed/test_features.parquet")
ARTIFACTS_DIR = Path("artifacts/logistic_regression")

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
VALIDATION_PREDICTIONS_PATH = ARTIFACTS_DIR / "validation_predictions.parquet"
TEST_PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.parquet"

TARGET_COLUMN = "home_win"
IDENTIFIER_COLUMNS = [
    "game_id",
    "gameday",
    "matchup",
    "home_score",
    "away_score",
    "split",
]

NUMERIC_FEATURES = [
    "season",
    "week",
    "is_division_game",
    "temp",
    "wind",
    "home_rolling_win_pct_3",
    "home_rolling_points_scored_3",
    "home_rolling_points_allowed_3",
    "home_rolling_win_pct_5",
    "home_rolling_points_scored_5",
    "home_rolling_points_allowed_5",
    "away_rolling_win_pct_3",
    "away_rolling_points_scored_3",
    "away_rolling_points_allowed_3",
    "away_rolling_win_pct_5",
    "away_rolling_points_scored_5",
    "away_rolling_points_allowed_5",
]

CATEGORICAL_FEATURES = [
    "home_team",
    "away_team",
    "roof",
    "surface",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the logistic regression baseline model."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=TRAIN_PATH,
        help=f"Training split path. Defaults to {TRAIN_PATH}.",
    )
    parser.add_argument(
        "--validation-path",
        type=Path,
        default=VALIDATION_PATH,
        help=f"Validation split path. Defaults to {VALIDATION_PATH}.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=TEST_PATH,
        help=f"Test split path. Defaults to {TEST_PATH}.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help=f"Output directory for model artifacts. Defaults to {ARTIFACTS_DIR}.",
    )
    return parser.parse_args()


def validate_features(df: pd.DataFrame) -> None:
    required_columns = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN])
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Feature data is missing required columns: {missing}")


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    validate_features(df)
    return df


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(max_iter=1_000, random_state=42),
            ),
        ]
    )


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return df[feature_columns], df[TARGET_COLUMN]


def evaluate_model(
    model: Pipeline,
    df: pd.DataFrame,
    split_name: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    x, y = split_xy(df)
    probabilities = model.predict_proba(x)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "roc_auc": roc_auc_score(y, probabilities),
        "log_loss": log_loss(y, probabilities),
        "rows": len(df),
    }

    prediction_columns = [
        column
        for column in ["game_id", "season", "week", "gameday", "home_team", "away_team"]
        if column in df.columns
    ]
    prediction_df = df[prediction_columns].copy()
    prediction_df["split"] = split_name
    prediction_df["actual_home_win"] = y.to_numpy()
    prediction_df["predicted_home_win"] = predictions
    prediction_df["home_win_probability"] = probabilities

    return metrics, prediction_df


def train_logistic_regression(
    train_path: Path = TRAIN_PATH,
    validation_path: Path = VALIDATION_PATH,
    test_path: Path = TEST_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> dict[str, dict[str, float]]:
    train_df = load_split(train_path)
    validation_df = load_split(validation_path)
    test_df = load_split(test_path)

    x_train, y_train = split_xy(train_df)
    model = build_pipeline()
    model.fit(x_train, y_train)

    validation_metrics, validation_predictions = evaluate_model(
        model, validation_df, "validation"
    )
    test_metrics, test_predictions = evaluate_model(model, test_df, "test")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / MODEL_PATH.name)
    validation_predictions.to_parquet(
        artifacts_dir / VALIDATION_PREDICTIONS_PATH.name, index=False
    )
    test_predictions.to_parquet(
        artifacts_dir / TEST_PREDICTIONS_PATH.name, index=False
    )

    metrics = {
        "model": "Logistic Regression",
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "train": {
            "rows": len(train_df),
            "seasons": [
                int(train_df["season"].min()),
                int(train_df["season"].max()),
            ],
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }
    (artifacts_dir / METRICS_PATH.name).write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    return metrics


def print_metrics(metrics: dict, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
    print("Trained logistic regression baseline")
    print(
        "Train rows: "
        f"{metrics['train']['rows']:,} "
        f"({metrics['train']['seasons'][0]}-{metrics['train']['seasons'][1]})"
    )
    for split_name in ("validation", "test"):
        split_metrics = metrics[split_name]
        print(
            f"{split_name.title()} rows: {split_metrics['rows']:,} | "
            f"Accuracy: {split_metrics['accuracy']:.3f} | "
            f"ROC-AUC: {split_metrics['roc_auc']:.3f} | "
            f"Log Loss: {split_metrics['log_loss']:.3f}"
        )
    print(f"Artifacts: {artifacts_dir}")


def main() -> None:
    args = parse_args()
    metrics = train_logistic_regression(
        args.train_path,
        args.validation_path,
        args.test_path,
        args.artifacts_dir,
    )
    print_metrics(metrics, args.artifacts_dir)


if __name__ == "__main__":
    main()
