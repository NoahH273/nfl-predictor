"""Train and evaluate the XGBoost model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from train_logistic_regression import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TEST_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
    load_split,
    split_xy,
)


ARTIFACTS_DIR = Path("artifacts/xgboost")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.parquet"
VALIDATION_PREDICTIONS_PATH = ARTIFACTS_DIR / "validation_predictions.parquet"
TEST_PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the XGBoost model.")
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


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


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


def extract_feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    xgb_model = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = xgb_model.feature_importances_

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )
    feature_importance["feature"] = (
        feature_importance["feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
    )
    return feature_importance.sort_values("importance", ascending=False).reset_index(
        drop=True
    )


def train_xgboost(
    train_path: Path = TRAIN_PATH,
    validation_path: Path = VALIDATION_PATH,
    test_path: Path = TEST_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> dict:
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
    feature_importance = extract_feature_importance(model)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / MODEL_PATH.name)
    validation_predictions.to_parquet(
        artifacts_dir / VALIDATION_PREDICTIONS_PATH.name, index=False
    )
    test_predictions.to_parquet(
        artifacts_dir / TEST_PREDICTIONS_PATH.name, index=False
    )
    feature_importance.to_parquet(
        artifacts_dir / FEATURE_IMPORTANCE_PATH.name, index=False
    )

    metrics = {
        "model": "XGBoost",
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
        "top_features": feature_importance.head(10).to_dict(orient="records"),
    }
    (artifacts_dir / METRICS_PATH.name).write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    return metrics


def print_metrics(metrics: dict, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
    print("Trained XGBoost model")
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
    print("Top features:")
    for feature in metrics["top_features"][:5]:
        print(f"- {feature['feature']}: {feature['importance']:.4f}")
    print(f"Artifacts: {artifacts_dir}")


def main() -> None:
    args = parse_args()
    metrics = train_xgboost(
        args.train_path,
        args.validation_path,
        args.test_path,
        args.artifacts_dir,
    )
    print_metrics(metrics, args.artifacts_dir)


if __name__ == "__main__":
    main()
