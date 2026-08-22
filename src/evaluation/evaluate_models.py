"""Standardized model evaluation utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


ARTIFACTS_ROOT = Path("artifacts")
DEFAULT_OUTPUT_PATH = ARTIFACTS_ROOT / "model_comparison_metrics.json"
DEFAULT_MODEL_NAMES = ("logistic_regression", "xgboost", "pytorch")
DEFAULT_SPLITS = ("validation", "test")

ACTUAL_COLUMN = "actual_home_win"
PROBABILITY_COLUMN = "home_win_probability"
PREDICTION_COLUMN = "predicted_home_win"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute standardized metrics from saved model predictions."
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=ARTIFACTS_ROOT,
        help=f"Model artifacts root directory. Defaults to {ARTIFACTS_ROOT}.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Metrics JSON output path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_NAMES),
        help="Model artifact directory names to evaluate.",
    )
    return parser.parse_args()


def calculate_binary_classification_metrics(
    y_true: Iterable[float],
    y_probability: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate the standard binary classification metrics for this project."""
    y_true_array = np.asarray(y_true, dtype=int)
    y_probability_array = np.asarray(y_probability, dtype=float)
    y_pred_array = (y_probability_array >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true_array, y_pred_array),
        "roc_auc": roc_auc_score(y_true_array, y_probability_array),
        "log_loss": log_loss(y_true_array, y_probability_array),
        "rows": int(len(y_true_array)),
    }


def validate_prediction_frame(predictions: pd.DataFrame) -> None:
    required_columns = {ACTUAL_COLUMN, PROBABILITY_COLUMN}
    missing_columns = sorted(required_columns - set(predictions.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Prediction data is missing required columns: {missing}")


def evaluate_prediction_frame(predictions: pd.DataFrame) -> dict[str, float]:
    """Calculate project metrics from a saved prediction frame."""
    validate_prediction_frame(predictions)
    return calculate_binary_classification_metrics(
        predictions[ACTUAL_COLUMN],
        predictions[PROBABILITY_COLUMN],
    )


def evaluate_prediction_file(prediction_path: Path) -> dict[str, float]:
    predictions = pd.read_parquet(prediction_path)
    return evaluate_prediction_frame(predictions)


def evaluate_model_artifacts(
    model_name: str,
    artifacts_root: Path = ARTIFACTS_ROOT,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> dict[str, dict[str, float]]:
    model_dir = artifacts_root / model_name
    split_metrics = {}

    for split_name in splits:
        prediction_path = model_dir / f"{split_name}_predictions.parquet"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {prediction_path}")
        split_metrics[split_name] = evaluate_prediction_file(prediction_path)

    return split_metrics


def evaluate_all_models(
    model_names: Iterable[str] = DEFAULT_MODEL_NAMES,
    artifacts_root: Path = ARTIFACTS_ROOT,
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        model_name: evaluate_model_artifacts(model_name, artifacts_root)
        for model_name in model_names
    }


def write_metrics(metrics: dict, output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    metrics = evaluate_all_models(args.models, args.artifacts_root)
    write_metrics(metrics, args.output_path)

    print("Wrote standardized model metrics")
    print(f"Models: {', '.join(args.models)}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
