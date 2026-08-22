"""Train a PyTorch MLP for NFL game outcome prediction."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_logistic_regression import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TEST_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
    load_split,
    split_xy,
)


ARTIFACTS_DIR = Path("artifacts/pytorch")
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
MODEL_PATH = ARTIFACTS_DIR / "model.pt"
TRAINING_HISTORY_PATH = ARTIFACTS_DIR / "training_history.json"
MODEL_CONFIG_PATH = ARTIFACTS_DIR / "model_config.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
VALIDATION_PREDICTIONS_PATH = ARTIFACTS_DIR / "validation_predictions.parquet"
TEST_PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.parquet"

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_DROPOUT = 0.2
DEFAULT_SEED = 42


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the PyTorch MLP model.")
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs. Defaults to {DEFAULT_EPOCHS}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size. Defaults to {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Adam learning rate. Defaults to {DEFAULT_LEARNING_RATE}.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help=f"Hidden layer size. Defaults to {DEFAULT_HIDDEN_SIZE}.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
        help=f"Dropout probability. Defaults to {DEFAULT_DROPOUT}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Defaults to {DEFAULT_SEED}.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_preprocessor() -> ColumnTransformer:
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

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def transform_features(
    preprocessor: ColumnTransformer,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, y_train = split_xy(train_df)
    x_validation, y_validation = split_xy(validation_df)

    x_train_array = preprocessor.fit_transform(x_train).astype(np.float32)
    x_validation_array = preprocessor.transform(
        x_validation).astype(np.float32)
    y_train_array = y_train.to_numpy(dtype=np.float32)
    y_validation_array = y_validation.to_numpy(dtype=np.float32)

    return x_train_array, y_train_array, x_validation_array, y_validation_array


def transform_split(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    x, y = split_xy(df)
    x_array = preprocessor.transform(x).astype(np.float32)
    y_array = y.to_numpy(dtype=np.float32)
    return x_array, y_array


def make_data_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: GameOutcomeMLP,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0

    for features, targets in data_loader:
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = features.shape[0]
        total_loss += loss.item() * batch_size
        total_rows += batch_size

    return total_loss / total_rows


def validate_one_epoch(
    model: GameOutcomeMLP,
    data_loader: DataLoader,
    loss_fn: nn.Module,
) -> float:
    model.eval()
    total_loss = 0.0
    total_rows = 0

    with torch.no_grad():
        for features, targets in data_loader:
            logits = model(features)
            loss = loss_fn(logits, targets)

            batch_size = features.shape[0]
            total_loss += loss.item() * batch_size
            total_rows += batch_size

    return total_loss / total_rows


def train_model(
    model: GameOutcomeMLP,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    learning_rate: float,
) -> list[dict[str, float]]:
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        validation_loss = validate_one_epoch(model, validation_loader, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"validation_loss={validation_loss:.4f}"
        )

    return history


def predict_probabilities(model: GameOutcomeMLP, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        probabilities = torch.sigmoid(logits).numpy()
    return probabilities


def evaluate_model(
    model: GameOutcomeMLP,
    x: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    split_name: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    probabilities = predict_probabilities(model, x)
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
    prediction_df["actual_home_win"] = y.astype(int)
    prediction_df["predicted_home_win"] = predictions
    prediction_df["home_win_probability"] = probabilities

    return metrics, prediction_df


def train_pytorch_mlp(
    train_path: Path = TRAIN_PATH,
    validation_path: Path = VALIDATION_PATH,
    test_path: Path = TEST_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    dropout: float = DEFAULT_DROPOUT,
    seed: int = DEFAULT_SEED,
) -> dict:
    set_seed(seed)

    train_df = load_split(train_path)
    validation_df = load_split(validation_path)
    test_df = load_split(test_path)
    preprocessor = build_preprocessor()
    x_train, y_train, x_validation, y_validation = transform_features(
        preprocessor, train_df, validation_df
    )
    x_test, y_test = transform_split(preprocessor, test_df)

    train_loader = make_data_loader(x_train, y_train, batch_size, shuffle=True)
    validation_loader = make_data_loader(
        x_validation, y_validation, batch_size, shuffle=False
    )
    model = GameOutcomeMLP(
        input_dim=x_train.shape[1],
        hidden_size=hidden_size,
        dropout=dropout,
    )
    history = train_model(model, train_loader,
                          validation_loader, epochs, learning_rate)
    validation_metrics, validation_predictions = evaluate_model(
        model, x_validation, y_validation, validation_df, "validation"
    )
    test_metrics, test_predictions = evaluate_model(
        model, x_test, y_test, test_df, "test"
    )

    config = {
        "model": "PyTorch MLP",
        "input_dim": x_train.shape[1],
        "hidden_size": hidden_size,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
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
        "validation": {
            "rows": len(validation_df),
            "seasons": [
                int(validation_df["season"].min()),
                int(validation_df["season"].max()),
            ],
        },
        "test": {
            "rows": len(test_df),
            "seasons": [
                int(test_df["season"].min()),
                int(test_df["season"].max()),
            ],
        },
    }
    metrics = {
        "model": "PyTorch MLP",
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "train": config["train"],
        "validation": validation_metrics,
        "test": test_metrics,
        "final_epoch": history[-1],
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, artifacts_dir / PREPROCESSOR_PATH.name)
    torch.save(model.state_dict(), artifacts_dir / MODEL_PATH.name)
    (artifacts_dir / TRAINING_HISTORY_PATH.name).write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (artifacts_dir / MODEL_CONFIG_PATH.name).write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (artifacts_dir / METRICS_PATH.name).write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    validation_predictions.to_parquet(
        artifacts_dir / VALIDATION_PREDICTIONS_PATH.name, index=False
    )
    test_predictions.to_parquet(
        artifacts_dir / TEST_PREDICTIONS_PATH.name, index=False
    )

    return {
        "config": config,
        "history": history,
        "metrics": metrics,
    }


def print_metrics(result: dict, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
    metrics = result["metrics"]
    print("Trained PyTorch MLP")
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
    result = train_pytorch_mlp(
        train_path=args.train_path,
        validation_path=args.validation_path,
        test_path=args.test_path,
        artifacts_dir=args.artifacts_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        seed=args.seed,
    )
    print_metrics(result, args.artifacts_dir)


if __name__ == "__main__":
    main()
