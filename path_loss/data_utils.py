"""Shared data loading and evaluation helpers for path-loss regression."""

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "paper.xlsx"
FEATURE_COLUMNS = ("Feature_1", "Feature_2", "Feature_3", "Feature_4")
TARGET_COLUMN = "Path_Loss"
RANDOM_STATE = 42
TEST_SIZE = 0.2


class Regressor(Protocol):
    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


def load_dataset(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the headerless Excel dataset and return features and path loss."""
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    frame = pd.read_excel(data_path, header=None, engine="openpyxl")
    if frame.empty:
        raise ValueError(f"Dataset is empty: {data_path}")
    if frame.shape[1] != len(FEATURE_COLUMNS) + 1:
        raise ValueError(
            f"Expected 5 columns (4 features and path loss), found "
            f"{frame.shape[1]} in {data_path}"
        )

    frame.columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    frame = frame.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Dataset contains missing or non-finite values: {data_path}")

    features = frame.loc[:, list(FEATURE_COLUMNS)]
    labels = frame[TARGET_COLUMN]
    if (labels == 0).any():
        raise ValueError(f"Path-loss values must be nonzero to calculate MAPE: {data_path}")
    return features, labels


def split_dataset(
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create reproducible 80/20 train/test sets."""
    return train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def regression_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
) -> tuple[float, float, float]:
    """Return MAE, RMSE, and MAPE (as a fraction)."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted)
    return float(mae), float(rmse), float(mape)


def evaluate_regressor(
    name: str,
    model: Regressor,
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[float, float, float]:
    """Print regression metrics for a fitted model and return them."""
    predictions = model.predict(features)
    mae, rmse, mape = regression_metrics(labels, predictions)
    print(f"\n{name}")
    print(f"MAE:  {mae:.4f} dB")
    print(f"RMSE: {rmse:.4f} dB")
    print(f"MAPE: {mape:.2%}")
    return mae, rmse, mape
