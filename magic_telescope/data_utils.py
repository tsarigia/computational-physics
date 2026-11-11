"""Shared data loading and evaluation helpers for the MAGIC dataset."""

from pathlib import Path
from typing import Any

# Import tools for model evaluation
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Finds the directory where this file is located
PROJECT_DIR = Path(__file__).resolve().parent

# Defines the path to the data file
DEFAULT_DATA_PATH = PROJECT_DIR / "magic04.data"

# The names of the 10 features (columns) of the dataset
FEATURE_COLUMNS = (
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
)
# The name of the target column (what we want to predict)
CLASS_COLUMN = "class"
# Locks the randomness to ensure reproducible results
RANDOM_STATE = 42
# The percentage of data allocated to the Test set (30%)
TEST_SIZE = 0.3


def load_dataset(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series]:# here we return a tuple of features and labels
    """Load the MAGIC data, encoding gamma as 1 and hadron as 0."""
    
    # Ensure the path is a Path object
    data_path = Path(data_path)
    
    if not data_path.is_file():# Check: If the file does not exist, raise an error
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
# Loads the CSV. Since it has no headers, we apply them from the constants
    frame = pd.read_csv(
        data_path,
        header=None,
        names=[*FEATURE_COLUMNS, CLASS_COLUMN],
    )
    # Check: If the file is empty, raise an error
    if frame.empty:
        raise ValueError(f"Dataset is empty: {data_path}")
    
    # Converts the class to binary: 'g' (gamma) -> 1, 'h' (hadron) -> 0
    labels = frame[CLASS_COLUMN].map({"g": 1, "h": 0})
    
    # Check: If there was an unknown label (e.g., 'x') it becomes NaN. We detect this and raise an error.
    if labels.isna().any():
        unknown_labels = sorted(frame.loc[labels.isna(), CLASS_COLUMN].unique())
        raise ValueError(f"Unexpected class labels in {data_path}: {unknown_labels}")
    
    # Isolates the feature columns and ensures they are all numeric
    features = frame.loc[:, list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="raise")
    
    # Check: Ensures there are no missing (NaN) or infinite (inf) numbers
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError(f"Dataset contains missing or non-finite feature values: {data_path}")
    
    # Returns the features (X) and labels (y) as integers
    return features, labels.astype("int64")


def split_dataset(
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create reproducible train/test sets while preserving class proportions."""
    return train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,# IMPORTANT: Maintains an equal percentage of Gamma/Hadron in both sets
    )


def evaluate_classifier(
    name: str,
    model: Any,
    features: pd.DataFrame,
    labels: pd.Series,
) -> None:
    """Print classification metrics; gamma (label 1) is the positive class."""
    
    # The model makes predictions (1 or 0) for the data
    predictions = model.predict(features)
    
    # Attempts to find the probabilities (how % confident the model is)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[:, 1]# Gets the probability for class 1
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(features)
    else:
        scores = None

    print(f"\n{name}")
    print(f"Accuracy:  {accuracy_score(labels, predictions):.4f}")
    print(f"Precision: {precision_score(labels, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(labels, predictions, zero_division=0):.4f}")
    if scores is not None:
        print(f"ROC-AUC:   {roc_auc_score(labels, scores):.4f}")
        
    # Creates the Confusion Matrix, locking 0 (hadron) to the first row/col and 1 (gamma) to the second
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    print("Confusion matrix (rows: actual hadron/gamma; columns: predicted hadron/gamma):")
    print(matrix)
