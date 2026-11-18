"""Load and validate the UCI Ionosphere classification dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "ionosphere.data"

FEATURE_COLUMNS = tuple(f"Attribute_{index:02d}" for index in range(1, 35))

CLASS_COLUMN = "class"
CLASS_MAPPING = {"b": 0, "g": 1}


def load_dataset(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    
    
    """Load 34 numeric attributes and encode bad/good returns as 0/1."""
    
    data_path = Path(data_path)
    
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    frame = pd.read_csv(data_path, header=None)
    
    if frame.empty:
        
        raise ValueError(f"Dataset is empty: {data_path}")
    
    if frame.shape[1] != len(FEATURE_COLUMNS) + 1:
        
        raise ValueError(
            f"Expected 35 columns (34 features and class), found "
            f"{frame.shape[1]} in {data_path}"
        )

    labels = frame.iloc[:, -1].astype("string").str.strip().str.lower()
    
    encoded_labels = labels.map(CLASS_MAPPING)
    
    if encoded_labels.isna().any():
        
        unknown = sorted(labels[encoded_labels.isna()].dropna().unique().tolist())
        
        raise ValueError(f"Unexpected class labels in {data_path}: {unknown}")
    
    if set(encoded_labels.unique()) != {0, 1}:
        
        raise ValueError(f"Both classes 'b' and 'g' are required in {data_path}")

    features = frame.iloc[:, :-1].copy()
    
    features.columns = FEATURE_COLUMNS
    
    features = features.apply(pd.to_numeric, errors="raise")
    
    if not np.isfinite(features.to_numpy()).all():
        
        raise ValueError(f"Dataset contains missing or non-finite features: {data_path}")

    encoded_labels.name = CLASS_COLUMN
    
    return features, encoded_labels.astype("int64")
