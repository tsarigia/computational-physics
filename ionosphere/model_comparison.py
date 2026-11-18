"""Compare baseline and logistic-regression models with stratified cross-validation."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    
    from .data_utils import DEFAULT_DATA_PATH, load_dataset
else:
    from data_utils import DEFAULT_DATA_PATH, load_dataset

RANDOM_STATE = 42
DEFAULT_CV_SPLITS = 5

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def build_models() -> dict[str, BaseEstimator]:
    """Create the baseline and scaled logistic-regression classifiers."""
    return {
        "Dummy baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
    }


def evaluate_models(
    data_path: str | Path = DEFAULT_DATA_PATH,
    n_splits: int = DEFAULT_CV_SPLITS,
) -> pd.DataFrame:
    
    """Return mean and standard deviation of stratified CV metrics per model."""
    
    features, labels = load_dataset(data_path)
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    rows = []

    for model_name, model in build_models().items():
        fold_scores = cross_validate(
            model,
            features,
            labels,
            cv=cv,
            scoring=SCORING,
            return_train_score=False,
        )
        
        row = {"model": model_name}
        for metric in SCORING:
            
            scores = fold_scores[f"test_{metric}"]
            
            row[f"{metric}_mean"] = float(np.mean(scores))
            
            row[f"{metric}_std"] = float(np.std(scores, ddof=1))
            
        rows.append(row)

    return pd.DataFrame(rows).set_index("model")


def main() -> None:
    
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    
    parser.add_argument("--folds", type=int, default=DEFAULT_CV_SPLITS)
    
    args = parser.parse_args()

    results = evaluate_models(args.data_path, n_splits=args.folds)
    
    print(f"Stratified {args.folds}-fold cross-validation (mean and std):")
    print(results.round(3).to_string())


if __name__ == "__main__":
    main()
