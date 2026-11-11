"""Tune and evaluate a gradient-boosted decision tree for the MAGIC dataset."""

import argparse
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

if __package__:
    from .data_utils import (
        DEFAULT_DATA_PATH,
        RANDOM_STATE,
        evaluate_classifier,
        load_dataset,
        split_dataset,
    )
else:
    from data_utils import (
        DEFAULT_DATA_PATH,
        RANDOM_STATE,
        evaluate_classifier,
        load_dataset,
        split_dataset,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to magic04.data (default: repository root).",
    )
    args = parser.parse_args()

    features, labels = load_dataset(args.data_path)
    X_train, X_test, y_train, y_test = split_dataset(features, labels)

    search = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid={
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        },
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    print(f"Best parameters: {search.best_params_}")
    evaluate_classifier("Gradient Boosting", search.best_estimator_, X_test, y_test)


if __name__ == "__main__":
    main()
