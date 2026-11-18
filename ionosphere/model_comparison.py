"""Compare Ionosphere classifiers with fixed and nested cross-validation."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

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
    """Create the baseline and initial set of classifiers."""
    return {
        "Dummy baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        "Linear SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", C=1.0),
        ),
        "RBF SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale"),
        ),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def build_parameter_grids() -> dict[str, dict[str, list[float | int | str]]]:
    """Return bounded hyperparameter grids for the nested comparison."""
    return {
        "Dummy baseline": {},
        "Logistic Regression": {
            "logisticregression__C": [0.1, 1.0, 10.0],
        },
        "Linear SVM": {
            "svc__C": [0.1, 1.0, 10.0],
        },
        "RBF SVM": {
            "svc__C": [0.1, 1.0, 10.0],
            "svc__gamma": ["scale", 0.01],
        },
        "K-Nearest Neighbors": {
            "kneighborsclassifier__n_neighbors": [3, 5, 7, 9],
            "kneighborsclassifier__weights": ["uniform", "distance"],
        },
        "Random Forest": {
            "max_features": ["sqrt", 0.5],
            "n_estimators": [200, 300],
        },
        "Gradient Boosting": {
            "learning_rate": [0.05, 0.1],
            "n_estimators": [100, 200],
        },
        "XGBoost": {
            "max_depth": [2, 3],
            "n_estimators": [100, 200],
        },
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
            n_jobs=1,
        )

        row = {"model": model_name}
        for metric in SCORING:

            scores = fold_scores[f"test_{metric}"]

            row[f"{metric}_mean"] = float(np.mean(scores))

            row[f"{metric}_std"] = float(np.std(scores, ddof=1))

        rows.append(row)

    return pd.DataFrame(rows).set_index("model")


def evaluate_models_nested(
    data_path: str | Path = DEFAULT_DATA_PATH,
    n_splits: int = DEFAULT_CV_SPLITS,
    inner_splits: int = DEFAULT_CV_SPLITS,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, object]]]]:

    """Tune each model in inner folds and score it on outer held-out folds."""


    features, labels = load_dataset(data_path)

    outer_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE + 1,
    )
    inner_cv = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = build_models()

    parameter_grids = build_parameter_grids()

    rows = []

    best_parameters: dict[str, list[dict[str, object]]] = {}

    for model_name, model in models.items():
        search = GridSearchCV(
            estimator=model,
            param_grid=parameter_grids[model_name],
            scoring="roc_auc",
            cv=inner_cv,
            refit=True,
            n_jobs=1,
            error_score="raise",
        )
        outer_scores = cross_validate(
            search,
            features,
            labels,
            cv=outer_cv,
            scoring=SCORING,
            return_estimator=True,
            n_jobs=1,
            error_score="raise",
        )
        row = {"model": model_name}

        for metric in SCORING:

            scores = outer_scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(scores))
            row[f"{metric}_std"] = float(np.std(scores, ddof=1))

        rows.append(row)

        best_parameters[model_name] = [
            dict(fitted_search.best_params_)
            for fitted_search in outer_scores["estimator"]
        ]

    return pd.DataFrame(rows).set_index("model"), best_parameters


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)

    parser.add_argument("--folds", type=int, default=DEFAULT_CV_SPLITS)

    parser.add_argument(
        "--nested",
        action="store_true",
        help="Tune each model in inner folds and evaluate it on outer folds.",
    )

    parser.add_argument("--inner-folds", type=int, default=DEFAULT_CV_SPLITS)

    args = parser.parse_args()

    if args.nested:
        results, best_parameters = evaluate_models_nested(
            args.data_path,
            n_splits=args.folds,
            inner_splits=args.inner_folds,
        )
        print(
            f"Nested stratified CV ({args.folds} outer x "
            f"{args.inner_folds} inner folds; mean and std):"
        )
    else:

        results = evaluate_models(args.data_path, n_splits=args.folds)

        print(f"Stratified {args.folds}-fold cross-validation (mean and std):")

    print(results.round(3).to_string())
    if args.nested:

        print("Best inner-CV parameters by outer fold:")

        for model_name, fold_parameters in best_parameters.items():

            print(f"  {model_name}: {fold_parameters}")


if __name__ == "__main__":
    main()
