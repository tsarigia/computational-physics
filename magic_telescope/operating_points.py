"""Select a MAGIC classifier and evaluate low-background operating points."""

import argparse
from math import sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

if __package__:
    from .data_utils import DEFAULT_DATA_PATH, RANDOM_STATE, load_dataset, split_dataset
else:
    from data_utils import DEFAULT_DATA_PATH, RANDOM_STATE, load_dataset, split_dataset

TARGET_ACCEPTANCE = (0.01, 0.02, 0.05, 0.10, 0.20)


def select_threshold(
    labels: np.ndarray, scores: np.ndarray, target: float
) -> float:
    
    """Choose the most permissive threshold satisfying the validation limit."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("Labels and scores must be equally sized 1D arrays.")
    
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Both hadron (0) and gamma (1) labels are required.")
    
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")
    
    if not 0 <= target <= 1:
        raise ValueError("Target acceptance must be between 0 and 1.")

    fpr, _, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    
    return float(thresholds[np.flatnonzero(fpr <= target)[-1]])


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson interval for a binomial proportion."""
    
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Counts must satisfy 0 <= successes <= total and total > 0.")
    
    z = NormalDist().inv_cdf(0.975)
    p = successes / total
    
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    
    half_width = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    
    return max(0.0, center - half_width / denominator), min(
        1.0, center + half_width / denominator
    )


def operating_point(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> tuple[int, int, int, int]:
    
    """Count accepted hadrons and gammas at a fixed threshold."""
    accepted = scores >= threshold
    hadrons = labels == 0
    gammas = labels == 1
    
    return (
        int(np.count_nonzero(accepted & hadrons)),
        int(np.count_nonzero(hadrons)),
        int(np.count_nonzero(accepted & gammas)),
        int(np.count_nonzero(gammas)),
    )


def main() -> None:
    
    parser = argparse.ArgumentParser(description=__doc__)
    
    
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    features, labels = load_dataset(args.data_path)
    
    X_development, X_test, y_development, y_test = split_dataset(features, labels)
    
    X_train, X_validation, y_train, y_validation = train_test_split(
        
        X_development,
        y_development,
        
        test_size=0.25,
        
        random_state=RANDOM_STATE,
        stratify=y_development,
    )

    pipeline = Pipeline([("scale", "passthrough"), ("model", DecisionTreeClassifier())])
    
    parameter_grid = [
        {
            "model": [DecisionTreeClassifier(random_state=RANDOM_STATE)],
            "model__min_samples_leaf": [10, 50],
        },
        {
            "model": [
                RandomForestClassifier(
                    n_estimators=100, random_state=RANDOM_STATE, n_jobs=1
                )
            ],
            "model__min_samples_leaf": [1, 5],
        },
        {
            "model": [GradientBoostingClassifier(random_state=RANDOM_STATE)],
            "model__n_estimators": [100],
            "model__max_depth": [2, 3],
        },
        {
            "scale": [StandardScaler()],
            "model": [SVC(kernel="rbf")],
            "model__C": [1, 10],
        },
    ]
    search = GridSearchCV(
        pipeline,
        parameter_grid,
        scoring=make_scorer(
            roc_auc_score,
            response_method=("decision_function", "predict_proba"),
            max_fpr=0.20,
        ),
        
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    
    if hasattr(model, "decision_function"):
        
        validation_scores = model.decision_function(X_validation)
        test_scores = model.decision_function(X_test)
        
    else:
        validation_scores = model.predict_proba(X_validation)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]

    print(f"Split sizes: train={len(X_train)}, validation={len(X_validation)}, test={len(X_test)}")
    print("Three-fold CV: standardized partial ROC-AUC, max FPR=0.20")
    
    for params, mean, std in zip(
        search.cv_results_["params"],
        search.cv_results_["mean_test_score"],
        search.cv_results_["std_test_score"],
    ):
        print(f"  {mean:.4f} +/- {std:.4f}: {params}")
        
    print(f"Selected: {search.best_params_}")
    
    print(f"Test ROC-AUC: {roc_auc_score(y_test, test_scores):.4f}")
    
    print(
        "\nTarget | Threshold | Validation hadrons | "
        "Test hadron acceptance (95% CI) | Test gamma recall (95% CI)"
    )
    
    for target in TARGET_ACCEPTANCE:
        threshold = select_threshold(y_validation.to_numpy(), validation_scores, target)
        
        vh, vnh, _, _ = operating_point(
            y_validation.to_numpy(), validation_scores, threshold
        )
        
        h, nh, g, ng = operating_point(y_test.to_numpy(), test_scores, threshold)
        
        h_low, h_high = wilson_interval(h, nh)
        
        g_low, g_high = wilson_interval(g, ng)
        
        print(
            f"{target:6.0%} | {threshold:9.6g} | {vh}/{vnh} {vh/vnh:6.2%} | "
            f"{h}/{nh} {h/nh:6.2%} [{h_low:.2%}, {h_high:.2%}] | "
            f"{g}/{ng} {g/ng:6.2%} [{g_low:.2%}, {g_high:.2%}]"
        )


if __name__ == "__main__":
    main()
