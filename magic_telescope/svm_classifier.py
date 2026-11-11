"""Compare linear and RBF support-vector classifiers for the MAGIC dataset."""

import argparse
from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

C_VALUES = (0.1, 1.0, 10.0)
KERNELS = ("linear", "rbf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to magic04.data (default: repository root).",
    )
    args = parser.parse_args()# Parse the command-line arguments provided by the user

    features, labels = load_dataset(args.data_path)
    X_train, X_test, y_train, y_test = split_dataset(features, labels)

    for kernel in KERNELS:
        for C in C_VALUES:
            model = make_pipeline(
                StandardScaler(),
                SVC(kernel=kernel, C=C, random_state=RANDOM_STATE),
            )
            model.fit(X_train, y_train)
            evaluate_classifier(f"SVM ({kernel} kernel, C={C})", model, X_test, y_test)


if __name__ == "__main__":
    main()
