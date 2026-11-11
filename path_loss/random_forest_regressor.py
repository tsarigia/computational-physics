"""Train and evaluate a random-forest regressor for path loss."""

import argparse
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

if __package__:
    from .data_utils import (
        DEFAULT_DATA_PATH,
        RANDOM_STATE,
        evaluate_regressor,
        load_dataset,
        split_dataset,
    )
else:
    from data_utils import (
        DEFAULT_DATA_PATH,
        RANDOM_STATE,
        evaluate_regressor,
        load_dataset,
        split_dataset,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to paper.xlsx (default: path_loss directory).",
    )
    args = parser.parse_args()

    features, labels = load_dataset(args.data_path)
    X_train, X_test, y_train, y_test = split_dataset(features, labels)

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    evaluate_regressor("Random Forest", model, X_test, y_test)


if __name__ == "__main__":
    main()
