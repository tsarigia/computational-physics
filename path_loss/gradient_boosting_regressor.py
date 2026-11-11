"""Train and evaluate a gradient-boosting regressor for path loss."""

import argparse
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor

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

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    evaluate_regressor("Gradient Boosting", model, X_test, y_test)


if __name__ == "__main__":
    main()
