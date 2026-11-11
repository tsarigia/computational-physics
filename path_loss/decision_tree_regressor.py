"""Compare decision-tree regressors at different maximum depths."""

import argparse
from pathlib import Path

from sklearn.tree import DecisionTreeRegressor

if __package__:
    from .data_utils import (
        DEFAULT_DATA_PATH,
        RANDOM_STATE,
        load_dataset,
        regression_metrics,
        split_dataset,
    )
else:
    from data_utils import (
        DEFAULT_DATA_PATH,
        RANDOM_STATE,
        load_dataset,
        regression_metrics,
        split_dataset,
    )

MAX_DEPTH_VALUES: tuple[int | None, ...] = (2, 5, 10, 15, 20, 25, None)


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

    print(f"{'Max depth':<12} | {'MAE (dB)':>10} | {'RMSE (dB)':>10} | {'MAPE':>10}")
    print("-" * 53)
    for depth in MAX_DEPTH_VALUES:
        model = DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae, rmse, mape = regression_metrics(y_test, predictions)
        depth_label = str(depth) if depth is not None else "None"
        print(f"{depth_label:<12} | {mae:>10.4f} | {rmse:>10.4f} | {mape:>9.2%}")


if __name__ == "__main__":
    main()
