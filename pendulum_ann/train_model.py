"""Train, validate, and evaluate a feedforward neural network."""

import argparse
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

if __package__:
    from .data_utils import DEFAULT_DATA_PATH, FEATURE_COLUMNS, load_dataset
else:
    from data_utils import DEFAULT_DATA_PATH, FEATURE_COLUMNS, load_dataset


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PLOT_PATH = PROJECT_DIR / "outputs" / "actual_vs_predicted.png"

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42
MAX_EPOCHS = 1000
PATIENCE = 50 # it will stop training if the validation loss does not improve for 50 consecutive epochs

# The code below defines a function to evaluate the performance of a regression model using various metrics. It calculates the Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE) between the actual and predicted values. The results are printed in a formatted table, and the function returns the calculated metrics as floats.
def evaluate(
    name: str,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float, float, float]:
    
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    
    
    print(
        f"{name:<12} | {mse:>12.6f} | {rmse:>12.6f} | "
        f"{mae:>12.6f} | {mape:>9.2%}"
    )
    
    return float(mse), float(rmse), float(mae), float(mape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    
    args = parser.parse_args()

    features, labels = load_dataset(args.data_path)
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )
    
    validation_fraction = VALIDATION_SIZE / (TEST_SIZE + VALIDATION_SIZE)
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=1 - validation_fraction,
        random_state=RANDOM_STATE,
    )

    input_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = input_scaler.fit_transform(X_train)
    
    
    X_validation_scaled = input_scaler.transform(X_validation)
    X_test_scaled = input_scaler.transform(X_test)

    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_train_scaled = target_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).ravel()
    
    
    y_validation_scaled = target_scaler.transform(
        y_validation.to_numpy().reshape(-1, 1)
    ).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.003,
        batch_size=256,
        max_iter=MAX_EPOCHS,
        random_state=RANDOM_STATE,
    )

    best_model: MLPRegressor | None = None
    best_validation_loss = float("inf")
    
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        
        model.partial_fit(X_train_scaled, y_train_scaled)
        
        validation_predictions = model.predict(X_validation_scaled)
        
        validation_loss = mean_squared_error(
            y_validation_scaled, validation_predictions
        )

        if validation_loss < best_validation_loss - 1e-8:
            best_validation_loss = validation_loss
            best_model = deepcopy(model)
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break


    if best_model is None:
        raise RuntimeError("Training did not produce a valid model.")
    

    validation_prediction = target_scaler.inverse_transform(
        best_model.predict(X_validation_scaled).reshape(-1, 1)
    ).ravel()
    
    
    test_prediction = target_scaler.inverse_transform(
        best_model.predict(X_test_scaled).reshape(-1, 1)
    ).ravel()
    
    
    y_validation_values = y_validation.to_numpy()
    y_test_values = y_test.to_numpy()


    print(f"Dataset rows: {len(features)}")
    print(
        f"Split sizes — train: {len(X_train)}, "
        f"validation: {len(X_validation)}, test: {len(X_test)}"
    )
    
    
    print(
        f"Network: {len(FEATURE_COLUMNS)} inputs -> 64 -> 32 -> 1 output; "
        f"best epoch: {best_epoch}"
    )
    
    print("\nSplit      |          MSE |         RMSE |          MAE |      MAPE")
    print("-" * 72)
    
    
    evaluate("Validation", y_validation_values, validation_prediction)
    evaluate("Test", y_test_values, test_prediction)

    args.plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    
    plt.scatter(y_test_values, test_prediction, alpha=0.55, s=18)
    lower = min(float(y_test_values.min()), float(test_prediction.min()))
    upper = max(float(y_test_values.max()), float(test_prediction.max()))
    
    
    plt.plot([lower, upper], [lower, upper], "r--", label="Ideal prediction")
    plt.xlabel("Actual period (s)")
    plt.ylabel("Predicted period (s)")
    
    
    plt.title("Pendulum period: actual vs. predicted (test set)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    
    
    plt.savefig(args.plot_path, dpi=150)
    plt.close()
    
    
    print(f"\nTest-set plot saved to: {args.plot_path}")


if __name__ == "__main__":
    main()
