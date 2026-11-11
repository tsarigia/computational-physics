"""Tune, evaluate, and plot a decision tree for the MAGIC dataset."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Conditional import to handle both script execution and package import
if __package__:
    from .data_utils import (
        DEFAULT_DATA_PATH,
        FEATURE_COLUMNS,
        PROJECT_DIR,
        RANDOM_STATE,
        evaluate_classifier,
        load_dataset,
        split_dataset,
    )
else:
    from data_utils import (
        DEFAULT_DATA_PATH,
        FEATURE_COLUMNS,
        PROJECT_DIR,
        RANDOM_STATE,
        evaluate_classifier,
        load_dataset,
        split_dataset,
    )

# Initialize the argument parser to read terminal commands
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Add an argument for the dataset path
    parser.add_argument(
        "--data-path",
        type=Path,# Automatically convert input string to a Path object
        default=DEFAULT_DATA_PATH,# Use default path if user doesn't provide one
        help="Path to magic04.data (default: repository root).",
    )
    # Add an argument for the plot save path
    parser.add_argument(
        "--plot-path",
        type=Path,# Automatically convert input string to a Path object
        default=PROJECT_DIR / "outputs" / "decision_tree.png",
        help="Where to save the decision-tree visualization.",
    )
    # Parse the arguments provided by the user in the terminal
    args = parser.parse_args() 

    # Load and preprocess the dataset using the chosen path
    features, labels = load_dataset(args.data_path)
    X_train, X_test, y_train, y_test = split_dataset(features, labels)
    
    # Set up GridSearchCV to find the best hyperparameters
    search = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid={
            "max_depth": [3, 5, 10, 15, None],
            "min_samples_leaf": [1, 5, 10, 20, 50],
        },
        cv=5,# 5-fold cross-validation
        scoring="roc_auc",# Metric used to select the best model
        n_jobs=-1,# Use all available CPU cores for faster training
    )
    # Train the models and find the best combination
    search.fit(X_train, y_train)

    print(f"Best parameters: {search.best_params_}")
    
    # Evaluate the best model on the test set using our custom function
    evaluate_classifier("Decision Tree", search.best_estimator_, X_test, y_test)
    
    # Create the 'outputs' directory 
    args.plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(20, 10))
    
    plot_tree(
        search.best_estimator_,
        feature_names=list(FEATURE_COLUMNS),
        class_names=["hadron", "gamma"],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title(f"Decision Tree (best parameters: {search.best_params_})")
    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=150)
    plt.show()
    plt.close()
    print(f"Decision tree plot saved to: {args.plot_path}")


if __name__ == "__main__":
    main()
