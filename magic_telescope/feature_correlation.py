"""Plot Pearson correlations between MAGIC features in the training data."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

if __package__:
    from .data_utils import (
        DEFAULT_DATA_PATH,
        PROJECT_DIR,
        RANDOM_STATE,
        load_dataset,
        split_dataset,
    )
else:
    from data_utils import (
        DEFAULT_DATA_PATH,
        PROJECT_DIR,
        RANDOM_STATE,
        load_dataset,
        split_dataset,
    )


def main() -> None:
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=PROJECT_DIR / "outputs" / "feature_correlation.png",
    )
    
    args = parser.parse_args()

    features, labels = load_dataset(args.data_path)
    X_development, _, y_development, _ = split_dataset(features, labels)
    
    
    # Match the operating-point study's training subset, excluding validation and test.
    X_train, _, _, _ = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_development,
    )
    
    # we compute the Pearson correlation matrix for the training features
    correlation = X_train.corr(method="pearson")
    values = correlation.to_numpy()
    
    if not np.isfinite(values).all():
        raise ValueError("Pearson correlation is undefined for a constant feature.")

    fig, ax = plt.subplots(figsize=(11, 9))
    
    
    image = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1)
    
    ax.set_xticks(
        range(len(correlation.columns)),
        labels=correlation.columns,
        rotation=45,
        ha="right",
    )
    
    ax.set_yticks(range(len(correlation.index)), labels=correlation.index)
    ax.set_title("MAGIC feature correlations (training data)")
    
    
    for row in range(len(values)):
        for col in range(len(values)):
            
            color = "white" if abs(values[row, col]) > 0.55 else "black"
            ax.text(
                col, row, f"{values[row, col]:.2f}",
                ha="center", va="center", color=color, fontsize=8,
            )
            
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    
    args.plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_path, dpi=150)
    
    plt.close(fig)
    
    print(f"Training rows: {len(X_train)}")
    print(f"Correlation plot saved to: {args.plot_path}")


if __name__ == "__main__":
    main()
