"""Generate exploratory summaries and plots for the Ionosphere dataset."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from .data_utils import (
        CLASS_COLUMN,
        DEFAULT_DATA_PATH,
        FEATURE_COLUMNS,
        PROJECT_DIR,
        load_dataset,
    )
else:
    from data_utils import (
        CLASS_COLUMN,  # noqa: F401
        DEFAULT_DATA_PATH,
        FEATURE_COLUMNS,  # noqa: F401
        PROJECT_DIR,
        load_dataset,
    )

DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
CLASS_NAMES = {0: "Bad (b)", 1: "Good (g)"}


def create_eda_report(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    
    """Summarize the dataset and save class, distribution, and relationship plots."""
    
    
    features, labels = load_dataset(data_path)
    
    output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    class_counts = labels.map(CLASS_NAMES).value_counts().reindex(
        ["Bad (b)", "Good (g)"], fill_value=0
    )
    
    constant_features = [
        name for name in features if features[name].nunique(dropna=False) == 1
    ]
    
    print(f"Rows: {len(features)}")
    print(f"Features: {features.shape[1]}")
    
    
    print(
        f"Classes: {class_counts.to_dict()} "
        f"({(class_counts / len(labels) * 100).round(1).to_dict()}%)"
    )
    
    
    print(f"Missing values: {int(features.isna().sum().sum() + labels.isna().sum())}")
    print(f"Constant features: {constant_features or 'none'}")
    
    print("No scaling or feature removal is applied during EDA.")

    paths = {
        "class_balance": output_dir / "class_balance.png",
        "feature_distributions": output_dir / "feature_distributions.png",
        "feature_correlations": output_dir / "feature_correlations.png",
        "feature_class_association": output_dir / "feature_class_association.png",
    }


    _plot_class_balance(class_counts, paths["class_balance"])
    _plot_feature_distributions(features, labels, paths["feature_distributions"])
    
    
    correlation = _plot_feature_correlations(
        features, paths["feature_correlations"]
    )
    association = _plot_feature_class_association(
        features, labels, paths["feature_class_association"]
    )

    if not correlation.empty:
        pairs = (
            correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
            .stack()
            .abs()
            .sort_values(ascending=False)
        )
        
        print("Strongest absolute feature correlations:")
        
        for (left, right) in pairs.head(5).keys():  # noqa: SIM118
            
            signed_value = correlation.loc[left, right]
            print(f"  {left} vs {right}: {signed_value:+.3f}")

    print("Strongest absolute point-biserial feature/class associations:")
    for name, value in association.abs().sort_values(ascending=False).head(5).items():  # noqa: PERF102
        print(f"  {name}: {association[name]:+.3f}")
        
    for name, path in paths.items():
        print(f"{name.replace('_', ' ').title()} plot: {path}")
    return paths


def _plot_class_balance(counts: pd.Series, output_path: Path) -> None:
    
    percentages = counts / counts.sum() * 100
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=["#4C78A8", "#F58518"])
    
    
    ax.set_title("Ionosphere radar-return classes")
    ax.set_ylabel("Number of observations")
    ax.set_ylim(0, max(counts.values) * 1.18)
    
    
    for bar, count, percentage in zip(bars, counts.values, percentages.values):
        
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count} ({percentage:.1f}%)",
            ha="center",
            va="bottom",
        )
        
        
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_feature_distributions(
    features: pd.DataFrame, labels: pd.Series, output_path: Path
) -> None:
    
    columns = 5
    rows = int(np.ceil(len(features.columns) / columns))
    
    
    fig, axes = plt.subplots(rows, columns, figsize=(16, rows * 2.5), squeeze=False)
    colors = {0: "#4C78A8", 1: "#F58518"}
    
    
    for index, name in enumerate(features.columns):
        
        ax = axes.flat[index]
        values = features[name]
        
        if values.nunique() == 1:
            
            ax.axvline(values.iloc[0], color="#555555", linewidth=2)
            ax.set_title(f"{name}\nconstant = {values.iloc[0]:g}", fontsize=9)
            
        else:
            
            for class_value, class_name in CLASS_NAMES.items():
                ax.hist(
                    values[labels == class_value],
                    bins=15,
                    alpha=0.55,
                    color=colors[class_value],
                    label=class_name,
                )
                
            ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        
    for ax in axes.flat[len(features.columns):]:
        ax.set_visible(False)
        
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    
    fig.legend(handles, legend_labels, loc="upper center", ncol=2)
    
    fig.suptitle("Feature distributions by radar-return class", y=1.01)
    
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    
    plt.close(fig)


def _plot_feature_correlations(
    features: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    
    
    varying_features = features.loc[:, features.nunique(dropna=False) > 1]
    correlation = varying_features.corr(method="pearson")
    
    if not np.isfinite(correlation.to_numpy()).all():
        raise ValueError("Feature correlation is undefined for the supplied data.")

    size = max(10, len(correlation.columns) * 0.34)
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    
    image = ax.imshow(correlation.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    
    ax.set_xticks(
        range(len(correlation.columns)),
        labels=correlation.columns,
        rotation=90,
        fontsize=7,
    )
    
    ax.set_yticks(
        range(len(correlation.index)),
        labels=correlation.index,
        fontsize=7,
    )
    
    ax.set_title("Pearson correlations between non-constant features")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    
    return correlation


def _plot_feature_class_association(
    features: pd.DataFrame, labels: pd.Series, output_path: Path
) -> pd.Series:
    
    varying_features = features.loc[:, features.nunique(dropna=False) > 1]
    
    association = varying_features.corrwith(labels, method="pearson").dropna()
    
    association = association.sort_values()
    
    colors = plt.get_cmap("RdBu_r")((association.to_numpy() + 1) / 2)
    
    fig, ax = plt.subplots(figsize=(9, 10))
    
    ax.barh(association.index, association.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1, 1)
    
    ax.set_xlabel("Point-biserial correlation (g = 1, b = 0)")
    ax.set_title("Univariate feature association with class")
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    
    return association


def main() -> None:
    
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    
    args = parser.parse_args()
    
    create_eda_report(args.data_path, args.output_dir)


if __name__ == "__main__":
    main()
