"""Dataset generation and loading for the pendulum ANN project."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "pendulum_dataset.csv"
FEATURE_COLUMNS = ("L_m", "theta0_rad", "g_m_s2")
TARGET_COLUMN = "period_s"
DEFAULT_SAMPLES = 10_000
DEFAULT_NOISE_STD = 0.02
RANDOM_STATE = 42


def generate_dataset(
    n_samples: int = DEFAULT_SAMPLES,
    noise_std: float = DEFAULT_NOISE_STD,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate pendulum periods using the finite-angle approximation."""
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than zero.")
    if noise_std < 0:
        raise ValueError("noise_std must not be negative.")

    rng = np.random.default_rng(random_state)
    length = rng.uniform(0.2, 2.0, n_samples)
    angle_rad = np.deg2rad(rng.uniform(5.0, 80.0, n_samples))
    gravity = rng.uniform(9.6, 9.9, n_samples)

    small_angle_period = 2 * np.pi * np.sqrt(length / gravity)
    angle_correction = 1 + angle_rad**2 / 16 + 11 * angle_rad**4 / 3072
    period = small_angle_period * angle_correction
    period += rng.normal(loc=0.0, scale=noise_std, size=n_samples)

    return pd.DataFrame(
        {
            "L_m": length,
            "theta0_rad": angle_rad,
            "g_m_s2": gravity,
            "period_s": period,
        }
    )


def save_dataset(
    data_path: str | Path = DEFAULT_DATA_PATH,
    n_samples: int = DEFAULT_SAMPLES,
    noise_std: float = DEFAULT_NOISE_STD,
    random_state: int = RANDOM_STATE,
) -> Path:
    """Generate the dataset and save it as a CSV file."""
    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_dataset(n_samples, noise_std, random_state)
    data.to_csv(data_path, index=False)
    return data_path


def load_dataset(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate a generated pendulum dataset."""
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Run generate_dataset.py first."
        )

    frame = pd.read_csv(data_path)
    expected_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    if frame.empty:
        raise ValueError(f"Dataset is empty: {data_path}")
    if list(frame.columns) != expected_columns:
        raise ValueError(
            f"Expected columns {expected_columns} in {data_path}; "
            f"found {list(frame.columns)}"
        )

    frame = frame.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Dataset contains missing or non-finite values: {data_path}")

    features = frame.loc[:, list(FEATURE_COLUMNS)]
    labels = frame[TARGET_COLUMN]
    return features, labels
