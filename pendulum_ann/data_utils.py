"""Dataset generation and loading for the pendulum ANN project."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "pendulum_dataset.csv"

FEATURE_COLUMNS = ("L_m", "theta0_rad", "g_m_s2")# our features are length, initial angle, and gravity
TARGET_COLUMN = "period_s"# the target is the period of the pendulum

DEFAULT_SAMPLES = 10_000 # the default number of samples to generate
DEFAULT_NOISE_STD = 0.02 # the default standard deviation of noise to add to the period
RANDOM_STATE = 42 # the random state for reproducibility


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
    
    # Initialize the random number generator for reproducibility
    rng = np.random.default_rng(random_state)
    
    # Generate random physical parameters within realistic ranges(Draw samples)
    length = rng.uniform(0.2, 2.0, n_samples)# Pendulum length between 0.2m and 2.0m
    angle_rad = np.deg2rad(rng.uniform(5.0, 80.0, n_samples))# Initial release angle (converted to radians)
    gravity = rng.uniform(9.6, 9.9, n_samples)# Gravitational acceleration between 9.6 m/s² and 9.9 m/s²

    # Calculate the theoretical period using the small-angle approximation (T = 2 * pi * sqrt(L/g))
    small_angle_period = 2 * np.pi * np.sqrt(length / gravity)
    
    # Apply the finite-angle correction factor (Taylor series expansion)
    angle_correction = 1 + angle_rad**2 / 16 + 11 * angle_rad**4 / 3072
    
    # Calculate the true theoretical period
    period = small_angle_period * angle_correction
    # Add normally distributed noise to simulate measurement errors
    period += rng.normal(loc=0.0, scale=noise_std, size=n_samples)

    # Combine all arrays into a structured Pandas DataFrame
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
    # Ensure data_path is a Path object
    data_path = Path(data_path)
    
    # Create the target directory (and any parent directories) if it doesn't exist
    data_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate the data
    data = generate_dataset(n_samples, noise_std, random_state)
    
    # Save the DataFrame to a CSV file without the index column
    data.to_csv(data_path, index=False)
    return data_path


def load_dataset(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate a generated pendulum dataset."""
    data_path = Path(data_path)
    
    # Check if the file actually exists before trying to read it
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Run generate_dataset.py first."
        )

    frame = pd.read_csv(data_path)
    
    # Define the exact column structure expected
    expected_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    
    # Structural validation, including checks for emptiness and correct column names
    if frame.empty:
        raise ValueError(f"Dataset is empty: {data_path}")
    if list(frame.columns) != expected_columns:# if the columns are not exactly as expected, raise an error
        raise ValueError(
            f"Expected columns {expected_columns} in {data_path}; "
            f"found {list(frame.columns)}"
        )

    # Force all data to be numeric, raising an error if text/invalid data is found
    frame = frame.apply(pd.to_numeric, errors="raise")
    
    # Ensures there are no NaN, Infinity, or missing values
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Dataset contains missing or non-finite values: {data_path}")

    features = frame.loc[:, list(FEATURE_COLUMNS)]
    
    # Split into features (X) and target/labels (y)
    labels = frame[TARGET_COLUMN]
    return features, labels
