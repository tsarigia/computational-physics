"""Generate a reproducible synthetic pendulum-period regression dataset."""

import argparse
from pathlib import Path

if __package__:
    from .data_utils import (
        DEFAULT_DATA_PATH,
        DEFAULT_NOISE_STD,
        DEFAULT_SAMPLES,
        RANDOM_STATE,
        save_dataset,
    )
else:
    from data_utils import (
        DEFAULT_DATA_PATH,
        DEFAULT_NOISE_STD,
        DEFAULT_SAMPLES,
        RANDOM_STATE,
        save_dataset,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    output_path = save_dataset(
        data_path=args.output,
        n_samples=args.samples,
        noise_std=args.noise_std,
        random_state=args.seed,
    )
    print(f"Generated {args.samples} samples: {output_path}")
    print(f"Gaussian noise standard deviation: {args.noise_std:g} s")


if __name__ == "__main__":
    main()
