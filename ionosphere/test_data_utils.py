"""Tests for Ionosphere dataset loading and validation."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ionosphere.data_utils import FEATURE_COLUMNS, load_dataset


class IonosphereLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.data_path = Path(self.temp_dir.name) / "ionosphere.data"

    def write_rows(self, rows: list[list[object]]) -> None:
        pd.DataFrame(rows).to_csv(self.data_path, header=False, index=False)

    def test_loads_features_and_encodes_both_classes(self) -> None:
        self.write_rows([[*range(34), "g"], [*[value + 1 for value in range(34)], "b"]])
        features, labels = load_dataset(self.data_path)
        self.assertEqual(list(features.columns), list(FEATURE_COLUMNS))
        self.assertEqual(features.shape, (2, 34))
        self.assertEqual(labels.tolist(), [1, 0])

    def test_rejects_wrong_column_count_and_unknown_labels(self) -> None:
        self.write_rows([[*range(33), "g"]])
        with self.assertRaisesRegex(ValueError, "Expected 35 columns"):
            load_dataset(self.data_path)

        self.write_rows([[*range(34), "unknown"], [*range(34), "g"]])
        with self.assertRaisesRegex(ValueError, "Unexpected class labels"):
            load_dataset(self.data_path)

    def test_rejects_nonfinite_features_and_single_class_data(self) -> None:
        self.write_rows([[float("inf"), *range(33), "g"], [*range(34), "b"]])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_dataset(self.data_path)

        self.write_rows([[*range(34), "g"], [*range(34), "g"]])
        with self.assertRaisesRegex(ValueError, "Both classes"):
            load_dataset(self.data_path)


if __name__ == "__main__":
    unittest.main()
