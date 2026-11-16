"""Tests for MAGIC dataset loading and splitting."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from magic_telescope.data_utils import FEATURE_COLUMNS, load_dataset, split_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        
        self.temp_dir = tempfile.TemporaryDirectory()
        
        self.addCleanup(self.temp_dir.cleanup)
        
        self.data_path = Path(self.temp_dir.name) / "data.csv"

    def write_rows(self, rows: list[list[object]]) -> None:
        
        pd.DataFrame(rows).to_csv(self.data_path, header=False, index=False)

    def test_load_encodes_labels_and_keeps_features(self) -> None:
        
        self.write_rows([[*range(10), "g"], [*range(10, 20), "h"]])
        features, labels = load_dataset(self.data_path)
        
        self.assertEqual(list(features.columns), list(FEATURE_COLUMNS))
        
        self.assertEqual(features.iloc[0].tolist(), list(range(10)))
        
        self.assertEqual(labels.tolist(), [1, 0])

    def test_load_rejects_unknown_labels_and_nonfinite_values(self) -> None:
        
        self.write_rows([[*range(10), "x"]])
        
        with self.assertRaisesRegex(ValueError, "Unexpected class labels"):
            load_dataset(self.data_path)

        self.write_rows([[float("inf"), *range(1, 10), "g"]])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            
            load_dataset(self.data_path)

    def test_split_is_reproducible_stratified_and_disjoint(self) -> None:
        
        self.write_rows([[*[i + j for j in range(10)], "g" if i < 10 else "h"]
                         for i in range(20)])
        
        features, labels = load_dataset(self.data_path)
        
        train, test, y_train, y_test = split_dataset(features, labels)
        
        train_again, test_again, _, _ = split_dataset(features, labels)

        self.assertEqual(train.index.tolist(), train_again.index.tolist())
        
        self.assertEqual(test.index.tolist(), test_again.index.tolist())
        
        self.assertTrue(set(train.index).isdisjoint(test.index))
        
        self.assertEqual(y_train.value_counts().to_dict(), {0: 7, 1: 7})
        
        self.assertEqual(y_test.value_counts().to_dict(), {0: 3, 1: 3})


if __name__ == "__main__":
    unittest.main()
