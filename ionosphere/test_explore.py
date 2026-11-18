"""Integration tests for Ionosphere exploratory plots."""

import tempfile
import unittest
from pathlib import Path

from ionosphere.explore import create_eda_report


class IonosphereExplorationTests(unittest.TestCase):
    def test_creates_all_expected_plot_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "eda"
            paths = create_eda_report(output_dir=output_dir)

            self.assertEqual(len(paths), 4)
            for path in paths.values():
                self.assertTrue(path.is_file())
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
