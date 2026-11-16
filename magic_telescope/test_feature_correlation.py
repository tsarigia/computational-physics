"""Integration tests for the feature-correlation plot."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class FeatureCorrelationTests(unittest.TestCase):
    def test_generates_png_at_requested_path(self) -> None:
        
        with tempfile.TemporaryDirectory() as temp_dir:
            
            output = Path(temp_dir) / "plots" / "correlation.png"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "magic_telescope.feature_correlation",
                    "--plot-path",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            
            self.assertIn("Training rows: 9985", result.stdout)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_rejects_constant_features(self) -> None:
        
        with tempfile.TemporaryDirectory() as temp_dir:
            
            data_path = Path(temp_dir) / "constant.csv"
            output = Path(temp_dir) / "plot.png"
            
            pd.DataFrame([[*[1] * 10, "g" if i < 20 else "h"]
                          for i in range(40)]).to_csv(
                data_path, header=False, index=False
            )
                          
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "magic_telescope.feature_correlation",
                    "--data-path",
                    str(data_path),
                    "--plot-path",
                    str(output),
                ],
                
                capture_output=True,
                text=True,
            )
            
            self.assertNotEqual(result.returncode, 0)
            
            self.assertIn("Pearson correlation is undefined", result.stderr)
            
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
