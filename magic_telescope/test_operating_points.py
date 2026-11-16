"""Focused tests for low-background operating-point evaluation."""

import unittest

import numpy as np

from magic_telescope.operating_points import (
    operating_point,
    select_threshold,
    wilson_interval,
)


class OperatingPointTests(unittest.TestCase):
    
    def test_threshold_respects_tied_hadron_scores(self) -> None:
        
        # Two hadrons share a score, so accepting both would exceed the 25% limit.
        labels = np.array([0, 0, 0, 0, 1, 1, 1])
        scores = np.array([0.9, 0.8, 0.8, 0.1, 0.95, 0.85, 0.7])
        
        threshold = select_threshold(labels, scores, 0.25)
        
        # The threshold keeps one hadron and both higher-scoring gamma events.
        self.assertEqual(threshold, 0.85)
        self.assertEqual(operating_point(labels, scores, threshold), (1, 4, 2, 3))

    def test_threshold_can_accept_no_events(self) -> None:
        # With the highest score assigned to a hadron, zero acceptance rejects all.
        labels = np.array([0, 1])
        
        scores = np.array([0.9, 0.1])
        
        threshold = select_threshold(labels, scores, 0.0)
        
        self.assertEqual(operating_point(labels, scores, threshold), (0, 1, 0, 1))

    def test_invalid_inputs_are_rejected(self) -> None:
        
        # ROC threshold selection needs both hadron and gamma events.
        with self.assertRaises(ValueError):
            select_threshold(np.array([0, 0]), np.array([0.2, 0.3]), 0.1)
            
        # Non-finite scores cannot define reliable ROC thresholds.
        with self.assertRaises(ValueError):
            select_threshold(np.array([0, 1]), np.array([0.2, np.nan]), 0.1)
            
        # Acceptance is a proportion and cannot exceed one.
        with self.assertRaises(ValueError):
            select_threshold(np.array([0, 1]), np.array([0.2, 0.3]), 1.1)

    def test_wilson_interval_at_boundaries(self) -> None:
        
        # Check that zero and complete acceptance have symmetric Wilson bounds.
        low_zero, high_zero = wilson_interval(0, 100)
        low_all, high_all = wilson_interval(100, 100)
        
        
        self.assertAlmostEqual(low_zero, 0.0)
        
        self.assertAlmostEqual(high_zero, 1 - low_all)
        
        self.assertAlmostEqual(high_all, 1.0)
        
        self.assertGreater(high_zero, 0.0)
        
        # An interval cannot be computed without any observations.
        with self.assertRaises(ValueError):
            wilson_interval(1, 0)


if __name__ == "__main__":
    unittest.main()
