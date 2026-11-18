"""Tests for Ionosphere baseline and logistic-regression evaluation."""

import unittest

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from ionosphere.model_comparison import (
    SCORING,
    build_models,
    evaluate_models,
)


class ModelComparisonTests(unittest.TestCase):
    def test_builds_prior_baseline_and_scaled_logistic_regression(self) -> None:
        models = build_models()
        self.assertEqual(list(models), ["Dummy baseline", "Logistic Regression"])
        self.assertIsInstance(models["Dummy baseline"], DummyClassifier)
        self.assertEqual(models["Dummy baseline"].strategy, "prior")

        logistic = models["Logistic Regression"]
        self.assertIsInstance(logistic, Pipeline)
        self.assertEqual(
            [name for name, _ in logistic.steps],
            ["standardscaler", "logisticregression"],
        )

    def test_evaluates_both_models_with_finite_cross_validation_metrics(self) -> None:
        results = evaluate_models()
        self.assertEqual(list(results.index), ["Dummy baseline", "Logistic Regression"])

        for metric in SCORING:
            for statistic in ("mean", "std"):
                values = results[f"{metric}_{statistic}"].to_numpy()
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue(((values >= 0) & (values <= 1)).all())


if __name__ == "__main__":
    unittest.main()
