"""Tests for Ionosphere baseline and logistic-regression evaluation."""

import unittest
from unittest.mock import patch

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ionosphere.model_comparison import (
    SCORING,
    build_models,
    build_parameter_grids,
    evaluate_models,
    evaluate_models_nested,
)


class ModelComparisonTests(unittest.TestCase):
    def test_builds_expected_models_and_scales_distance_based_models(self) -> None:
        models = build_models()
        self.assertEqual(
            list(models),
            [
                "Dummy baseline",
                "Logistic Regression",
                "Linear SVM",
                "RBF SVM",
                "K-Nearest Neighbors",
                "Random Forest",
                "Gradient Boosting",
                "XGBoost",
            ],
        )
        self.assertEqual(list(build_parameter_grids()), list(models))
        self.assertIsInstance(models["Dummy baseline"], DummyClassifier)
        self.assertEqual(models["Dummy baseline"].strategy, "prior")

        for model_name in (
            "Logistic Regression",
            "Linear SVM",
            "RBF SVM",
            "K-Nearest Neighbors",
        ):
            with self.subTest(model=model_name):
                self.assertIsInstance(models[model_name], Pipeline)
                self.assertEqual(
                    [name for name, _ in models[model_name].steps][0],
                    "standardscaler",
                )
        for model_name, kernel in (("Linear SVM", "linear"), ("RBF SVM", "rbf")):
            with self.subTest(model=model_name):
                svc = models[model_name].named_steps["svc"]
                self.assertIsInstance(svc, SVC)
                self.assertEqual(svc.kernel, kernel)

    def test_evaluates_all_models_with_finite_cross_validation_metrics(self) -> None:
        results = evaluate_models()
        self.assertEqual(list(results.index), list(build_models()))

        for metric in SCORING:
            for statistic in ("mean", "std"):
                values = results[f"{metric}_{statistic}"].to_numpy()
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue(((values >= 0) & (values <= 1)).all())

    def test_nested_cv_scores_outer_folds_and_returns_inner_parameters(self) -> None:
        dummy_models = {"Dummy baseline": DummyClassifier(strategy="prior")}
        with (
            patch(
                "ionosphere.model_comparison.build_models",
                return_value=dummy_models,
            ),
            patch(
                "ionosphere.model_comparison.build_parameter_grids",
                return_value={"Dummy baseline": {}},
            ),
        ):
            results, best_parameters = evaluate_models_nested(
                n_splits=2,
                inner_splits=2,
            )

        self.assertEqual(list(results.index), ["Dummy baseline"])
        self.assertEqual(best_parameters, {"Dummy baseline": [{}, {}]})
        for metric in SCORING:
            self.assertTrue(np.isfinite(results[f"{metric}_mean"]).all())
            self.assertTrue(np.isfinite(results[f"{metric}_std"]).all())


if __name__ == "__main__":
    unittest.main()
