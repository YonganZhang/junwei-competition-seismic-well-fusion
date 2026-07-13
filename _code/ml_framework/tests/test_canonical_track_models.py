from __future__ import annotations

import unittest

import numpy as np

from _code.ml_framework.contracts import TaskSpec
from _code.ml_framework.model_discovery import discover_model


def spec(track: str, task_type: str, target: str, **metadata: object) -> TaskSpec:
    metrics = ("mae",) if task_type in {"regression", "reconstruction"} else ("score",)
    return TaskSpec(
        track_id=track, task_id=f"{track}.smoke", task_type=task_type,
        input_modalities=("smoke",), targets=(target,), units={target: "unit"},
        label_version="smoke-v1", target_masks={target: "finite"}, group_keys=("group",),
        target_transform={target: "identity"}, inverse_transform={target: "identity"},
        train_loss={target: "smoke"}, inference_transform={target: "identity"},
        threshold_policy={}, calibration_policy={}, primary_metrics=metrics,
        metric_directions={metric: "minimize" if metric == "mae" else "maximize" for metric in metrics},
        visualizer_id=f"{track}_smoke", required_figures=("smoke.png",), metadata=metadata,
    )


class CanonicalTrackModelTests(unittest.TestCase):
    def test_fault_raw_logits_and_probabilities(self):
        task = spec("fault", "binary", "fault")
        model = discover_model("fault", "fault_local_logistic").build(task)
        x = np.random.default_rng(2693).normal(size=(2, 1, 6, 6)).astype(np.float32)
        y = np.zeros((2, 6, 6), dtype=np.uint8); y[:, 2:4, 2:4] = 1
        weight = np.ones_like(y, dtype=np.float32)
        self.assertTrue(np.isfinite(model.train_batch(x, y, weight)))
        output = model.predict_output(x)
        self.assertEqual(output.raw["fault"].shape, (2, 6, 6))
        self.assertTrue(np.all((output.transformed["fault"] >= 0) & (output.transformed["fault"] <= 1)))

    def test_facies_logits(self):
        task = spec("facies", "multiclass", "facies", num_classes=4)
        model = discover_model("facies", "facies_pixel_logistic").build(task)
        x = np.random.default_rng(2693).normal(size=(2, 1, 8, 9))
        y = np.tile(np.arange(4), 36).reshape(2, 8, 9)
        model.train_batch(x, y)
        self.assertEqual(model.predict_output(x).raw["facies"].shape, (2, 4, 8, 9))

    def test_property_mask_capable_ridge(self):
        task = spec("property", "regression", "PHIF")
        model = discover_model("property", "reservoir_ridge").build(task, n_features=3)
        x = np.arange(30, dtype=float).reshape(10, 3) / 10.0
        y = (x[:, :1] * 0.2) + 0.1
        before = model.validation_loss((x, y))
        for _ in range(5):
            model.train_batch((x, y))
        self.assertTrue(np.isfinite(before))
        self.assertEqual(len(model.predict(x).raw["PHIF"]), 10)

    def test_lithofacies_fixed_nine_class_logits(self):
        task = spec("lithofacies", "multiclass", "GM09", num_classes=9)
        model = discover_model("lithofacies", "lithofacies_concat_logistic").build(task)
        logs = np.random.default_rng(2693).normal(size=(18, 4, 5))
        seismic = np.random.default_rng(2694).normal(size=(18, 1, 3, 3))
        labels = np.tile(np.arange(9), 2)
        model.train_batch(logs, seismic, labels)
        self.assertEqual(model.predict_output(logs, seismic).raw["GM09"].shape, (18, 9))

    def test_reconstruction_output_envelope(self):
        task = spec("reconstruction", "reconstruction", "property")
        model = discover_model("reconstruction", "reconstruction_linear_sgd").build(task, n_features=4)
        x = np.random.default_rng(2693).normal(size=(12, 4)); y = x[:, 0] - 0.2 * x[:, 1]
        model.train_batch((x, y))
        self.assertEqual(len(model.predict(x).raw["property"]), 12)


if __name__ == "__main__":
    unittest.main()
