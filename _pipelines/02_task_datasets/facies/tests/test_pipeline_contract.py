from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

import train_baseline
from build_dataset import fit_training_normalization, preprocess_patch
from pipeline_contract import (
    TASK_SCHEMAS,
    get_task_schema,
    is_near_constant_patch,
    ordered_spatial_split,
    segmentation_metrics_from_confusion,
    validate_label_array,
)


class PipelineContractTests(unittest.TestCase):
    def test_fixed_task_label_schemas(self) -> None:
        self.assertEqual(TASK_SCHEMAS["facies_f3"].valid_label_ids, tuple(range(10)))
        self.assertEqual(
            TASK_SCHEMAS["facies_penobscot"].valid_label_ids, tuple(range(8))
        )
        self.assertIsNone(TASK_SCHEMAS["facies_f3"].ignore_index)
        self.assertIsNone(TASK_SCHEMAS["facies_penobscot"].ignore_index)
        self.assertFalse(hasattr(train_baseline, "infer_num_classes"))

    def test_label_schema_rejects_out_of_range_id(self) -> None:
        validate_label_array(np.array([[0, 9]], dtype=np.uint8), get_task_schema("facies_f3"))
        with self.assertRaises(ValueError):
            validate_label_array(
                np.array([[0, 10]], dtype=np.uint8), get_task_schema("facies_f3")
            )

    def test_external_and_internal_spatial_guards(self) -> None:
        train, guard, test = ordered_spatial_split(range(100, 751), 0.20, 0.05)
        self.assertEqual((min(train), max(train), len(train)), (100, 586, 487))
        self.assertEqual((min(guard), max(guard), len(guard)), (587, 619, 33))
        self.assertEqual((min(test), max(test), len(test)), (620, 750, 131))

        model_train, val_guard, val = ordered_spatial_split(train, 0.20, 0.05)
        self.assertEqual((min(model_train), max(model_train)), (100, 463))
        self.assertEqual((min(val_guard), max(val_guard)), (464, 488))
        self.assertEqual((min(val), max(val)), (489, 586))
        self.assertFalse(model_train & val_guard)
        self.assertFalse(model_train & val)
        self.assertFalse(val_guard & val)

    def test_train_only_normalization_round_trip(self) -> None:
        fit_samples = [
            {"seismic_patch": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)},
            {"seismic_patch": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)},
        ]
        stats = fit_training_normalization(fit_samples, "zscore")
        held_out = np.array([[100.0, 120.0], [140.0, 160.0]], dtype=np.float32)
        before = stats.to_dict().copy()
        normalized, error = preprocess_patch(held_out, stats)
        self.assertEqual(stats.to_dict(), before)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertLessEqual(error, 2e-5)

    def test_near_constant_rule_is_amplitude_only(self) -> None:
        self.assertTrue(is_near_constant_patch(np.ones((4, 4), dtype=np.float32)))
        self.assertTrue(
            is_near_constant_patch(
                np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            )
        )
        self.assertFalse(
            is_near_constant_patch(np.arange(16, dtype=np.float32).reshape(4, 4))
        )

    def test_metrics_include_finite_per_class_support(self) -> None:
        confusion = np.array([[8, 2], [1, 9]], dtype=np.int64)
        metrics = segmentation_metrics_from_confusion(confusion)
        self.assertEqual(metrics["per_class_support"], [10, 10])
        self.assertEqual(metrics["evaluated_pixels"], 20)
        self.assertEqual(metrics["ignored_pixels"], 0)
        for key in ("accuracy", "miou", "macro_f1"):
            self.assertTrue(np.isfinite(metrics[key]))
        self.assertTrue(np.isfinite(metrics["per_class_iou"]).all())
        self.assertTrue(np.isfinite(metrics["per_class_f1"]).all())

    def test_empty_batch_source_fails_loudly(self) -> None:
        train_baseline.assert_nonempty_loader([object()], "train")
        with self.assertRaises(RuntimeError):
            train_baseline.assert_nonempty_loader([], "train")


if __name__ == "__main__":
    unittest.main()
