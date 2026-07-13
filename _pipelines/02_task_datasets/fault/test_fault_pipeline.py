#!/usr/bin/env python3
"""Focused unit tests for fault-stick parsing and voxel rasterization."""
from __future__ import annotations

import io
import unittest
from unittest.mock import Mock, patch

import joblib
import numpy as np

import build_dataset
import baseline
from audit_utils import verify_historical_artifacts_if_present
from ml_framework.model_registry import get_model
from ml_framework.preprocess import (
    denoise_identity,
    denormalize,
    fit_minmax,
    fit_zscore,
    normalize,
)


class FaultDatasetTests(unittest.TestCase):
    def test_fault_sticks_parse_complete_sequences(self) -> None:
        sticks = build_dataset.fault_sticks(np.asarray([1, 2, 3, 1, 3], dtype=np.int32))
        self.assertEqual([stick.tolist() for stick in sticks], [[0, 1, 2], [3, 4]])

    def test_fault_sticks_reject_malformed_sequence(self) -> None:
        with self.assertRaisesRegex(ValueError, "has no start"):
            build_dataset.fault_sticks(np.asarray([2, 3], dtype=np.int32))

    def test_rasterization_connects_vertices_without_dilation(self) -> None:
        faults = {
            "inline": np.asarray([10, 12], dtype=np.int32),
            "crossline": np.asarray([20, 22], dtype=np.int32),
            "twt_ms": np.asarray([4.0, 12.0]),
            "stick_no": np.asarray([1, 3], dtype=np.int32),
            "fault_name": np.asarray(["F1", "F1"], dtype=object),
        }
        index = {
            "il_min": np.asarray(10),
            "il_max": np.asarray(12),
            "xl_min": np.asarray(20),
            "xl_max": np.asarray(22),
            "samples_ms": np.asarray([0.0, 4.0, 8.0, 12.0, 16.0]),
        }
        voxels = build_dataset.rasterize_fault_voxels(faults, index)
        self.assertEqual(voxels.tolist(), [[10, 20, 1], [11, 21, 2], [12, 22, 3]])

    def test_label_patch_is_inline_specific(self) -> None:
        grouped = {10: np.asarray([[20, 5], [21, 6]], dtype=np.int32)}
        shape = build_dataset.PatchShape(3, 3)
        positive = build_dataset.label_patch(grouped, 10, 20, 5, shape)
        other_inline = build_dataset.label_patch(grouped, 11, 20, 5, shape)
        self.assertEqual(int(positive.sum()), 2)
        self.assertEqual(int(other_inline.sum()), 0)

    def test_shared_normalization_is_invertible_and_denoising_is_identity(self) -> None:
        values = np.asarray([[-3.5, -0.25, 1.5], [2.0, 4.25, 9.0]], dtype=np.float32)
        self.assertIs(denoise_identity(values), values)
        for fit in (fit_zscore, fit_minmax):
            stats = fit(values)
            restored = denormalize(normalize(values, stats), stats)
            np.testing.assert_allclose(restored, values, rtol=1e-6, atol=1e-6)

    def test_model_is_discovered_through_shared_registry(self) -> None:
        model = get_model("fault_local_logistic", models_package="models", seed=7)
        self.assertTrue(callable(model.train_batch))
        self.assertTrue(callable(model.loss_batch))
        self.assertTrue(callable(model.predict_batch))

    def test_model_alternatives_satisfy_batch_and_checkpoint_contracts(self) -> None:
        patches = np.linspace(-1.5, 1.5, num=24, dtype=np.float32).reshape(2, 1, 3, 4)
        labels = (patches[:, 0] > 0.0).astype(np.uint8)
        weights = np.where(labels == 1, 4.0, 1.0).astype(np.float32)
        validation_patches = patches * np.float32(0.8) + np.float32(0.05)

        for name in ("fault_raw_logistic", "fault_local_huber"):
            with self.subTest(model=name):
                model = get_model(name, models_package="models", seed=7)
                self.assertEqual(model.__class__.__module__, f"models.{name}")

                train_loss = model.train_batch(patches, labels, weights)
                validation_loss = model.loss_batch(validation_patches, labels, weights)
                probabilities = model.predict_batch(validation_patches)

                self.assertTrue(np.isfinite(train_loss))
                self.assertTrue(np.isfinite(validation_loss))
                self.assertEqual(probabilities.shape, labels.shape)
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue(np.logical_and(probabilities >= 0.0, probabilities <= 1.0).all())

                checkpoint = io.BytesIO()
                joblib.dump(model, checkpoint)
                checkpoint.seek(0)
                restored = joblib.load(checkpoint)
                np.testing.assert_allclose(
                    restored.predict_batch(validation_patches), probabilities, rtol=0.0, atol=0.0
                )

    def test_split_plan_has_explicit_disjoint_guard(self) -> None:
        plan = build_dataset.make_split_plan(100, 199, test_fraction=0.2, guard_inlines=5)
        self.assertEqual(plan.train, (100, 174))
        self.assertEqual(plan.guard, (175, 179))
        self.assertEqual(plan.test, (180, 199))
        train, guard, test = plan.inline_sets()
        self.assertFalse(train & guard)
        self.assertFalse(train & test)
        self.assertFalse(guard & test)

    def test_guard_is_mandatory(self) -> None:
        with self.assertRaisesRegex(ValueError, "guard-inlines"):
            build_dataset.make_split_plan(100, 199, test_fraction=0.2, guard_inlines=0)

    def test_normalization_is_fitted_only_on_train_and_reused(self) -> None:
        built = {
            "train": [],
            "test": [
                {
                    "seismic_patch": np.asarray([[[1000.0, 2000.0]]], dtype=np.float32),
                    "label": np.ones((1, 2), dtype=np.uint8),
                    "position": {"inline": 100},
                    "meta": {},
                }
            ],
        }
        for inline in range(10):
            built["train"].append(
                {
                    "seismic_patch": np.asarray([[[float(inline), float(inline + 2)]]], dtype=np.float32),
                    "label": np.ones((1, 2), dtype=np.uint8),
                    "position": {"inline": inline},
                    "meta": {},
                }
            )
        stats, plan = build_dataset.apply_training_normalization(
            built, val_fraction=0.2, val_guard_inlines=1
        )
        self.assertEqual(plan.val_start_inline, 8)
        self.assertAlmostEqual(stats.mean, 4.0)
        self.assertEqual(
            built["train"][0]["meta"]["normalization"],
            built["test"][0]["meta"]["normalization"],
        )
        self.assertEqual(built["test"][0]["meta"]["normalization_fit_split"], "train_fit")

    def test_physical_voxel_metrics_deduplicate_overlapping_patches(self) -> None:
        labels = np.zeros((2, 3, 3), dtype=np.uint8)
        labels[:, 1, 1] = 1
        samples = baseline.SampleSet(
            patches=np.zeros((2, 1, 3, 3), dtype=np.float32),
            labels=labels,
            positions=[
                {"inline": 10, "crossline": 20, "time_index": 30},
                {"inline": 10, "crossline": 20, "time_index": 30},
            ],
            roundtrip_max_abs_error=0.0,
        )
        probabilities = np.stack(
            [np.full((3, 3), 0.2, dtype=np.float32), np.full((3, 3), 0.8, dtype=np.float32)]
        )
        truth, averaged, audit = baseline.aggregate_physical_voxels(samples, probabilities)
        self.assertEqual(len(truth), 9)
        np.testing.assert_allclose(averaged, 0.5)
        self.assertEqual(audit["repeated_physical_voxels"], 9)
        self.assertEqual(audit["max_coverage_multiplicity"], 2)

    def test_threshold_is_selected_from_validation_probabilities(self) -> None:
        truth = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        probabilities = np.asarray([0.1, 0.4, 0.6, 0.9])
        threshold, source, f1 = baseline.select_validation_threshold(truth, probabilities, "auto")
        self.assertEqual(source, "validation_max_f1")
        self.assertAlmostEqual(threshold, 0.6)
        self.assertAlmostEqual(f1, 1.0)

    def test_missing_optional_historical_bundle_does_not_block_portable_code(self) -> None:
        missing = Mock()
        missing.is_file.return_value = False
        with patch("audit_utils.historical_artifact_paths", return_value={"old.ckpt": missing}):
            self.assertEqual(verify_historical_artifacts_if_present(), {})

    def test_partial_historical_bundle_fails_loudly(self) -> None:
        present = Mock()
        present.is_file.return_value = True
        missing = Mock()
        missing.is_file.return_value = False
        with patch(
            "audit_utils.historical_artifact_paths",
            return_value={"present.ckpt": present, "missing.ckpt": missing},
        ):
            with self.assertRaisesRegex(FileNotFoundError, "partial historical baseline bundle"):
                verify_historical_artifacts_if_present()

    def test_batch_sources_can_produce_two_complete_epochs(self) -> None:
        samples = baseline.SampleSet(
            patches=np.zeros((4, 1, 3, 3), dtype=np.float32),
            labels=np.zeros((4, 3, 3), dtype=np.uint8),
            positions=[{"inline": i} for i in range(4)],
            roundtrip_max_abs_error=0.0,
        )
        weights = np.ones_like(samples.labels, dtype=np.float32)
        source = baseline.ShuffledBatches(samples, weights, batch_size=2, seed=7)
        first_epoch = list(iter(source))
        second_epoch = list(iter(source))
        self.assertEqual([len(batch[0]) for batch in first_epoch], [2, 2])
        self.assertEqual([len(batch[0]) for batch in second_epoch], [2, 2])


if __name__ == "__main__":
    unittest.main()
