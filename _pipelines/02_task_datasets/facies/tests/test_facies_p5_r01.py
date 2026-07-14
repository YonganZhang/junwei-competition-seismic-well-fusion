from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


TRACK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_ROOT.parents[2]
for path in (str(PROJECT_ROOT), str(TRACK_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import facies_p5_r01 as r01


def _section(task: str, inline: int, shape: tuple[int, int] = (8, 8)) -> r01.FullSection:
    seismic = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - np.prod(shape) / 2
    label = (seismic >= 0).astype(np.uint8)
    return r01.FullSection(task, inline, seismic, label)


class _SignedBinaryModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat((-inputs, inputs), dim=1)


class FaciesP51R01ContractTests(unittest.TestCase):
    def test_track_prefixed_entry_and_no_test_surface(self) -> None:
        self.assertEqual(Path(r01.__file__).name, "facies_p5_r01.py")
        parameters = inspect.signature(r01.run_r01).parameters
        for forbidden in ("test", "test_root", "processed_root", "frozen_test", "holdout"):
            self.assertNotIn(forbidden, parameters)

    def test_independent_task_specs_and_frozen_internal_guards(self) -> None:
        f3 = r01.task_r01_spec("facies_f3")
        pen = r01.task_r01_spec("facies_penobscot")
        self.assertEqual((f3.num_classes, pen.num_classes), (10, 8))
        self.assertNotEqual(f3.label_version, pen.label_version)
        self.assertEqual(f3.legal_guard_range, (464, 488))
        self.assertEqual(pen.legal_guard_range, (1336, 1358))
        config = r01.R01Config(train_sections=2, validation_sections=2)
        for spec in (f3, pen):
            train, validation = r01.bounded_section_ids(spec, config)
            nearest = min(abs(left - right) for left in train for right in validation)
            self.assertGreater(nearest, spec.buffer_groups)

    def test_reader_rejects_outer_inline_before_any_payload_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "f3demo").mkdir()
            (root / "f3demo" / "inlines.zip").touch()
            (root / "f3demo" / "masks.tar.gz").touch()
            reader = r01.DevelopmentOnlyFullSectionReader(root, "facies_f3")
            with self.assertRaisesRegex(PermissionError, "outside development"):
                reader.read_section(620)
            evidence = reader.firewall_evidence()
            self.assertEqual(evidence["requested_payload_count"], 0)
            self.assertEqual(evidence["outer_payloads_read"], 0)
            self.assertFalse(evidence["test_archive_opened"])
            self.assertFalse(evidence["test_labels_read"])

    def test_sliding_windows_cover_edges_and_count_duplicates_once(self) -> None:
        section = _section("facies_f3", 489, (9, 11))
        config = r01.R01Config(window_size=4, stride=3, updates=1)
        windows = r01.plan_windows(section, config)
        self.assertIn((5, 7), {(ref.row, ref.col) for ref in windows})
        evidence = r01.coverage_evidence(section, windows)
        self.assertEqual(evidence["coverage_fraction"], 1.0)
        self.assertEqual(evidence["valid_voxels"], 99)
        self.assertEqual(evidence["covered_unique_voxels"], 99)
        self.assertGreater(evidence["duplicate_prediction_assignments"], 0)

    def test_legal_overlap_is_zero_and_random_overlap_is_nonzero(self) -> None:
        config = r01.R01Config(window_size=4, stride=2, updates=1)
        legal_section = _section("facies_f3", 463)
        validation_section = _section("facies_f3", 489)
        legal = r01.plan_windows(legal_section, config)
        evaluation = r01.plan_windows(validation_section, config)
        shapes = {463: (8, 8), 489: (8, 8)}
        legal_audit = r01.overlap_audit(legal, evaluation, shapes)
        self.assertEqual(legal_audit["exact_sample_id_overlap"], 0)
        self.assertEqual(legal_audit["section_overlap_count"], 0)
        self.assertEqual(legal_audit["intersecting_rectangle_pairs"], 0)
        self.assertEqual(legal_audit["unique_shared_voxels"], 0)

        random_train, random_validation = r01.diagnostic_random_split(
            legal, evaluation, seed=2693
        )
        leaky = r01.overlap_audit(random_train, random_validation, shapes)
        contaminated_evaluation = r01.overlap_audit(random_train, evaluation, shapes)
        self.assertEqual(len(random_train), len(legal))
        self.assertEqual(leaky["exact_sample_id_overlap"], 0)
        self.assertGreater(leaky["section_overlap_count"], 0)
        self.assertGreater(leaky["intersecting_rectangle_pairs"], 0)
        self.assertGreater(leaky["unique_shared_voxels"], 0)
        self.assertGreater(contaminated_evaluation["exact_sample_id_overlap"], 0)

    def test_preprocessing_fits_only_supplied_training_windows(self) -> None:
        shape = (10, 10)
        train = r01.FullSection(
            "facies_f3",
            463,
            np.arange(100, dtype=np.float32).reshape(shape),
            np.tile(np.arange(10, dtype=np.uint8), (10, 1)),
        )
        held_out = r01.FullSection(
            "facies_f3",
            489,
            np.full(shape, 10000.0, dtype=np.float32),
            np.tile(np.arange(10, dtype=np.uint8), (10, 1)),
        )
        ref = r01.WindowRef("facies_f3", 463, 0, 0, 10)
        stats, weights, evidence = r01.fit_window_preprocessor(
            {463: train, 489: held_out}, [ref], num_classes=10
        )
        self.assertAlmostEqual(stats.mean, 49.5)
        self.assertEqual(evidence["fit_window_count"], 1)
        self.assertTrue(np.isfinite(weights).all())
        self.assertEqual(evidence["threshold_or_calibration_fit"], "none")

    def test_full_volume_metrics_score_each_voxel_once(self) -> None:
        section = _section("facies_f3", 489, (9, 11))
        config = r01.R01Config(window_size=4, stride=3, batch_size=3, updates=1)
        windows = r01.plan_windows(section, config)
        stats = r01.fit_zscore(section.seismic)
        metrics, predictions = r01.evaluate_full_volume(
            _SignedBinaryModel(),
            {489: section},
            {489: windows},
            stats,
            num_classes=2,
            batch_size=3,
        )
        self.assertEqual(metrics["full_volume_valid_voxels"], section.label.size)
        self.assertEqual(metrics["unique_scored_voxels"], section.label.size)
        self.assertEqual(metrics["evaluated_pixels"], section.label.size)
        self.assertEqual(metrics["coverage_fraction"], 1.0)
        self.assertGreater(metrics["duplicate_prediction_assignments_before_blend"], 0)
        np.testing.assert_array_equal(predictions[489], section.label)

    def test_config_seed_and_overlap_requirements_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "root_seed"):
            r01.R01Config(root_seed=1).validate()
        with self.assertRaisesRegex(ValueError, "overlap"):
            r01.R01Config(window_size=4, stride=4).validate()


@unittest.skipUnless(
    os.environ.get("FACIES_R01_DATA_ROOT"),
    "set FACIES_R01_DATA_ROOT for the explicit real-development R0/R1 smoke",
)
class FaciesP51R01RealDevelopmentSmoke(unittest.TestCase):
    def test_both_tasks_complete_without_test_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = r01.run_r01(
                data_root=Path(os.environ["FACIES_R01_DATA_ROOT"]),
                output_root=Path(directory) / "p5_r01",
                config=r01.R01Config(
                    train_sections=1,
                    validation_sections=1,
                    batch_size=2,
                    updates=2,
                ),
            )
            self.assertFalse(summary["test_archive_opened"])
            self.assertFalse(summary["test_labels_read"])
            self.assertFalse(summary["fresh_blind"])
            for task_id in r01.TASK_IDS:
                with self.subTest(task_id=task_id):
                    task = summary["tasks"][task_id]
                    self.assertEqual(task["r0_status"], "completed")
                    self.assertEqual(task["r1_status"], "completed")
                    self.assertEqual(task["ranking_status"], "not_rankable")
                    self.assertTrue(np.isfinite(task["legal_metrics"]["miou"]))
            verified = r01.verify_artifacts(Path(directory) / "p5_r01")
            self.assertEqual(verified["status"], "verified")
            print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    unittest.main()
