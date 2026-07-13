#!/usr/bin/env python3
"""Integration checks for the completed audited real-Volve fault run."""
from __future__ import annotations

import json
import hashlib
import math
import os
import unittest
from pathlib import Path

from audit_utils import (
    missing_historical_artifacts,
    sha256_file,
    validated_run_dir,
    verify_historical_artifacts,
)
from baseline import load_samples


RUN_NAME = os.environ.get("FAULT_RUN_NAME", "audited_v2")
RUN_DIR = validated_run_dir(RUN_NAME)
TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]


def integration_asset_gaps() -> list[str]:
    required = (
        PROJECT_ROOT / "_data" / "processed" / "fault" / "train.h5",
        PROJECT_ROOT / "_data" / "processed" / "fault" / "test.h5",
        RUN_DIR / "build_summary.json",
        RUN_DIR / "baseline_metrics.json",
        RUN_DIR / "visualization_report.json",
        RUN_DIR / "reproducibility_report.json",
        RUN_DIR / "prediction_visualization.png",
        RUN_DIR / "checkpoints" / "best.ckpt",
    )
    gaps = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.is_file()]
    try:
        gaps.extend(f"historical:{relative}" for relative in missing_historical_artifacts())
    except FileNotFoundError as exc:
        gaps.append(str(exc))
    return gaps


INTEGRATION_ASSET_GAPS = integration_asset_gaps()


class AuditedRealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if INTEGRATION_ASSET_GAPS:
            raise unittest.SkipTest(
                "data-dependent integration assets are unavailable: "
                + ", ".join(INTEGRATION_ASSET_GAPS)
            )
        cls.summary = json.loads((RUN_DIR / "build_summary.json").read_text(encoding="utf-8"))
        cls.metrics = json.loads((RUN_DIR / "baseline_metrics.json").read_text(encoding="utf-8"))
        cls.visualization = json.loads(
            (RUN_DIR / "visualization_report.json").read_text(encoding="utf-8")
        )
        cls.train = load_samples("train")
        cls.test = load_samples("test")

    def test_train_guard_test_inline_contract(self) -> None:
        split = self.summary["split_plan"]
        train_allowed = set(range(split["train"][0], split["train"][1] + 1))
        guard = set(range(split["guard"][0], split["guard"][1] + 1))
        test_allowed = set(range(split["test"][0], split["test"][1] + 1))
        train_inlines = {int(position["inline"]) for position in self.train.positions}
        test_inlines = {int(position["inline"]) for position in self.test.positions}
        self.assertTrue(train_inlines <= train_allowed)
        self.assertTrue(test_inlines <= test_allowed)
        self.assertFalse(train_inlines & guard)
        self.assertFalse(test_inlines & guard)
        self.assertFalse(train_inlines & test_inlines)
        train_hashes = {
            hashlib.sha256(patch.tobytes()).hexdigest() for patch in self.train.patches
        }
        test_hashes = {
            hashlib.sha256(patch.tobytes()).hexdigest() for patch in self.test.patches
        }
        self.assertFalse(train_hashes & test_hashes)

    def test_train_fitted_normalization_is_identical(self) -> None:
        self.assertEqual(self.train.normalization_stats, self.test.normalization_stats)
        self.assertEqual(self.summary["normalization"]["fit_split"], "train_fit")
        self.assertFalse(self.summary["normalization"]["validation_refit"])
        self.assertFalse(self.summary["normalization"]["test_refit"])

    def test_real_dataset_and_metrics_are_finite(self) -> None:
        self.assertEqual(len(self.train.patches), 256)
        self.assertEqual(len(self.test.patches), 96)
        for key in ("precision", "recall", "f1", "dice", "iou"):
            self.assertTrue(math.isfinite(float(self.metrics["test_metrics"][key])))

    def test_best_checkpoint_drives_visualization(self) -> None:
        checkpoint = TRACK_DIR / self.visualization["checkpoint"]
        image = TRACK_DIR / self.visualization["output"]
        self.assertTrue(checkpoint.is_file())
        self.assertTrue(image.is_file())
        self.assertEqual(sha256_file(image), self.visualization["output_sha256"])
        for key in ("precision", "recall", "f1"):
            self.assertAlmostEqual(
                float(self.visualization["metrics"][key]),
                float(self.metrics["test_metrics"][key]),
            )

    def test_dataset_hashes_match_build_and_training_records(self) -> None:
        self.assertEqual(self.summary["dataset_sha256"], self.metrics["dataset_sha256"])

    def test_independent_reproduction_matches_exactly(self) -> None:
        report = json.loads(
            (RUN_DIR / "reproducibility_report.json").read_text(encoding="utf-8")
        )
        self.assertTrue(report["all_checks_passed"])
        self.assertFalse(report["failed_checks"])

    def test_historical_baseline_is_unchanged(self) -> None:
        self.assertTrue(verify_historical_artifacts())


if __name__ == "__main__":
    unittest.main()
