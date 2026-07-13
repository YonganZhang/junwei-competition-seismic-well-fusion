from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from _code.dataset_io import load_dataset
from _code.ml_framework.model_registry import get_model
from _code.ml_framework.preprocess import denormalize, fit_zscore, normalize
from audit_pipeline import run_audit
from build_dataset import parse_label_intervals
from pipeline_contract import (
    CLASS_NAMES,
    FAMILY_PARTITIONS,
    LOG_ALIASES,
    TARGET_CURVE_TYPE,
    TARGET_SOURCE,
    assert_family_isolation,
    classification_metrics_from_confusion,
    mother_family,
    normalize_well_id,
    validate_class_name,
)
from train_baseline import LithofaciesSamples, assert_nonempty_loader


FORCE_NO_ARTIFACTS = os.environ.get("LITHOFACIES_FORCE_NO_ARTIFACTS") == "1"
RUN_INTEGRATION = os.environ.get("LITHOFACIES_RUN_INTEGRATION") == "1"

DATA_ARTIFACTS = (
    PROJECT_ROOT / "_data/processed/lithofacies/train.h5",
    PROJECT_ROOT / "_data/processed/lithofacies/test.h5",
    PROJECT_ROOT / "_sandbox/volve_data/Volve_Well_logs.zip",
    TRACK_DIR / "_outputs/split_manifest.json",
)
COMPLETION_ARTIFACTS = DATA_ARTIFACTS + (
    TRACK_DIR / "_outputs/multimodal_mlp/checkpoints/best.ckpt",
    TRACK_DIR / "_outputs/multimodal_mlp/checkpoints/history.json",
    TRACK_DIR / "_outputs/multimodal_mlp/metrics.json",
    TRACK_DIR / "_outputs/multimodal_mlp/loss_curve.png",
    TRACK_DIR / "_outputs/multimodal_mlp/confusion_matrix.png",
    TRACK_DIR / "_outputs/multimodal_mlp/best_checkpoint_predictions.png",
    TRACK_DIR / "_outputs/completion_audit.json",
)


def integration_gate(paths: tuple[Path, ...], gate_name: str) -> tuple[bool, str]:
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in paths if not path.is_file()]
    if not RUN_INTEGRATION or FORCE_NO_ARTIFACTS:
        return False, f"set LITHOFACIES_RUN_INTEGRATION=1 to run {gate_name} integration gate"
    if missing:
        return False, f"{gate_name} integration gate缺少资产: {missing}"
    return True, ""


DATA_INTEGRATION_AVAILABLE, DATA_INTEGRATION_SKIP_REASON = integration_gate(
    DATA_ARTIFACTS,
    "dataset",
)
COMPLETION_INTEGRATION_AVAILABLE, COMPLETION_INTEGRATION_SKIP_REASON = integration_gate(
    COMPLETION_ARTIFACTS,
    "completion",
)


class PipelineUnitContractTests(unittest.TestCase):
    """Artifact-free contracts that must pass in a clean checkout."""

    def test_fixed_nine_classes_and_unknown_rejection(self) -> None:
        self.assertEqual(len(CLASS_NAMES), 9)
        self.assertEqual(len(set(CLASS_NAMES)), 9)
        with self.assertRaises(ValueError):
            validate_class_name("UNKNOWN")
        with self.assertRaises(ValueError):
            validate_class_name("UNDEFINED")

    def test_mother_family_normalization_and_isolation(self) -> None:
        self.assertEqual(normalize_well_id("NO 15_9-F-15 a"), "15/9-F-15 A")
        self.assertEqual(mother_family("15/9-F-15 C"), "15/9-F-15")
        self.assertEqual(mother_family("15/9-19 BT2"), "15/9-19")
        records = [
            {"partition": partition, "family_id": family}
            for family, partition in FAMILY_PARTITIONS.items()
        ]
        isolated = assert_family_isolation(records)
        self.assertTrue(isolated["train"])
        self.assertTrue(isolated["guard"])
        self.assertTrue(isolated["test"])

    def test_normalization_round_trip(self) -> None:
        physical = np.array([-3.0, 0.5, 2.0, 8.0], dtype=np.float64)
        stats = fit_zscore(physical)
        reconstructed = denormalize(normalize(physical, stats), stats)
        np.testing.assert_allclose(reconstructed, physical, atol=1e-10, rtol=1e-10)

    def test_dynamic_registration_and_nonempty_batch(self) -> None:
        well_log_shape = (26, 33)
        seismic_shape = (3, 3, 33)
        model = get_model(
            "multimodal_mlp",
            models_package="models",
            num_classes=len(CLASS_NAMES),
            well_log_shape=well_log_shape,
            seismic_shape=seismic_shape,
            hidden_size=8,
        )
        synthetic_samples = [
            {
                "well_log_seq": np.zeros(well_log_shape, dtype=np.float32),
                "seismic_patch": np.zeros(seismic_shape, dtype=np.float32),
                "label": index,
            }
            for index in range(2)
        ]
        loader = DataLoader(LithofaciesSamples(synthetic_samples), batch_size=2)
        assert_nonempty_loader(loader, "unit")
        well_log, seismic, label = next(iter(loader))
        logits = model(well_log, seismic)
        self.assertEqual(tuple(logits.shape), (2, len(CLASS_NAMES)))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(label.ndim, 1)

    def test_metrics_remain_finite_with_zero_support_classes(self) -> None:
        confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
        confusion[0, 0] = 3
        confusion[1, 0] = 2
        metrics = classification_metrics_from_confusion(confusion)
        self.assertTrue(np.isfinite(metrics["accuracy"]))
        self.assertTrue(np.isfinite(metrics["balanced_accuracy"]))
        self.assertTrue(np.isfinite(metrics["macro_f1"]))
        self.assertEqual(metrics["per_class"][2]["support"], 0)
        self.assertEqual(metrics["per_class"][2]["recall"], 0.0)


@unittest.skipUnless(
    DATA_INTEGRATION_AVAILABLE,
    DATA_INTEGRATION_SKIP_REASON,
)
class DatasetIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.saved_train = list(load_dataset("lithofacies", "train"))
        cls.saved_test = list(load_dataset("lithofacies", "test"))

    def test_original_label_trace_and_fixed_nine_classes(self) -> None:
        intervals = parse_label_intervals()
        self.assertEqual(len(intervals), 139)
        self.assertEqual(len({interval.well_id for interval in intervals}), 11)
        self.assertEqual({interval.class_name for interval in intervals}, set(CLASS_NAMES))
        self.assertTrue(all(interval.source_member.endswith("Facies.xlsx") for interval in intervals))
        self.assertTrue(all(interval.source_row >= 2 for interval in intervals))

    def test_saved_labels_exclude_unknown_and_non_gm09(self) -> None:
        for sample in self.saved_train + self.saved_test:
            trace = sample["meta"]["label_trace"]
            self.assertEqual(trace["source"], TARGET_SOURCE)
            self.assertEqual(trace["curve_type"], TARGET_CURVE_TYPE)
            self.assertNotIn(trace["class_name"], ("UNKNOWN", "UNDEFINED"))
            self.assertIn(trace["class_name"], CLASS_NAMES)

    def test_saved_family_isolation_and_train_only_normalization(self) -> None:
        records = [
            {
                "partition": sample["meta"]["partition"],
                "family_id": sample["meta"]["family_id"],
            }
            for sample in self.saved_train + self.saved_test
        ]
        isolated = assert_family_isolation(records)
        for partition, families in isolated.items():
            for family in families:
                self.assertEqual(FAMILY_PARTITIONS[family], partition)
        scopes = {
            sample["meta"]["normalization_fit_scope"]
            for sample in self.saved_train + self.saved_test
        }
        self.assertEqual(scopes, {"train_mother_well_families_only"})

    def test_raw_las_curve_whitelist_has_no_target_derived_channel(self) -> None:
        manifest = json.loads((TRACK_DIR / "_outputs/split_manifest.json").read_text())
        allowed = {alias for aliases in LOG_ALIASES.values() for alias in aliases}
        for report in manifest["per_well"].values():
            for member in report["selected_las_members"]:
                upper = member.upper()
                self.assertNotIn("FACIES", upper)
                self.assertNotIn("INTERPRETATION_CUSTOMER", upper)
                self.assertNotIn("WLC_PETRO_COMPUTED", upper)
            for member_report in report["las_member_reports"]:
                for curve in member_report.get("selected_curves", {}).values():
                    self.assertIn(curve["mnemonic"].upper(), allowed)

    def test_saved_arrays_are_finite_and_multimodal(self) -> None:
        for sample in self.saved_train + self.saved_test:
            seismic = np.asarray(sample["seismic_patch"])
            well_log = np.asarray(sample["well_log_seq"])
            self.assertEqual(seismic.ndim, 3)
            self.assertEqual(well_log.ndim, 2)
            self.assertEqual(well_log.shape[0] % 2, 0)
            self.assertTrue(np.isfinite(seismic).all())
            self.assertTrue(np.isfinite(well_log).all())
            mask = well_log[well_log.shape[0] // 2 :]
            self.assertGreater(mask.sum(), 0)
            self.assertTrue(np.isin(mask, (0.0, 1.0)).all())


@unittest.skipUnless(
    COMPLETION_INTEGRATION_AVAILABLE,
    COMPLETION_INTEGRATION_SKIP_REASON,
)
class CompletionIntegrationTests(unittest.TestCase):
    def test_completion_audit_passes_on_real_artifacts(self) -> None:
        audit = run_audit()
        self.assertEqual(audit["status"], "PASS")
        self.assertTrue(all(check["status"] == "PASS" for check in audit["checks"]))


if __name__ == "__main__":
    unittest.main()
