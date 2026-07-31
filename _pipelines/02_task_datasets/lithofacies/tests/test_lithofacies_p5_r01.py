"""Contract and bounded lifecycle tests for the lithofacies P5.1 R0/R1 runner."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TRACK_DIR = Path(__file__).resolve().parents[1]
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

import lithofacies_p5_r01 as r01  # noqa: E402
from p4_contract import CLASS_NAMES, DEVELOPMENT_FAMILIES, lithofacies_task_spec  # noqa: E402


EXPECTED_CLASSES = (
    "F-MARSH",
    "F-MOUTHBAR",
    "F-OFFSHORE",
    "F-LOWER SHOREFACE",
    "F-UPPER SHOREFACE",
    "F-TIDAL BAR",
    "F-TIDAL CHANNEL",
    "F-TIDAL FLAT MUDDY",
    "F-TIDAL FLAT SANDY",
)


def synthetic_arrays() -> dict[str, np.ndarray]:
    """Make four-family, fixed-nine data with deliberately overlapping windows."""
    rng = np.random.default_rng(r01.ROOT_SEED)
    labels = np.tile(np.arange(r01.NUM_CLASSES, dtype=np.int64), 4)
    families = np.repeat(np.asarray(DEVELOPMENT_FAMILIES), r01.NUM_CLASSES)
    wells = np.repeat(np.asarray([f"synthetic-{index}" for index in range(4)]), r01.NUM_CLASSES)
    logs = np.empty((len(labels), len(r01.LOG_CHANNELS), r01.CONTEXT_LENGTH), dtype=np.float32)
    seismic = np.empty((len(labels), 3, 3, r01.CONTEXT_LENGTH), dtype=np.float32)
    for family_index in range(4):
        log_base = rng.normal(size=(len(r01.LOG_CHANNELS), r01.CONTEXT_LENGTH + 8)).astype(np.float32)
        seismic_base = rng.normal(size=(3, 3, r01.CONTEXT_LENGTH + 8)).astype(np.float32)
        for offset in range(r01.NUM_CLASSES):
            index = family_index * r01.NUM_CLASSES + offset
            logs[index] = log_base[:, offset : offset + r01.CONTEXT_LENGTH]
            seismic[index] = seismic_base[:, :, offset : offset + r01.CONTEXT_LENGTH]
    return {
        "physical_logs": logs,
        "log_masks": np.ones_like(logs, dtype=np.uint8),
        "physical_seismic": seismic,
        "labels": labels,
        "families": families.astype(np.str_),
        "wells": wells.astype(np.str_),
        "sample_ids": np.asarray([f"sample-{index:03d}" for index in range(len(labels))]),
        "interval_keys": np.asarray([f"interval-{index // r01.NUM_CLASSES}" for index in range(len(labels))]),
        "center_md_m": np.full(len(labels), np.nan, dtype=np.float64),
        "time_ms": np.arange(len(labels), dtype=np.float64),
    }


class R0ContractTest(unittest.TestCase):
    def test_fixed_schema_metric_lanes_and_sealed_identity(self) -> None:
        arrays = synthetic_arrays()
        contract = r01.build_r0_contract(arrays, development_sha256="a" * 64)
        self.assertEqual(CLASS_NAMES, EXPECTED_CLASSES)
        self.assertEqual(contract["label_contract"]["class_names"], list(EXPECTED_CLASSES))
        self.assertEqual(contract["metric_contract"]["primary_metric"], "fixed_schema_macro_f1")
        self.assertEqual(contract["metric_contract"]["primary_metric_count"], 1)
        self.assertEqual(
            contract["metric_contract"]["supported_class_macro_f1_role"], "diagnostic_only"
        )
        lanes = {row["lane_id"]: row for row in contract["lane_contract"]["lanes"]}
        self.assertEqual(set(lanes), {"W-P", "M-P", "W-S", "M-S"})
        self.assertEqual(lanes["W-P"]["status"], "available")
        self.assertEqual(lanes["M-P"]["status"], "available")
        self.assertEqual(lanes["W-S"]["status"], "not_rankable")
        self.assertEqual(lanes["M-S"]["finite_center_md_count"], 0)
        self.assertFalse(contract["sealed_holdout"]["fresh_blind"])
        self.assertFalse(contract["sealed_holdout"]["physical_test_accessed"])
        self.assertTrue(contract["sealed_holdout"]["identity_only"])
        self.assertFalse(contract["completion_contract"]["seventy_five_percent_rankable"])
        self.assertEqual(lithofacies_task_spec().primary_metrics, ("fixed_schema_macro_f1",))

    def test_source_has_no_physical_test_or_legacy_prepare_entry(self) -> None:
        source = (TRACK_DIR / "lithofacies_p5_r01.py").read_text(encoding="utf-8")
        forbidden = ("test" + ".h5", "load_frozen_test", "prepare_" + "run(")
        for token in forbidden:
            self.assertNotIn(token, source)
        self.assertEqual(set(r01._parser()._subparsers._group_actions[0].choices), {"prepare", "run"})


class SplitAndPreprocessingTest(unittest.TestCase):
    def test_logo_is_family_clean_while_random_split_is_diagnostic(self) -> None:
        arrays = synthetic_arrays()
        for fold in r01.logo4(arrays["families"]):
            leakage = r01.leakage_diagnostics(
                arrays, fold["train_indices"], fold["validation_indices"]
            )
            self.assertEqual(leakage["family_overlap_count"], 0)
            self.assertEqual(leakage["well_overlap_count"], 0)
            self.assertEqual(leakage["source_interval_overlap_count"], 0)
            self.assertEqual(leakage["exact_shifted_window_pair_count"], 0)
        random_leakage = [
            r01.leakage_diagnostics(arrays, fold["train_indices"], fold["validation_indices"])
            for fold in r01.random_kfold4(len(arrays["labels"]))
        ]
        self.assertTrue(all(row["family_overlap_count"] > 0 for row in random_leakage))
        self.assertTrue(any(row["exact_shifted_window_pair_count"] > 0 for row in random_leakage))

    def test_fold_train_statistics_ignore_validation_outlier(self) -> None:
        arrays = synthetic_arrays()
        fold = r01.logo4(arrays["families"])[0]
        validation = fold["validation_indices"]
        arrays["physical_logs"][validation, 0] = 1.0e9
        arrays["physical_seismic"][validation] = -1.0e9
        expected_log_mean = float(
            arrays["physical_logs"][fold["train_indices"], 0].mean()
        )
        expected_seismic_mean = float(
            arrays["physical_seismic"][fold["train_indices"]].mean()
        )
        _, _, weights, evidence = r01._fit_fold_preprocessing(
            arrays, fold["train_indices"], validation, modality="M"
        )
        evidence.pop("seismic_train")
        evidence.pop("seismic_validation")
        self.assertAlmostEqual(evidence["log_stats"][0]["mean"], expected_log_mean, places=5)
        self.assertAlmostEqual(evidence["seismic_stats"]["mean"], expected_seismic_mean, places=5)
        self.assertEqual(evidence["fit_families"], sorted(DEVELOPMENT_FAMILIES[1:]))
        self.assertTrue(np.isfinite(weights).all())


@unittest.skipUnless(
    os.environ.get("LITHOFACIES_R01_TINY") == "1",
    "set LITHOFACIES_R01_TINY=1 for the bounded 32-cell sklearn lifecycle",
)
class TinyLifecycleTest(unittest.TestCase):
    def test_all_preregistered_cells_and_portable_artifacts(self) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError as exc:  # pragma: no cover - explicit environment gate
            self.skipTest(str(exc))
        arrays = synthetic_arrays()
        temporary = Path(tempfile.mkdtemp(prefix="r01-test-", dir=TRACK_DIR / "_outputs"))
        self.addCleanup(shutil.rmtree, temporary, True)
        batch = temporary / "runtime" / "development.npz"
        output = temporary / "portable"
        output.mkdir(parents=True)
        contract = r01.build_r0_contract(arrays, development_sha256="a" * 64)
        r01._atomic_json(output / "r0_contract.json", contract)
        batch.parent.mkdir(parents=True)
        manifest = {
            "schema_version": r01.BATCH_SCHEMA,
            "task_id": contract["task_id"],
            "root_seed": r01.ROOT_SEED,
            "class_names": list(CLASS_NAMES),
            "development_data": {"basename": "train.h5", "sha256": "a" * 64},
            "r0_contract_hash": contract["contract_hash"],
            "development_families": list(DEVELOPMENT_FAMILIES),
            "sample_count": len(arrays["labels"]),
            "physical_test_accessed": False,
            "known_holdout_artifacts_read": False,
        }
        np.savez_compressed(batch, manifest=np.asarray(r01._canonical_json(manifest)), **arrays)
        summary, exit_code = r01.run_r1(batch, output)
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["completed_cells"], 32)
        self.assertEqual(len(summary["conditions"]), 8)
        self.assertEqual(len(summary["leakage_matrix"]), 8)
        self.assertEqual(len(summary["paired_protocol_deltas"]), 4)
        self.assertTrue(all(not row["rank_eligible"] for row in summary["conditions"]))
        legal_rows = [
            row for row in summary["leakage_matrix"] if row["split_role"] == "legal_grouped"
        ]
        self.assertTrue(all(row["family_overlap_count"] == 0 for row in legal_rows))
        self.assertTrue(all(row["exact_shifted_window_pair_count"] == 0 for row in legal_rows))
        self.assertEqual(summary["s_lane"]["status"], "not_rankable")
        results = (output / "r1_results.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(results), 32)
        manifest_payload = json.loads((output / "artifact_manifest.json").read_text())
        self.assertFalse(manifest_payload["absolute_paths_serialized"])
        self.assertFalse(manifest_payload["checkpoints_written"])


if __name__ == "__main__":
    unittest.main()
