"""Fail-closed contracts for reconstruction-prefixed P5 Stage 3."""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import unittest
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
sys.path.insert(0, str(HERE))

MODULE_NAME = "reconstruction_p5_stage3"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, HERE / f"{MODULE_NAME}.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load reconstruction-prefixed Stage-3 module")
stage3 = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = stage3
SPEC.loader.exec_module(stage3)


def _rehash(record: dict[str, Any]) -> dict[str, Any]:
    record["result_hash"] = stage3.hash_payload(
        {key: value for key, value in record.items() if key != "result_hash"}
    )
    return record


def fake_passed_record(
    mode: str, model_id: str, fold_id: int, repeat_id: int
) -> dict[str, Any]:
    reference = stage3._stage2_reference_record(mode, model_id)
    seed = stage3.REPEAT_SEEDS[repeat_id]
    config = dict(reference["model_config"])
    config["seed"] = seed
    train_id = f"{mode}:fold{fold_id}:train"
    validation_id = f"{mode}:fold{fold_id}:validation"
    purged_id = f"{mode}:fold{fold_id}:purged"
    gpu = model_id in stage3.GPU_MODELS
    record = {
        "schema_version": stage3.SCHEMA_VERSION,
        "track_id": "reconstruction",
        "cell_id": stage3._cell_id(mode, model_id, fold_id, repeat_id),
        "task_id": stage3.p4.protocol(mode).task_id,
        "lane": mode,
        "evaluation_mode": mode,
        "model_id": model_id,
        "fold_id": fold_id,
        "repeat_id": repeat_id,
        "repeat_seed": seed,
        "seed": {"root": 2693, "repeat_id": repeat_id, "model": seed},
        "split_hash": stage3._frozen_split_hashes()[mode],
        "fold_hash": "a" * 64,
        "cache_contract_hash": "b" * 64,
        "input_budget": {"point_train_voxels": 512, "shared_validation_voxels": 2048},
        "fold_fit_audit": {
            "fit_scope": "fold.purge.effective_train_sample_ids only",
            "preprocess_sha256": "c" * 64,
            "target_transform": "identity",
            "class_weights": "not_applicable_regression",
            "calibration": "not_applied",
            "effective_train_sample_ids": [train_id],
            "validation_sample_ids": [validation_id],
            "purged_train_sample_ids": [purged_id],
        },
        "mode_isolation": {
            "input_whitelist": reference["mode_isolation"]["input_whitelist"],
            "constraint_audit": reference["mode_isolation"]["constraint_audit"],
            "strict_constraints_supplied": 0 if mode == "strict" else None,
        },
        "test_firewall": {
            "development_only": True,
            "frozen_test_i_blocks_loaded": [],
            "test_loader_argument_exists": False,
            "test_path_argument_exists": False,
            "test_metrics_computed": False,
            "historical_test_metrics_read": False,
        },
        "stage2_reuse": {
            "hpo_performed": False,
            "preprocessing_changed": False,
            "loss_changed": False,
            "updates_changed": False,
        },
        "status": "passed",
        "reason": None,
        "budget": reference["budget"],
        "model_config": config,
        "updates": reference["budget"]["max_updates"],
        "wall_seconds": 1.0,
        "metrics": {
            "rmse": 0.02 + fold_id / 1000 + repeat_id / 10000,
            "mae": 0.015,
            "spectral_log_rmse": 0.4,
            "bias": 0.0,
            "r2": 0.2,
            "pearson_r": 0.5,
            "valid_voxels": 2048,
        },
        "oof_prediction": {
            "archive_name": "oof_prediction.npz",
            "sha256": "d" * 64,
            "bytes": 100,
            "validation_voxels": 2048,
            "scope": "sampled buffered-development OOF validation only",
        },
        "resources": {
            "peak_rss_kib": 1000,
            "peak_cuda_bytes": 1024 if gpu else 0,
            "gpu_lock": (
                {
                    "required": True,
                    "acquired": True,
                    "mechanism": "external flock -w 900",
                    "timeout_seconds": 900,
                    "wait_seconds": 0.1,
                }
                if gpu
                else {"required": False, "acquired": False, "wait_seconds": 0.0}
            ),
        },
    }
    return _rehash(record)


class ReconstructionStage3FrozenMatrixTest(unittest.TestCase):
    def test_exact_top_three_five_folds_three_seeds_and_ninety_cells(self) -> None:
        self.assertEqual(
            stage3.MODELS["strict"],
            ("pykrige_ok3d", "gpytorch_svgp", "gstools_krige_condsrf"),
        )
        self.assertEqual(
            stage3.MODELS["conditional"],
            ("pykrige_ok3d", "gpytorch_svgp", "scipy_rbf_neighbors"),
        )
        self.assertEqual(stage3.FOLD_IDS, (0, 1, 2, 3, 4))
        self.assertEqual(
            stage3.REPEAT_SEEDS, (1867973658, 2137841944, 3902865753)
        )
        self.assertEqual(len(stage3.expected_cell_keys()), 90)
        self.assertEqual(len(set(stage3.expected_cell_keys())), 90)

    def test_repeat_seeds_are_stable_shared_derivations(self) -> None:
        for repeat_id, seed in enumerate(stage3.REPEAT_SEEDS):
            self.assertEqual(
                stage3.SeedTree(2693).seed("model", "p5-stage3", repeat_id), seed
            )

    def test_cli_has_no_test_or_hpo_surface(self) -> None:
        parameters = inspect.signature(stage3.run_cell).parameters
        for name in ("data_dir", "test_loader", "test_data", "test_path", "hpo"):
            self.assertNotIn(name, parameters)
        parser = stage3.build_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        self.assertEqual(set(subparsers.choices), {"prepare-cache", "run-cell", "collate"})
        self.assertNotIn("scan_patch_catalog", inspect.getsource(stage3.prepare_cache))


class ReconstructionStage3FailClosedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [fake_passed_record(*key) for key in stage3.expected_cell_keys()]

    def test_full_valid_matrix_passes_and_duplicate_is_rejected(self) -> None:
        stage3.validate_record_set(self.records)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            stage3.validate_record_set([*self.records, dict(self.records[0])])

    def test_missing_cell_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing expected"):
            stage3.validate_record_set(self.records[:-1])

    def test_seed_and_split_tampering_are_rejected(self) -> None:
        record = dict(self.records[0])
        record["repeat_seed"] += 1
        _rehash(record)
        with self.assertRaisesRegex(ValueError, "repeat seed"):
            stage3.validate_cell_record(record, "strict", "pykrige_ok3d", 0, 0)
        record = dict(self.records[0])
        record["split_hash"] = "0" * 64
        _rehash(record)
        with self.assertRaisesRegex(ValueError, "split hash"):
            stage3.validate_cell_record(record, "strict", "pykrige_ok3d", 0, 0)

    def test_test_firewall_and_cross_lane_pollution_are_rejected(self) -> None:
        record = dict(self.records[0])
        record["test_firewall"] = dict(record["test_firewall"])
        record["test_firewall"]["historical_test_metrics_read"] = True
        _rehash(record)
        with self.assertRaisesRegex(ValueError, "frozen-test"):
            stage3.validate_cell_record(record, "strict", "pykrige_ok3d", 0, 0)
        record = dict(self.records[0])
        record["lane"] = "conditional"
        record["evaluation_mode"] = "conditional"
        _rehash(record)
        with self.assertRaisesRegex(ValueError, "identity|lane"):
            stage3.validate_cell_record(record, "strict", "pykrige_ok3d", 0, 0)

    def test_budget_config_and_gpu_evidence_tampering_are_rejected(self) -> None:
        record = dict(self.records[0])
        record["budget"] = dict(record["budget"])
        record["budget"]["max_updates"] += 1
        _rehash(record)
        with self.assertRaisesRegex(ValueError, "budget"):
            stage3.validate_cell_record(record, "strict", "pykrige_ok3d", 0, 0)
        gpu = next(record for record in self.records if record["model_id"] == "gpytorch_svgp")
        gpu = json.loads(json.dumps(gpu))
        gpu["resources"]["gpu_lock"]["wait_seconds"] = None
        _rehash(gpu)
        with self.assertRaises((TypeError, ValueError)):
            stage3.validate_cell_record(
                gpu, gpu["lane"], gpu["model_id"], gpu["fold_id"], gpu["repeat_id"]
            )

    def test_structured_failed_cell_is_retained_without_fake_metrics(self) -> None:
        record = dict(self.records[0])
        record.update(
            {
                "status": "failed",
                "reason": {"code": "data_blocked", "message": "fixture"},
                "metrics": None,
            }
        )
        _rehash(record)
        stage3.validate_cell_record(record, "strict", "pykrige_ok3d", 0, 0)

    def test_leaderboards_are_independent_and_rank_by_frozen_order(self) -> None:
        strict = stage3.build_leaderboard("strict", self.records)
        conditional = stage3.build_leaderboard("conditional", self.records)
        self.assertTrue(strict["rankable"])
        self.assertTrue(conditional["rankable"])
        self.assertNotEqual(strict["task_id"], conditional["task_id"])
        self.assertEqual(strict["expected_cells"], 45)
        self.assertEqual(conditional["expected_cells"], 45)
        self.assertTrue(all(entry["rmse_95pct_bootstrap_ci"] for entry in strict["entries"]))


@unittest.skipUnless(
    (HERE / "p5_stage3_summary.json").is_file(),
    "canonical Stage-3 evidence has not been generated yet",
)
class ReconstructionStage3PortableEvidenceTest(unittest.TestCase):
    def test_canonical_results_cover_exact_matrix_without_lane_or_test_leakage(self) -> None:
        rows = [
            json.loads(line)
            for line in (HERE / "p5_stage3_results.jsonl").read_text().splitlines()
        ]
        summary = json.loads((HERE / "p5_stage3_summary.json").read_text())
        self.assertEqual(len(rows), 90)
        self.assertEqual(summary["expected_cells"], 90)
        self.assertEqual(summary["attempted_cells"], 90)
        self.assertEqual(sum(summary["counts"].values()), 90)
        self.assertEqual(summary["frozen_test_i_blocks_loaded"], [])
        self.assertFalse(summary["historical_test_metrics_read"])
        stage3.validate_record_set(rows)
        for row in rows:
            if row["status"] == "passed":
                self.assertEqual(row["environment"]["primary_environment"], "torch-common")
            if row["status"] == "passed" and row["model_id"] in stage3.GPU_MODELS:
                self.assertEqual(row["model_config"]["device"], "cuda:0")
                self.assertGreater(row["resources"]["peak_cuda_bytes"], 0)
                self.assertTrue(row["resources"]["gpu_lock"]["acquired"])
                self.assertGreaterEqual(row["resources"]["gpu_lock"]["wait_seconds"], 0)

    def test_oof_and_visualization_manifests_are_private_and_development_only(self) -> None:
        oof = json.loads((HERE / "p5_stage3_oof_manifest.json").read_text())
        visualization = json.loads(
            (HERE / "p5_stage3_visualization_manifest.json").read_text()
        )
        self.assertEqual(oof["frozen_test_i_blocks_loaded"], [])
        self.assertTrue(
            all(entry["path"].startswith("_tmp/p5_stage3_reconstruction/") for entry in oof["entries"])
        )
        self.assertEqual(visualization["frozen_test_i_blocks_loaded"], [])
        for mode in stage3.MODES:
            figure = PROJECT_ROOT / visualization["figures"][mode]["path"]
            self.assertTrue(figure.is_file())
            self.assertEqual(figure.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            diagnostics = visualization["figures"][mode]["diagnostics"]
            self.assertTrue(any("CDF" in name for name in diagnostics))
            self.assertTrue(any("variogram" in name for name in diagnostics))
            self.assertTrue(any("fold-by-repeat" in name for name in diagnostics))


if __name__ == "__main__":
    unittest.main()
