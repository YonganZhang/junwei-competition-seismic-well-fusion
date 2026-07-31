"""Reconstruction-prefixed P5 Stage-2 budget, split, and firewall tests."""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
sys.path.insert(0, str(HERE))

import p4_reconstruction as p4  # noqa: E402


MODULE_NAME = "reconstruction_p5_stage2"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, HERE / f"{MODULE_NAME}.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load reconstruction-prefixed Stage-2 module")
stage2 = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = stage2
SPEC.loader.exec_module(stage2)


class ReconstructionStage2Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        catalog, records = p4.synthetic_catalog_and_records()
        modes = {
            mode: stage2.prepare_mode_cache(mode, catalog, records, self.cache)
            for mode in stage2.MODES
        }
        stage2.atomic_write_json(
            self.cache / "cache_manifest.json",
            {
                "schema_version": stage2.CACHE_SCHEMA_VERSION,
                "track_id": "reconstruction",
                "root_seed": stage2.ROOT_SEED,
                "source_container_names": ["synthetic"],
                "frozen_test_i_blocks_loaded": [],
                "modes": modes,
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()


class ReconstructionStage2SplitAndIsolationTest(ReconstructionStage2Fixture):
    def test_uses_p4_first_fold_and_disjoint_buffered_development_sets(self) -> None:
        catalog, _ = p4.synthetic_catalog_and_records()
        manifest = p4.build_spatial_manifest("strict", catalog)
        audit = json.loads((self.cache / "cache_manifest.json").read_text())["modes"]["strict"]
        self.assertEqual(audit["fold_id"], 0)
        self.assertEqual(audit["split_hash"], manifest.stable_hash())
        train = set(audit["effective_train_sample_ids"])
        validation = set(audit["validation_sample_ids"])
        purged = set(audit["purged_train_sample_ids"])
        self.assertFalse(train & validation)
        self.assertFalse(train & purged)
        self.assertFalse(validation & purged)
        self.assertEqual(audit["frozen_test_i_blocks_loaded"], [])

    def test_strict_and_conditional_caches_are_independent_and_share_lane_metrics(self) -> None:
        strict_spec, strict_train, strict_val, strict_audit, _ = stage2.load_mode_cache(
            self.cache, "strict", "point"
        )
        conditional_spec, conditional_train, conditional_val, conditional_audit, _ = (
            stage2.load_mode_cache(self.cache, "conditional", "point")
        )
        self.assertNotEqual(strict_spec.task_id, conditional_spec.task_id)
        self.assertEqual(strict_train[0].inputs["features"].shape[1], 6)
        self.assertEqual(conditional_train[0].inputs["features"].shape[1], 7)
        self.assertEqual(strict_audit["constraint_audit"]["constraints_supplied_to_model"], 0)
        self.assertIn("conditional_idw_porosity", conditional_audit["input_whitelist"])
        self.assertEqual(
            strict_val.coordinates["metric_indices_kji"].shape[0],
            strict_audit["input_budget"]["shared_validation_voxels"],
        )
        self.assertEqual(
            conditional_val.coordinates["metric_indices_kji"].shape[0],
            conditional_audit["input_budget"]["shared_validation_voxels"],
        )

    def test_point_and_volume_representations_use_identical_validation_targets(self) -> None:
        for mode in stage2.MODES:
            with self.subTest(mode=mode):
                spec, _, point, _, _ = stage2.load_mode_cache(self.cache, mode, "point")
                _, _, volume, _, _ = stage2.load_mode_cache(self.cache, mode, "volume")
                target_name = spec.targets[0]
                point_target = np.asarray(point.targets[target_name])
                volume_target = np.asarray(volume.targets[target_name])[
                    np.asarray(volume.target_masks[target_name], dtype=bool)
                ]
                np.testing.assert_allclose(point_target, volume_target)
                np.testing.assert_array_equal(
                    point.coordinates["metric_indices_kji"],
                    volume.coordinates["metric_indices_kji"],
                )


class ReconstructionStage2BudgetAndFirewallTest(ReconstructionStage2Fixture):
    def test_budget_caps_match_frozen_stage2_protocol(self) -> None:
        traditional = stage2.budget_for({"trainable": False, "batch_representation": "point"})
        point = stage2.budget_for({"trainable": True, "batch_representation": "point"})
        volume = stage2.budget_for({"trainable": True, "batch_representation": "volume"})
        self.assertLessEqual(traditional["max_wall_seconds"], 300)
        self.assertLessEqual(point["max_updates"], 200)
        self.assertLessEqual(point["max_wall_seconds"], 600)
        self.assertLessEqual(volume["max_updates"], 80)
        self.assertLessEqual(volume["max_wall_seconds"], 900)
        with self.assertRaises(ValueError):
            stage2.validate_budget(volume, 81, 1.0)

    def test_runner_has_no_frozen_test_surface(self) -> None:
        parameters = inspect.signature(stage2.run_cell).parameters
        self.assertNotIn("test_loader", parameters)
        self.assertNotIn("test_data", parameters)
        self.assertNotIn("test_path", parameters)
        parser = stage2.build_parser()
        subparsers = next(
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
        )
        self.assertEqual(set(subparsers.choices), {"prepare-cache", "run-cell", "collate"})
        self.assertNotIn('add_parser("test"', inspect.getsource(stage2))

    def test_collation_rejects_unresolved_pilot_exception(self) -> None:
        record = {
            "schema_version": stage2.SCHEMA_VERSION,
            "lane": "strict",
            "evaluation_mode": "strict",
            "model_id": "monai_segresnet3d",
            "fold_id": 0,
            "reason": {"code": "pilot_exception"},
        }
        record["result_hash"] = stage2.hash_payload(record)
        with self.assertRaisesRegex(ValueError, "pilot_exception"):
            stage2.validate_cell_record(record, "strict", "monai_segresnet3d")
        record["reason"] = {"code": "gpu_required_by_stage2_protocol"}
        record["result_hash"] = stage2.hash_payload(
            {key: value for key, value in record.items() if key != "result_hash"}
        )
        with self.assertRaisesRegex(ValueError, "gpu_required_by_stage2_protocol"):
            stage2.validate_cell_record(record, "strict", "monai_segresnet3d")

    def test_cuda_path_uses_protocol_flock(self) -> None:
        lock = self.root / "gpu0.lock"
        previous = os.environ.get("VOLVE_P5_GPU_LOCK")
        os.environ["VOLVE_P5_GPU_LOCK"] = str(lock)
        try:
            with stage2.gpu_flock("cuda:0") as audit:
                self.assertTrue(audit["required"])
                self.assertTrue(audit["acquired"])
                self.assertIn("flock", audit["mechanism"])
                self.assertEqual(audit["timeout_seconds"], 900)
            self.assertTrue(lock.is_file())
        finally:
            if previous is None:
                os.environ.pop("VOLVE_P5_GPU_LOCK", None)
            else:
                os.environ["VOLVE_P5_GPU_LOCK"] = previous


class ReconstructionStage2ExecutionAndCollationTest(ReconstructionStage2Fixture):
    def test_real_adapter_pilot_and_preserved_skips_are_structured(self) -> None:
        cells = self.root / "cells"
        passed = stage2.run_cell(
            mode="strict",
            model_id="scipy_rbf_neighbors",
            cache_dir=self.cache,
            cell_root=cells,
            device="cpu",
        )
        self.assertEqual(passed["status"], "passed")
        self.assertEqual(passed["evidence_status"], "development_piloted")
        self.assertTrue(np.isfinite(passed["metrics"]["rmse"]))
        for model_id, code in (
            ("mpslib_snesim3d", "missing_legal_training_image"),
            ("tcnn_hashgrid_inr", "dependency_missing"),
        ):
            skipped = stage2.run_cell(
                mode="strict",
                model_id=model_id,
                cache_dir=self.cache,
                cell_root=cells,
                device="cpu",
            )
            self.assertEqual(skipped["status"], "skipped")
            self.assertEqual(skipped["reason"]["code"], code)

    def test_neural_candidate_cannot_publish_a_cpu_pilot(self) -> None:
        record = stage2.run_cell(
            mode="strict",
            model_id="siren_inr",
            cache_dir=self.cache,
            cell_root=self.root / "cpu-neural",
            device="cpu",
        )
        self.assertEqual(record["status"], "skipped")
        self.assertEqual(record["reason"]["code"], "gpu_required_by_stage2_protocol")
        self.assertIsNone(record["metrics"])

    def test_collation_requires_all_cells_and_keeps_leaderboards_separate(self) -> None:
        cells = self.root / "cells"
        template = stage2.run_cell(
            mode="strict",
            model_id="scipy_rbf_neighbors",
            cache_dir=self.cache,
            cell_root=cells,
            device="cpu",
        )
        for mode in stage2.MODES:
            task_id = p4.protocol(mode).task_id
            mode_audit = json.loads((self.cache / "cache_manifest.json").read_text())["modes"][mode]
            for index, model_id in enumerate(stage2.CANDIDATES):
                record = json.loads(json.dumps(template))
                record.update(
                    {
                        "model_id": model_id,
                        "task_id": task_id,
                        "lane": mode,
                        "evaluation_mode": mode,
                        "split_hash": mode_audit["split_hash"],
                        "mode_isolation": {
                            "input_whitelist": mode_audit["input_whitelist"],
                            "constraint_audit": mode_audit["constraint_audit"],
                            "strict_constraints_supplied": (
                                0 if mode == "strict" else None
                            ),
                        },
                    }
                )
                if model_id in stage2.PRESERVED_STAGE1_SKIPS:
                    record.update(
                        {
                            "status": "skipped",
                            "metrics": None,
                            "budget": None,
                            "updates": 0,
                            "wall_seconds": 0.0,
                            "reason": stage2.PRESERVED_STAGE1_SKIPS[model_id],
                        }
                    )
                else:
                    record["metrics"] = {
                        "rmse": 0.01 + index / 1000.0,
                        "mae": 0.008 + index / 1000.0,
                        "spectral_log_rmse": 0.1 + index / 100.0,
                        "valid_voxels": 24,
                    }
                    if model_id in stage2.GPU_CANDIDATES:
                        record["model_config"]["device"] = "cuda:0"
                        record["resources"]["gpu_lock"] = {
                            "required": True,
                            "acquired": True,
                            "mechanism": "external flock -w 900",
                            "timeout_seconds": 900,
                        }
                record["result_hash"] = stage2.hash_payload(
                    {key: value for key, value in record.items() if key != "result_hash"}
                )
                stage2.atomic_write_json(cells / mode / model_id / "status.json", record)
        output = self.root / "portable"
        summary = stage2.collate(cells, output)
        self.assertEqual(summary["expected_cells"], 20)
        self.assertEqual(summary["counts"], {"passed": 16, "skipped": 4, "failed": 0, "timeout": 0})
        strict = json.loads((output / "p5_stage2_leaderboard_strict.json").read_text())
        conditional = json.loads((output / "p5_stage2_leaderboard_conditional.json").read_text())
        self.assertEqual(strict["lane"], "strict")
        self.assertEqual(conditional["lane"], "conditional")
        self.assertNotEqual(strict["task_id"], conditional["task_id"])
        self.assertEqual(len(strict["entries"]), 8)
        self.assertEqual(len(conditional["entries"]), 8)


class ReconstructionStage2PortableEvidenceTest(unittest.TestCase):
    def test_portable_results_cover_all_cells_and_enforce_gpu_neural_lane(self) -> None:
        results_path = HERE / "p5_stage2_results.jsonl"
        summary_path = HERE / "p5_stage2_summary.json"
        rows = [json.loads(line) for line in results_path.read_text().splitlines()]
        summary = json.loads(summary_path.read_text())
        self.assertEqual(len(rows), 20)
        self.assertEqual(summary["attempted_cells"], 20)
        self.assertEqual(
            summary["counts"],
            {"passed": 16, "skipped": 4, "failed": 0, "timeout": 0},
        )
        self.assertEqual(summary["frozen_test_i_blocks_loaded"], [])
        for row in rows:
            with self.subTest(mode=row["lane"], model=row["model_id"]):
                stage2.validate_cell_record(row, row["lane"], row["model_id"])
                self.assertNotEqual((row.get("reason") or {}).get("code"), "pilot_exception")
                if row["status"] == "passed" and row["model_id"] in stage2.GPU_CANDIDATES:
                    self.assertEqual(row["model_config"]["device"], "cuda:0")
                    lock = row["resources"]["gpu_lock"]
                    self.assertEqual(lock["mechanism"], "external flock -w 900")
                    self.assertEqual(lock["timeout_seconds"], 900)
                if row["status"] == "passed" and row["model_id"] == "monai_basicunet3d":
                    determinism = row["determinism_audit"]
                    self.assertTrue(determinism["strict_requested"])
                    self.assertTrue(determinism["warn_only"])
                    self.assertIn("deterministic", determinism["warning"].lower())
                if row["lane"] == "strict":
                    self.assertEqual(row["mode_isolation"]["strict_constraints_supplied"], 0)

    def test_portable_mode_leaderboards_never_mix_tasks(self) -> None:
        strict = json.loads((HERE / "p5_stage2_leaderboard_strict.json").read_text())
        conditional = json.loads(
            (HERE / "p5_stage2_leaderboard_conditional.json").read_text()
        )
        self.assertEqual(strict["lane"], "strict")
        self.assertEqual(conditional["lane"], "conditional")
        self.assertNotEqual(strict["task_id"], conditional["task_id"])
        self.assertTrue(strict["not_frozen_test"])
        self.assertTrue(conditional["not_frozen_test"])


if __name__ == "__main__":
    unittest.main()
