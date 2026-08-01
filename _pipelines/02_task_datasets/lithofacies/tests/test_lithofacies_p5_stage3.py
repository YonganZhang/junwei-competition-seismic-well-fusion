"""Track-prefixed P5 Stage-3 roster, budget, firewall, and artifact tests."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import sys
import unittest
from pathlib import Path

import numpy as np


TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_track_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, TRACK_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename} as {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_pipeline_contract = _load_track_module(
    "lithofacies_pipeline_contract_stage3", "pipeline_contract.py"
)
sys.modules["pipeline_contract"] = _pipeline_contract
_p4_contract = _load_track_module("lithofacies_p4_contract_stage3", "p4_contract.py")
sys.modules["p4_contract"] = _p4_contract
_p5_stage1 = _load_track_module("lithofacies_p5_stage1_for_stage3", "p5_stage1.py")
sys.modules["p5_stage1"] = _p5_stage1
_p5_stage2 = _load_track_module(
    "lithofacies_p5_stage2_for_stage3", "lithofacies_p5_stage2.py"
)
sys.modules["lithofacies_p5_stage2"] = _p5_stage2
stage3 = _load_track_module("lithofacies_p5_stage3", "lithofacies_p5_stage3.py")


def _model_config(model_id: str, seed: int) -> dict:
    common = {
        "num_classes": 9,
        "well_log_shape": (26, 33),
        "seismic_shape": (3, 3, 33),
    }
    if model_id == "xgboost_multisoftprob_window":
        return {
            **common,
            "rounds": 40,
            "max_depth": 2,
            "eta": 0.2,
            "seed": seed,
        }
    if model_id == "catboost_multiclass_window":
        return {**common, "iterations": 40, "depth": 3, "seed": seed}
    return {**common, "nf": 8, "kernel_size": 31}


def _fake_cell(model_id: str, fold_id: int, repeat_id: int, *, status: str = "PASS") -> dict:
    seed = stage3.REPEAT_SEEDS[repeat_id]
    validation = stage3.DEVELOPMENT_FAMILIES[fold_id]
    train = [family for family in stage3.DEVELOPMENT_FAMILIES if family != validation]
    passing = status == "PASS"
    score = 0.10 + 0.01 * stage3.TOP3.index(model_id) + 0.005 * fold_id + 0.001 * repeat_id
    result = {
        "schema_version": stage3.CELL_SCHEMA,
        "track_id": "lithofacies",
        "task_id": stage3.TASK_ID,
        "lane": "P",
        "model_id": model_id,
        "fold_id": fold_id,
        "repeat_id": repeat_id,
        "seed": seed,
        "component_seeds": stage3.component_seeds(model_id, fold_id, repeat_id),
        "status": status,
        "reason": None if passing else {"code": "fixture_failure", "message": "fixture"},
        "source_revision": "fixture",
        "source_lock_sha256": "a" * 64,
        "split_hash": "b" * 64,
        "fold_partition_hash": f"{fold_id}" * 64,
        "train_groups": train,
        "validation_groups": [validation],
        "input_budget": {
            "context_positions": 33,
            "fold_train_samples_used": 20,
            "fold_validation_samples_used": 8,
            "budget_hash": stage3.stage2_budget_contract()["budget_hash"],
        },
        "loss_contract": stage3.locked_loss_contract(model_id),
        "hpo": False,
        "preprocessing": {
            "fit_scope": "fold_train_mother_families_only",
            "fit_families": sorted(train),
            "preprocessor_hash": "c" * 64,
            "class_weight_fit_scope": "fold_train_mother_families_only",
            "target_transform": "identity_class_id",
            "calibration": "not_applied_locked_stage2_configuration",
        },
        "frozen_test_accessed": False,
        "test_metrics_used": False,
        "rank_eligible": passing,
        "parameter_updates": 40 if model_id == "inceptiontime_window" and passing else 0,
        "wall_seconds": 1.0,
        "wall_limit_seconds": 600.0 if model_id == "inceptiontime_window" else 300.0,
        "peak_resources": {"process_peak_rss_kib": 1, "peak_vram_bytes": 0},
        "validation_metrics": {
            "fixed_schema_macro_f1": score,
            "supported_class_macro_f1": score + 0.05,
        } if passing else None,
        "model_config": _model_config(model_id, seed) if passing else None,
        "oof_prediction": None,
    }
    if model_id == "inceptiontime_window" and passing:
        result["optimizer"] = "AdamW(lr=0.001,weight_decay=0.0001)"
        result["tiny_gate"] = {"updates": 3}
    return result


def _all_fake_cells() -> list[dict]:
    return [
        _fake_cell(model_id, fold_id, repeat_id)
        for model_id in stage3.TOP3
        for fold_id in stage3.FOLD_IDS
        for repeat_id in range(len(stage3.REPEAT_SEEDS))
    ]


class LithofaciesStage3FrozenContractTests(unittest.TestCase):
    def test_module_test_roster_and_repeat_seeds_are_frozen(self) -> None:
        self.assertEqual(Path(stage3.__file__).name, "lithofacies_p5_stage3.py")
        self.assertEqual(Path(__file__).name, "test_lithofacies_p5_stage3.py")
        self.assertEqual(stage3.__name__, "lithofacies_p5_stage3")
        self.assertEqual(
            stage3.TOP3,
            (
                "xgboost_multisoftprob_window",
                "catboost_multiclass_window",
                "inceptiontime_window",
            ),
        )
        self.assertEqual(stage3.REPEAT_SEEDS, (1867973658, 2137841944, 3902865753))
        self.assertEqual(stage3.EXPECTED_CELL_COUNT, 36)
        self.assertEqual(len(stage3.expected_cell_keys()), 36)

    def test_stage2_budget_is_reused_without_hpo(self) -> None:
        budget = stage3.stage2_budget_contract()
        self.assertEqual(budget["well_shape"], [26, 33])
        self.assertEqual(budget["seismic_shape"], [3, 3, 33])
        self.assertEqual(budget["fold_train_sample_limit"], 320)
        self.assertEqual(budget["fold_validation_sample_limit"], 160)
        self.assertEqual(budget["batch_size"], 32)
        self.assertEqual(budget["neural_parameter_updates"], 40)
        self.assertEqual(budget["tiny_gate_updates_included"], 3)
        self.assertEqual(budget["xgboost"], {"rounds": 40, "max_depth": 2})
        self.assertEqual(budget["catboost"], {"iterations": 40, "depth": 3})
        self.assertEqual(budget["inceptiontime"], {"nf": 8, "kernel_size": 31})
        self.assertFalse(budget["hpo"])

    def test_archived_xgboost_runner_keeps_legacy_eta_explicit(self) -> None:
        lock = {
            "model_id": "xgboost_multisoftprob_window",
            "smoke_config": {"rounds": 8, "max_depth": 2, "seed": 2693},
        }
        config = _p5_stage2._stage2_model_config(
            lock,
            np.zeros((2, 26, 33), dtype=np.float32),
            np.zeros((2, 3, 3, 33), dtype=np.float32),
            model_seed=123,
        )
        self.assertEqual(config["rounds"], 40)
        self.assertEqual(config["max_depth"], 2)
        self.assertEqual(config["eta"], 0.2)
        self.assertEqual(config["seed"], 123)

    def test_cell_seed_lane_loss_budget_and_preprocessing_fail_closed(self) -> None:
        valid = _fake_cell(stage3.TOP3[0], 0, 0)
        stage3.validate_cell_result(valid)
        mutations = [
            ("seed", 2693, "repeat seed"),
            ("lane", "S", "roster|cross-lane"),
            ("loss_contract", "focal", "loss"),
            ("hpo", True, "HPO"),
        ]
        for key, value, message in mutations:
            with self.subTest(key=key):
                changed = copy.deepcopy(valid)
                changed[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    stage3.validate_cell_result(changed)
        changed = copy.deepcopy(valid)
        changed["preprocessing"]["fit_families"].append("15/9-19")
        with self.assertRaisesRegex(ValueError, "families|validation"):
            stage3.validate_cell_result(changed)
        changed = copy.deepcopy(valid)
        changed["model_config"]["rounds"] = 41
        with self.assertRaisesRegex(ValueError, "configuration"):
            stage3.validate_cell_result(changed)
        changed = copy.deepcopy(valid)
        changed["input_budget"]["budget_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "budget"):
            stage3.validate_cell_result(changed)
        changed = copy.deepcopy(valid)
        changed["input_budget"]["fold_train_samples_used"] = 321
        with self.assertRaisesRegex(ValueError, "sample budget"):
            stage3.validate_cell_result(changed)

    def test_split_and_frozen_test_firewall_fail_closed(self) -> None:
        valid = _fake_cell(stage3.TOP3[0], 0, 0)
        changed = copy.deepcopy(valid)
        changed["validation_groups"] = [stage3.TEST_FAMILY]
        with self.assertRaisesRegex((ValueError, RuntimeError), "validation|frozen"):
            stage3.validate_cell_result(changed)
        for key in ("frozen_test_accessed", "test_metrics_used"):
            with self.subTest(key=key):
                changed = copy.deepcopy(valid)
                changed[key] = True
                with self.assertRaisesRegex(RuntimeError, "frozen-test firewall"):
                    stage3.validate_cell_result(changed)

        cells = _all_fake_cells()
        changed_cells = copy.deepcopy(cells)
        changed_cells[0]["split_hash"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "split hashes"):
            stage3.validate_result_collection(changed_cells)
        changed_cells = copy.deepcopy(cells)
        changed_cells[0]["fold_partition_hash"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "partition hashes"):
            stage3.validate_result_collection(changed_cells)

    def test_duplicate_and_missing_cells_fail_closed(self) -> None:
        cells = _all_fake_cells()
        self.assertEqual(len(stage3.validate_result_collection(cells)), 36)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            stage3.validate_result_collection([*cells, copy.deepcopy(cells[0])])
        with self.assertRaisesRegex(ValueError, "roster mismatch"):
            stage3.validate_result_collection(cells[:-1])

    def test_completion_below_eighty_percent_is_not_rankable(self) -> None:
        cells = _all_fake_cells()
        for index in range(8):
            cells[index] = _fake_cell(
                cells[index]["model_id"], cells[index]["fold_id"], cells[index]["repeat_id"],
                status="FAIL",
            )
        board = stage3.build_leaderboard(cells, "b" * 64)
        self.assertLess(board["completion_rate"], 0.80)
        self.assertEqual(board["status"], "not_rankable")

    def test_source_has_no_temporary_split_or_frozen_test_entry(self) -> None:
        source = (TRACK_DIR / "lithofacies_p5_stage3.py").read_text(encoding="utf-8")
        self.assertNotIn("train_test_split", source)
        self.assertNotIn("random_split", source)
        self.assertNotIn("load_frozen_test", source)
        self.assertNotIn('"test.h5"', source)
        self.assertNotIn("'test.h5'", source)
        signature = inspect.signature(stage3.run_cells)
        self.assertEqual(
            tuple(signature.parameters),
            ("batch_file", "output", "model_ids", "fold_ids", "repeat_ids", "device"),
        )


class LithofaciesStage3CanonicalArtifactTests(unittest.TestCase):
    def test_canonical_results_summary_and_leaderboard(self) -> None:
        output = TRACK_DIR / "_outputs" / "p5_stage3"
        paths = {
            "results": output / stage3.RESULTS_FILENAME,
            "summary": output / stage3.SUMMARY_FILENAME,
            "leaderboard": output / stage3.LEADERBOARD_FILENAME,
            "oof": output / stage3.OOF_MANIFEST_FILENAME,
            "visualization": output / stage3.VISUALIZATION_MANIFEST_FILENAME,
        }
        for path in paths.values():
            self.assertTrue(path.is_file(), path)
        results = stage3.read_jsonl(paths["results"])
        self.assertEqual(len(results), 36)
        self.assertEqual(len(stage3.validate_result_collection(results)), 36)
        self.assertEqual({result["seed"] for result in results}, set(stage3.REPEAT_SEEDS))
        self.assertEqual({result["fold_id"] for result in results}, set(stage3.FOLD_IDS))
        self.assertEqual({result["model_id"] for result in results}, set(stage3.TOP3))
        self.assertTrue(all(result["lane"] == "P" for result in results))
        self.assertTrue(all(result["frozen_test_accessed"] is False for result in results))
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        board = json.loads(paths["leaderboard"].read_text(encoding="utf-8"))
        self.assertEqual(summary["expected_cells"], 36)
        self.assertGreaterEqual(summary["completion_rate"], 0.80)
        self.assertEqual(board["primary_metric"], "fixed_schema_macro_f1_mean")
        self.assertEqual(board["supported_class_metric_role"], "diagnostic_only")
        self.assertEqual(summary["results_sha256"], hashlib.sha256(paths["results"].read_bytes()).hexdigest())
        self.assertEqual(summary["leaderboard_sha256"], hashlib.sha256(paths["leaderboard"].read_bytes()).hexdigest())
        for path in paths.values():
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(".claude/worktrees", text)
            self.assertNotIn("/mnt/data/", text)

    def test_oof_and_visualization_manifests_are_development_only(self) -> None:
        output = TRACK_DIR / "_outputs" / "p5_stage3"
        oof = json.loads((output / stage3.OOF_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        visualization_path = output / stage3.VISUALIZATION_MANIFEST_FILENAME
        visualization = json.loads(visualization_path.read_text(encoding="utf-8"))
        self.assertFalse(oof["full_predictions_committed"])
        self.assertEqual(len(oof["coverage"]), 9)
        complete = [entry for entry in oof["coverage"] if entry["complete_oof_cover"]]
        incomplete = [entry for entry in oof["coverage"] if not entry["complete_oof_cover"]]
        self.assertEqual(len(complete), 6)
        self.assertEqual(
            {(entry["model_id"], entry["repeat_id"]) for entry in incomplete},
            {("catboost_multiclass_window", repeat_id) for repeat_id in range(3)},
        )
        self.assertTrue(all(len(entry["folds"]) == 3 for entry in incomplete))
        self.assertFalse(oof["frozen_test_accessed"])
        self.assertFalse(visualization["frozen_test_accessed"])
        figure_ids = {entry["figure_id"] for entry in visualization["figures"]}
        self.assertEqual(
            figure_ids,
            {
                "fixed9_confusion",
                "fixed9_per_class_pr_f1",
                "calibration_reliability",
                "fold_seed_matrix",
                "missing_modality_diagnostic",
                "continuous_depth_facies_track",
            },
        )
        depth = next(
            entry for entry in visualization["figures"]
            if entry["figure_id"] == "continuous_depth_facies_track"
        )
        self.assertEqual(depth["status"], "not_feasible")
        self.assertEqual(depth["finite_md_rows"], 0)
        for entry in visualization["figures"]:
            path = output / entry["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(entry["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
