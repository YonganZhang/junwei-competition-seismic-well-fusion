from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

DEVELOPMENT = importlib.import_module("_pipelines.02_task_datasets.sweetspot.p5.development_data")
LABEL_GATE = importlib.import_module("_pipelines.02_task_datasets.sweetspot.p5.label_gate")
MATRIX = importlib.import_module("_pipelines.02_task_datasets.sweetspot.p5.matrix")
RUNNER = importlib.import_module("_pipelines.02_task_datasets.sweetspot.p5.runner")
SOURCE_LOCK = importlib.import_module("_pipelines.02_task_datasets.sweetspot.p5.source_lock")
TARGET_SPECS = importlib.import_module("_pipelines.02_task_datasets.sweetspot.p5.target_specs")
load_development_batch = DEVELOPMENT.load_development_batch
evaluate_label_spec = LABEL_GATE.evaluate_label_spec
MODEL_ORDER, TARGET_ORDER, matrix_gate = MATRIX.MODEL_ORDER, MATRIX.TARGET_ORDER, MATRIX.matrix_gate
run_stage1 = RUNNER.run_stage1
load_source_lock = SOURCE_LOCK.load_source_lock
inspect_runtime = SOURCE_LOCK.inspect_runtime
TARGETS, build_task_spec = TARGET_SPECS.TARGETS, TARGET_SPECS.build_task_spec


TRACK = Path(__file__).resolve().parents[1]
INVENTORY = TRACK / "audit" / "data_availability.json"


def approved_fixture(*, note: str = "unit-test-only") -> dict:
    return {
        "schema_version": "sweetspot-label-spec/v1",
        "spec_version": "9.9.9-test",
        "status": "approved",
        "target_semantics": "geological",
        "output": {"type": "continuous_score", "classes": [], "units": "fraction", "probability_interpretation": None},
        "allowed_source_fields": [
            {"source": "layer1.well_logs_clean.clean", "field": "LFP_GR", "role": "inference_input"},
            {"source": "layer1.well_logs_clean.clean", "field": "LFP_PHIE", "role": "label_only"},
        ],
        "label_construction": {
            "formula": "unit-test fixture passthrough of an audited label-only field",
            "formula_field_refs": [{"source": "layer1.well_logs_clean.clean", "field": "LFP_PHIE"}],
            "thresholds": [], "thresholds_not_applicable_reason": "continuous unit-test fixture",
            "weights": [], "weights_not_applicable_reason": "single unit-test fixture field",
            "fit_domain": {"statistics_scope": "external_fixed", "population": "unit-test fixture only", "uses_test_statistics": False},
        },
        "time_window": {"definition": "static unit-test fixture", "start": None, "end": None, "timezone": None, "leakage_cutoff": "no future inputs"},
        "spatial_scale": {"support": "well_interval", "coordinate_system": "MD", "vertical_domain": "depth", "resolution": "one fixture interval", "alignment_tolerance": "zero in fixture"},
        "class_rules": {"positive": "finite fixture value above zero", "negative": "finite fixture value equal to zero", "unlabeled": "exclude non-finite fixture values"},
        "split_strategy": {"strategy": "well_holdout", "group_key": "well_family", "train_rule": "fixture train groups", "validation_rule": "fixture held-out development group", "test_rule": "separate unopened fixture group", "fit_statistics_scope": "train_only", "leakage_guards": ["well families do not cross splits"]},
        "inference_allowed_inputs": [{"source": "layer1.well_logs_clean.clean", "field": "LFP_GR"}],
        "metrics": [{"name": "mae", "aggregation": "macro by fixture well", "decision_threshold": None}],
        "approval": {"approved": True, "approved_by": "TEST FIXTURE ONLY", "approved_role": "unit test", "approved_at": "2000-01-01", "decision_record": note},
        "notes": note,
    }


class P5ContractTests(unittest.TestCase):
    def test_source_lock_matrix_and_modules_are_exactly_first_ten(self):
        lock = load_source_lock()
        self.assertEqual(tuple(lock), MODEL_ORDER)
        self.assertEqual(len(MODEL_ORDER), 10)
        self.assertEqual(len(TARGET_ORDER), 7)
        self.assertEqual(matrix_gate("temporal_fusion_transformer", "T1").reason_code, "matrix_not_applicable")
        self.assertEqual(matrix_gate("monai_unet3d", "T7").rating, "A")
        for entry in lock.values():
            runtime = inspect_runtime(entry)
            if runtime["available"]:
                self.assertTrue(runtime["version_allowed"], (entry["model_id"], runtime))

    def test_seven_targets_have_distinct_single_heads_and_t6_t7_never_alias(self):
        self.assertEqual(tuple(TARGETS), TARGET_ORDER)
        self.assertEqual(len({value.head_name for value in TARGETS.values()}), 7)
        self.assertNotEqual(TARGETS["T6"].head_name, TARGETS["T7"].head_name)
        self.assertNotEqual(TARGETS["T6"].slug, TARGETS["T7"].slug)

    def test_draft_contract_is_structured_skip_and_build_is_forbidden(self):
        gate = evaluate_label_spec("T6", TRACK / "label_spec.template.v1.yml", inventory_path=INVENTORY)
        self.assertFalse(gate.approved)
        self.assertIn("label_not_approved", gate.reason_codes)
        with self.assertRaises(PermissionError):
            build_task_spec("T6", gate)

    def test_approved_test_fixture_builds_one_head_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t6.yml"
            path.write_text(yaml.safe_dump(approved_fixture(), sort_keys=False), encoding="utf-8")
            gate = evaluate_label_spec("T6", path, inventory_path=INVENTORY)
            self.assertTrue(gate.approved, gate.errors)
            spec = build_task_spec("T6", gate)
            self.assertEqual(spec.targets, ("T6_POROSITY",))
            self.assertTrue(spec.metadata["single_target_head"])
            self.assertIn("layer1.well_logs_clean.clean.LFP_PHIE", spec.forbidden_inputs)
            self.assertNotIn("layer1.well_logs_clean.clean.LFP_PHIE", spec.input_whitelist)

    def test_all_current_cells_fail_closed_before_data_or_model_access(self):
        self.assertFalse(any("test" in name.lower() for name in inspect.signature(run_stage1).parameters))
        report = run_stage1(inventory_path=INVENTORY)
        self.assertEqual(len(report["results"]), 70)
        self.assertEqual(report["counts"]["PASS"], 0)
        self.assertEqual(report["counts"]["FAILED"], 0)
        self.assertTrue(all(item["status"] == "SKIP" for item in report["results"]))
        self.assertFalse(report["test_loader_api_present"])
        self.assertFalse(report["test_accessed"])
        self.assertFalse(report["labels_generated"])

    def test_same_approved_spec_cannot_define_t6_and_t7(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.yml"
            path.write_text(yaml.safe_dump(approved_fixture(), sort_keys=False), encoding="utf-8")
            report = run_stage1(
                model_ids=("xgboost",), target_ids=("T6", "T7"),
                label_specs={"T6": path, "T7": path}, inventory_path=INVENTORY,
            )
            self.assertEqual(report["counts"]["PASS"], 0)
            self.assertTrue(all(item["reason_code"] == "shared_label_spec_forbidden" for item in report["results"]))

    def test_approved_fixture_reaches_end_to_end_stage1_without_test(self):
        if importlib.util.find_spec("xgboost") is None:
            self.skipTest("xgboost is not in this shared interpreter")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "t6.yml"
            spec_path.write_text(yaml.safe_dump(approved_fixture(note="t6-unit-test-only"), sort_keys=False), encoding="utf-8")
            gate = evaluate_label_spec("T6", spec_path, inventory_path=INVENTORY)
            self.assertTrue(gate.approved, gate.errors)
            data = root / "development.npz"
            rng = np.random.default_rng(2693)
            np.savez(data, x=rng.normal(size=(24, 5)), y=rng.normal(size=24), mask=np.ones(24, dtype=bool), ids=np.asarray([f"d{i}" for i in range(24)]))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": "sweetspot-p5-development-batch/v1", "split": "development",
                "contains_test": False, "test_accessed": False, "target_id": "T6",
                "label_spec_sha256": gate.spec_sha256, "format": "npz", "data_file": data.name,
                "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "arrays": {"inputs": {"tabular": "x"}, "target": "y", "target_mask": "mask", "sample_ids": "ids"},
            }), encoding="utf-8")
            report = run_stage1(
                model_ids=("xgboost",), target_ids=("T6",), label_specs={"T6": spec_path},
                development_manifests={"T6": manifest}, inventory_path=INVENTORY,
            )
            self.assertEqual(report["counts"], {"PASS": 1, "SKIP": 0, "FAILED": 0})
            result = report["results"][0]
            self.assertTrue(result["synthetic_smoke"]["same_seed_replay"])
            self.assertFalse(result["test_accessed"])
            self.assertIsNone(result["scientific_metrics"])


class P5DevelopmentFirewallTests(unittest.TestCase):
    def test_content_addressed_development_npz_loads_without_test_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "development.npz"
            np.savez(data, x=np.ones((6, 3)), y=np.arange(6.0), mask=np.ones(6, dtype=bool), ids=np.asarray([f"d{i}" for i in range(6)]))
            spec_hash = "a" * 64
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": "sweetspot-p5-development-batch/v1", "split": "development",
                "contains_test": False, "test_accessed": False, "target_id": "T6",
                "label_spec_sha256": spec_hash, "format": "npz", "data_file": data.name,
                "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "arrays": {"inputs": {"tabular": "x"}, "target": "y", "target_mask": "mask", "sample_ids": "ids"},
            }), encoding="utf-8")
            batch = load_development_batch(manifest, target_id="T6", label_spec_sha256=spec_hash, limit=4)
            self.assertEqual(batch.inputs["tabular"].shape, (4, 3))
            self.assertEqual(batch.sample_ids, ("d0", "d1", "d2", "d3"))

    def test_manifest_that_attests_test_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"schema_version": "sweetspot-p5-development-batch/v1", "split": "development", "contains_test": True}), encoding="utf-8")
            with self.assertRaises(PermissionError):
                load_development_batch(path, target_id="T6", label_spec_sha256="a" * 64)


if __name__ == "__main__":
    unittest.main()
