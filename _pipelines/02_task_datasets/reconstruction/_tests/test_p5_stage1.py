"""Portable P5 reconstruction adapter and Stage-1 firewall tests."""
from __future__ import annotations

import inspect
import json
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

from ml_framework.model_discovery import discover_model  # noqa: E402

import p4_reconstruction as p4  # noqa: E402
import p5_stage1 as p5  # noqa: E402
from _models.reconstruction._p5_adapter import AdapterSkip  # noqa: E402


class P5SourceAndDiscoveryContractTest(unittest.TestCase):
    def test_source_lock_has_exact_ten_revision_locked_scratch_candidates(self):
        lock = p5.load_source_lock()
        self.assertEqual(tuple(lock["models"]), p5.STAGE1_MODELS)
        for model_name, record in lock["models"].items():
            with self.subTest(model=model_name):
                self.assertEqual(len(record["revision"]), 40)
                int(record["revision"], 16)
                self.assertTrue(record["upstream_url"].startswith("https://"))
                self.assertTrue(record["license"])
                self.assertFalse(record["weights"]["required"])
                self.assertIsNone(record["weights"]["sha256"])

    def test_portable_evidence_records_fno_real_smoke_and_no_test_access(self):
        evidence = json.loads(
            (HERE / "p5_stage1_results.json").read_text(encoding="utf-8")
        )
        self.assertEqual(evidence["data_policy"]["frozen_test_i_blocks_loaded"], [])
        self.assertEqual(evidence["counts"]["models_failed"], 0)
        fno = evidence["models"]["neuralop_fno3d"]
        self.assertEqual(fno["status"], "passed_both_modes")
        self.assertEqual(fno["distribution"], "neuraloperator==2.0.0")
        self.assertTrue(fno["backward_executed"])
        self.assertTrue(fno["finite_prediction"])
        self.assertEqual(fno["checkpoint_roundtrip_max_abs_error"], 0.0)

    def test_all_ten_adapters_are_dynamically_discoverable_without_central_imports(self):
        for model_name in p5.STAGE1_MODELS:
            with self.subTest(model=model_name):
                discovered = discover_model("reconstruction", model_name)
                self.assertEqual(discovered.model_id, model_name)
                self.assertIn(discovered.capabilities["batch_representation"], {
                    "point", "volume", "categorical_volume"
                })
                self.assertIn("trainable", discovered.capabilities)


class P5StrictConditionalFirewallTest(unittest.TestCase):
    def test_mode_specific_development_batches_never_share_task_or_features(self):
        strict_spec = p4.task_spec("strict")
        conditional_spec = p4.task_spec("conditional")
        self.assertNotEqual(strict_spec.task_id, conditional_spec.task_id)
        self.assertNotEqual(strict_spec.input_whitelist, conditional_spec.input_whitelist)
        strict = p5.synthetic_development_bundle("strict")
        conditional = p5.synthetic_development_bundle("conditional")
        strict_train, strict_validation = p5.make_batches(
            strict_spec, strict, "point", max_train_points=32, max_validation_points=16
        )
        conditional_train, _ = p5.make_batches(
            conditional_spec, conditional, "point", max_train_points=32, max_validation_points=16
        )
        self.assertEqual(strict_train.inputs["features"].shape[1], 6)
        self.assertEqual(conditional_train.inputs["features"].shape[1], 7)
        self.assertFalse(any("idw" in name for name in strict_train.metadata["feature_names"]))
        self.assertTrue(any("idw" in name for name in conditional_train.metadata["feature_names"]))
        self.assertEqual(strict.prepared.constraint_audit["constraints_supplied_to_model"], 0)
        self.assertEqual(strict_validation.metadata["frozen_test_i_blocks_loaded"], [])

    def test_volume_batches_obey_six_vs_seven_channel_whitelists(self):
        for mode, expected in (("strict", 6), ("conditional", 7)):
            with self.subTest(mode=mode):
                spec = p4.task_spec(mode)
                bundle = p5.synthetic_development_bundle(mode)
                train, validation = p5.make_batches(
                    spec, bundle, "volume", max_train_points=32, max_validation_points=16
                )
                self.assertEqual(train.inputs["volume"].shape[1], expected)
                self.assertEqual(validation.inputs["volume"].shape[1], expected)
                self.assertTrue(np.asarray(train.target_masks[spec.targets[0]]).any())
                self.assertEqual(train.metadata["frozen_test_i_blocks_loaded"], [])
                if mode == "strict":
                    self.assertEqual(train.metadata["constraint_count_supplied"], 0)

    def test_runner_public_api_has_no_test_loader_or_test_command(self):
        parameters = inspect.signature(p5.run_stage1).parameters
        self.assertNotIn("test_loader", parameters)
        self.assertNotIn("test_data", parameters)
        source = inspect.getsource(p5.main)
        self.assertNotIn('"test"', source)


class P5AdapterExecutionContractTest(unittest.TestCase):
    def test_mps_without_approved_training_image_is_structured_skip(self):
        discovered = discover_model("reconstruction", "mpslib_snesim3d")
        with self.assertRaises(AdapterSkip) as caught:
            discovered.build(
                p4.task_spec("strict"), n_features=6,
                training_image_provenance_approved=False,
            )
        self.assertEqual(caught.exception.code, "missing_legal_training_image")

    def test_scipy_rbf_synthetic_fit_loss_checkpoint_and_seed(self):
        mode = "strict"
        spec = p4.task_spec(mode)
        bundle = p5.synthetic_development_bundle(mode)
        batches = p5.make_batches(
            spec, bundle, "point", max_train_points=48, max_validation_points=24
        )
        discovered = discover_model("reconstruction", "scipy_rbf_neighbors")
        seed = 2693
        config = p5.model_config(
            "scipy_rbf_neighbors", spec, batches[0], device="cpu", seed=seed
        )
        with tempfile.TemporaryDirectory() as directory:
            result = p5.exercise_contract(
                discovered, spec, *batches, config=config,
                output_dir=Path(directory), root_seed=seed, device="cpu",
            )
        self.assertTrue(result["finite_prediction"])
        self.assertFalse(result["backward_executed"])
        self.assertEqual(result["checkpoint_roundtrip_max_abs_error"], 0.0)
        self.assertEqual(result["same_seed_max_abs_error"], 0.0)
        self.assertTrue(np.isfinite(result["validation_loss"]))


if __name__ == "__main__":
    unittest.main()
