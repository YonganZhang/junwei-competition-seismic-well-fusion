from __future__ import annotations

import inspect
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path

from _code.ml_framework.artifacts import ArtifactManifest, atomic_write_json, hash_payload
from _code.ml_framework.checkpoint import load_checkpoint, restore_rng_state, save_checkpoint
from _code.ml_framework.contracts import ModelBatch, ModelOutput, TaskSpec
from _code.ml_framework.cv import run_development_cv
from _code.ml_framework.hpo import HPOPlan, rank_trials, run_fixed_trials
from _code.ml_framework.lifecycle import ExperimentLifecycle, ExperimentState
from _code.ml_framework.model_discovery import discover_model
from _code.ml_framework.reduction import WeightedReducer, weighted_mean
from _code.ml_framework.run_layout import assert_visualization_is_read_only, create_run_layout
from _code.ml_framework.seeding import SeedTree, derive_seed, seed_everything
from _code.ml_framework.splits import build_group_folds, validate_manifest
from _code.ml_framework.trainer import StepResult, TrainerConfig, train_with_validation


def task_spec() -> TaskSpec:
    return TaskSpec(
        track_id="property",
        task_id="porosity_phif",
        task_type="regression",
        input_modalities=("seismic", "logs"),
        targets=("PHIF",),
        units={"PHIF": "fraction"},
        label_version="cpi-phif-v1",
        target_masks={"PHIF": "phif_valid"},
        group_keys=("well_family",),
        target_transform={"PHIF": "identity"},
        inverse_transform={"PHIF": "identity"},
        train_loss={"PHIF": "huber"},
        inference_transform={"PHIF": "identity"},
        threshold_policy={},
        calibration_policy={},
        primary_metrics=("mae",),
        input_whitelist=("seismic", "GR"),
        forbidden_inputs=("PHIF", "PHIE"),
    )


class ContractTests(unittest.TestCase):
    def test_task_spec_roundtrip_and_leakage_guard(self):
        spec = task_spec()
        self.assertEqual(TaskSpec.from_dict(spec.to_dict()), spec)
        payload = spec.to_dict()
        payload["input_whitelist"] = ["PHIF"]
        with self.assertRaisesRegex(ValueError, "forbidden"):
            TaskSpec.from_dict(payload)

    def test_batch_and_output_envelopes(self):
        batch = ModelBatch(
            inputs={"x": [1, 2]},
            targets={"y": [3, 4]},
            input_masks={},
            target_masks={"y": [True, True]},
            sample_ids=["a", "b"],
            groups={"well": ["w1", "w2"]},
            coordinates={},
        )
        self.assertEqual(len(batch.sample_ids), 2)
        self.assertEqual(ModelOutput(raw={"y": [0.1]}, transformed={"y": [0.1]}).raw["y"], [0.1])
        with self.assertRaises(ValueError):
            ModelBatch({"x": [1]}, None, {}, {"y": [True]}, ["a"], {"well": ["w"]}, {})

    def test_canonical_model_dynamic_discovery(self):
        module_name = "_models.property.contract_dummy"
        module = types.ModuleType(module_name)
        module.model_id = "contract_dummy"
        module.capabilities = lambda: {
            "task_types": ["regression"],
            "input_modalities": ["seismic"],
            "supports_missing_mask": True,
            "supports_uncertainty": False,
        }
        module.build_model = lambda spec, **config: (spec.task_id, config)
        sys.modules[module_name] = module
        try:
            discovered = discover_model("property", "contract_dummy")
            self.assertEqual(discovered.build(task_spec(), width=8), ("porosity_phif", {"width": 8}))
        finally:
            sys.modules.pop(module_name, None)


class SeedReducerTests(unittest.TestCase):
    def test_seed_tree_is_stable_and_role_separated(self):
        first = SeedTree(2693)
        second = SeedTree(2693)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertNotEqual(first.seed("split"), first.seed("model"))
        self.assertEqual(derive_seed(2693, "fold", 2), derive_seed(2693, "fold", 2))
        report = seed_everything(2693, include_torch=False)
        self.assertTrue(report.python_seeded)

    def test_weighted_reducer_is_not_batch_mean(self):
        reducer = WeightedReducer()
        reducer.update_mean(1.0, 1)
        reducer.update_mean(3.0, 9)
        self.assertAlmostEqual(reducer.mean, 2.8)
        self.assertAlmostEqual(weighted_mean([(1.0, 1), (3.0, 9)]), 2.8)
        with self.assertRaises(RuntimeError):
            _ = WeightedReducer().mean


class SplitCVTests(unittest.TestCase):
    def setUp(self):
        self.sample_ids = [f"s{i}" for i in range(12)]
        self.groups = [f"w{i // 2}" for i in range(12)]

    def test_group_split_locks_test_and_downgrades_honestly(self):
        manifest = build_group_folds(
            self.sample_ids,
            self.groups,
            group_key="well_family",
            test_groups=["w5"],
            requested_n_splits=5,
            max_splits_by_support=4,
            support_reason="only four label-supported families",
        )
        self.assertEqual(manifest.effective_n_splits, 4)
        self.assertIn("four label-supported", manifest.downgrade_reason)
        validate_manifest(manifest)
        self.assertTrue(set(manifest.test_sample_ids).isdisjoint(manifest.development_sample_ids))

    def test_cv_has_no_test_argument_and_enforces_oof(self):
        self.assertNotIn("test", inspect.signature(run_development_cv).parameters)
        manifest = build_group_folds(
            self.sample_ids, self.groups, group_key="well", test_groups=["w5"], requested_n_splits=5
        )
        with tempfile.TemporaryDirectory() as directory:
            def runner(fold):
                return {
                    "validation_sample_ids": fold.validation_sample_ids,
                    "metrics": {"score": 0.5 + 0.01 * fold.fold_id},
                    "valid_label_count": len(fold.validation_sample_ids),
                }

            summary = run_development_cv(manifest, runner, output_dir=Path(directory), primary_metric="score")
            self.assertEqual(summary["oof_sample_count"], len(manifest.development_sample_ids))


class LifecycleArtifactTests(unittest.TestCase):
    def test_single_use_test_firewall(self):
        life = ExperimentLifecycle("run-1")
        life.advance(ExperimentState.SPLIT_LOCKED, {"split_hash": "s"})
        life.advance(ExperimentState.SMOKE_PASSED, {"tests": "ok"})
        life.advance(ExperimentState.CV_COMPLETE, {"oof_hash": "o"})
        life.advance(ExperimentState.CONFIG_FROZEN, {"config_hash": "c"})
        life.advance(ExperimentState.REFIT_COMPLETE, {"checkpoint_hash": "k"})
        life.require_test_access(config_hash="c", checkpoint_hash="k", split_hash="s")
        life.advance(
            ExperimentState.TEST_CONSUMED,
            {"config_hash": "c", "checkpoint_hash": "k", "split_hash": "s"},
        )
        with self.assertRaises(RuntimeError):
            life.require_development_access()
        life.advance(ExperimentState.VERIFIED, {"reviewer": "independent"})

    def test_artifact_manifest_and_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = create_run_layout(Path(directory))
            atomic_write_json(root / "task_spec.json", task_spec().to_dict())
            manifest = ArtifactManifest("run-1", root)
            manifest.register("task_spec.json", role="task_spec")
            manifest.write()
            manifest.verify()
            atomic_write_json(root / "oof" / "predictions.json", {"x": [1]})
            atomic_write_json(root / "oof" / "metrics.json", {"mae": 1})
            assert_visualization_is_read_only(
                prediction_path=root / "oof" / "predictions.json",
                metrics_path=root / "oof" / "metrics.json",
            )

    def test_checkpoint_contains_resume_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pkl"
            random.seed(17)
            save_checkpoint(
                path,
                epoch=3,
                model_state={"weight": 1},
                optimizer_state={"lr": 0.1},
                scheduler_state={"step": 3},
                scaler_state=None,
                config_hash="config",
                split_hash="split",
                seed_report={"root_seed": 2693},
                environment={"python": "test"},
                include_torch_rng=False,
            )
            expected = random.random()
            loaded = load_checkpoint(path)
            restore_rng_state(loaded["rng_state"])
            self.assertEqual(random.random(), expected)
            self.assertEqual(loaded["epoch"], 3)


class HPOTests(unittest.TestCase):
    def test_fixed_baseline_hpo_does_not_need_optuna_or_test(self):
        self.assertNotIn("test", inspect.signature(run_fixed_trials).parameters)
        self.assertEqual(HPOPlan().top_configs, 3)
        with tempfile.TemporaryDirectory() as directory:
            results = run_fixed_trials(
                [{"lr": 0.1}, {"lr": 0.2}],
                lambda params, seed: {
                    "fold_scores": [1.0 - params["lr"], 0.8 - params["lr"]],
                    "guardrails": {"finite": 1.0},
                },
                root_seed=2693,
                output_dir=Path(directory),
            )
            self.assertEqual(rank_trials(results)[0].params["lr"], 0.1)
            payload = json.loads((Path(directory) / "trials.json").read_text())
            self.assertEqual(len(payload), 2)
            self.assertEqual(len(hash_payload(payload)), 64)


class TrainerTests(unittest.TestCase):
    def test_trainer_weights_by_valid_labels_and_early_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = []

            def checkpoint_writer(state, path):
                checkpoints.append(path.name)

            state = train_with_validation(
                train_step=lambda batch: StepResult(loss_sum=batch[0], valid_count=batch[1]),
                validation_step=lambda batch: StepResult(loss_sum=batch[0], valid_count=batch[1]),
                train_batches_fn=lambda: [(1.0, 1), (27.0, 9)],
                validation_batches_fn=lambda: [(2.0, 1), (18.0, 9)],
                config=TrainerConfig(max_epochs=5, min_epochs=2, patience=1),
                output_dir=Path(directory),
                checkpoint_writer=checkpoint_writer,
            )
            self.assertAlmostEqual(state.history[0]["train_loss"], 2.8)
            self.assertAlmostEqual(state.history[0]["validation_loss"], 2.0)
            self.assertTrue(state.stopped_early)
            self.assertIn("checkpoint_best.pkl", checkpoints)


if __name__ == "__main__":
    unittest.main()
