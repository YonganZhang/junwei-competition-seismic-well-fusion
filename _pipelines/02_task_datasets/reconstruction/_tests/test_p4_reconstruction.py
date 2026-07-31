"""Portable P4 reconstruction unit, contract and tiny-training tests."""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
sys.path.insert(0, str(HERE))

from ml_framework.artifacts import atomic_write_json  # noqa: E402
from ml_framework.lifecycle import ExperimentState  # noqa: E402

import p4_reconstruction as p4  # noqa: E402
import p4_visualize  # noqa: E402


def write_synthetic_unified_hdf5(root: Path, records: list[p4.PatchRecord]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for source_split in ("train", "test"):
        selected = [record for record in records if record.location.source_split == source_split]
        with h5py.File(root / f"{source_split}.h5", "w") as handle:
            handle.attrs["task"] = "reconstruction"
            handle.attrs["split"] = source_split
            handle.attrs["n_samples"] = len(selected)
            for index, record in enumerate(selected):
                group = handle.create_group(f"sample_{index:07d}")
                group.create_dataset("seismic_patch", data=record.seismic_patch)
                group.create_dataset("well_log_seq", data=record.well_log_seq)
                group.create_dataset("label", data=record.label)
                group.attrs["position"] = "{}"
                group.attrs["meta"] = json.dumps(
                    {
                        "patch_index_kji": [
                            record.location.k_block,
                            record.location.j_block,
                            record.location.i_block,
                        ],
                        "patch_start_kji": list(record.location.patch_start_kji),
                        "patch_shape_kji": list(record.location.patch_shape_kji),
                        "task": "reconstruction",
                        "split": source_split,
                    }
                )


class P4TaskAndSplitContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog, cls.records = p4.synthetic_catalog_and_records()

    def test_conditional_and_strict_tasks_are_independent_and_leakage_explicit(self):
        conditional = p4.task_spec("conditional")
        strict = p4.task_spec("strict")
        self.assertNotEqual(conditional.task_id, strict.task_id)
        self.assertNotEqual(conditional.label_version, strict.label_version)
        self.assertNotEqual(conditional.targets, strict.targets)
        self.assertNotEqual(conditional.primary_metrics, strict.primary_metrics)
        self.assertNotEqual(conditional.input_whitelist, strict.input_whitelist)
        self.assertEqual(conditional.target_transform[conditional.targets[0]], "identity")
        self.assertEqual(strict.inference_transform[strict.targets[0]], "identity")
        self.assertEqual(strict.train_loss[strict.targets[0]]["name"], "mse")
        self.assertEqual(strict.metric_directions[strict.primary_metrics[0]], "minimize")
        self.assertEqual(strict.metric_directions[p4.metric_names("strict")["r2"]], "maximize")
        self.assertEqual(
            strict.metric_directions[p4.metric_names("strict")["pearson_r"]], "maximize"
        )
        self.assertNotIn(strict.targets[0], strict.input_whitelist)
        self.assertNotIn("sparse_well_constraints", strict.input_modalities)
        self.assertFalse(any("idw" in name for name in strict.input_whitelist))
        self.assertIn("test_region_well_porosity", strict.forbidden_inputs)
        self.assertIsNone(p4.protocol("strict").idw_feature_name)
        p4.assert_feature_contract("strict", (*p4.SEISMIC_FEATURES, *p4.COORDINATE_FEATURES))
        with self.assertRaisesRegex(ValueError, "forbidden"):
            p4.assert_feature_contract("strict", ("test_region_well_porosity",))

        strict_manifest = p4.build_spatial_manifest("strict", self.catalog)
        strict_development = [
            record
            for record in self.records
            if record.i_block in p4.protocol("strict").development_i_blocks
        ]
        strict_fold = p4.prepare_fold(
            "strict", strict_manifest.folds[0], strict_development
        )
        self.assertEqual(strict_fold.constraint_audit["constraints_supplied_to_model"], 0)
        self.assertTrue(
            strict_fold.constraint_audit["strict_reference_derived_well_values_excluded"]
        )

    def test_contiguous_test_is_frozen_before_buffered_five_fold_oof(self):
        conditional_manifest = p4.build_spatial_manifest("conditional", self.catalog)
        strict_manifest = p4.build_spatial_manifest("strict", self.catalog)
        self.assertNotEqual(conditional_manifest.test_groups, strict_manifest.test_groups)
        self.assertTrue(
            set(conditional_manifest.test_sample_ids).isdisjoint(
                strict_manifest.test_sample_ids
            )
        )
        for mode in p4.MODES:
            with self.subTest(mode=mode):
                manifest = p4.build_spatial_manifest(mode, self.catalog)
                self.assertEqual(manifest.requested_n_splits, 5)
                self.assertEqual(manifest.effective_n_splits, 5)
                self.assertTrue(manifest.metadata["test_i_blocks_contiguous"])
                self.assertEqual(manifest.metadata["freeze_order"], "continuous test I-blocks frozen before development CV")
                seen = [sample_id for fold in manifest.folds for sample_id in fold.validation_sample_ids]
                self.assertEqual(sorted(seen), sorted(manifest.development_sample_ids))
                self.assertEqual(len(seen), len(set(seen)))
                for fold in manifest.folds:
                    self.assertEqual(fold.purge["buffer_blocks"], 1)
                    self.assertGreater(fold.support["effective_train_patches"], 0)
                p4.validate_buffered_manifest(manifest, self.catalog)

    def test_requested_five_folds_downgrades_honestly_when_groups_are_fewer(self):
        reduced_catalog = [item for item in self.catalog if item.k_block < 4]
        manifest = p4.build_spatial_manifest(
            "strict", reduced_catalog, requested_n_splits=5
        )
        self.assertEqual(manifest.requested_n_splits, 5)
        self.assertEqual(manifest.effective_n_splits, 4)
        self.assertIn("only 4 independent", manifest.downgrade_reason)
        p4.validate_buffered_manifest(manifest, reduced_catalog)

    def test_fold_preprocessing_and_well_constraints_are_train_only(self):
        manifest = p4.build_spatial_manifest("conditional", self.catalog)
        development = [record for record in self.records if record.i_block in p4.protocol("conditional").development_i_blocks]
        # K=6 carries the only conditional-development constraint.  Its
        # validation fold must have zero supplied constraints and use the
        # explicitly audited neutral fallback.
        held_constraint_fold = next(
            fold for fold in manifest.folds if "dev_k06" in fold.validation_groups
        )
        prepared = p4.prepare_fold("conditional", held_constraint_fold, development)
        self.assertEqual(prepared.constraint_audit["fit_constraint_count"], 0)
        self.assertEqual(prepared.constraint_audit["validation_constraints_used"], 0)
        self.assertEqual(prepared.constraint_audit["zero_constraint_fallback"], "fold_train_target_mean")
        self.assertEqual(
            prepared.preprocess_report["effective_train_sample_ids"],
            held_constraint_fold.purge["effective_train_sample_ids"],
        )
        np.testing.assert_allclose(prepared.train_features.mean(axis=0), 0.0, atol=1e-7)
        self.assertLessEqual(prepared.preprocess_report["roundtrip_max_abs_error"], 1e-10)

    def test_empty_active_patch_keeps_split_identity_but_adds_no_voxels(self):
        record = self.records[0]
        empty_patch = record.seismic_patch.copy()
        empty_patch[8] = 0.0
        empty = p4.PatchRecord(record.location, empty_patch, record.label, record.well_log_seq)
        cells = p4.flatten_records([empty, self.records[1]])
        self.assertNotIn(empty.sample_id, set(cells.sample_ids.tolist()))
        self.assertGreater(cells.target.size, 0)
        self.assertGreaterEqual(cells.volume_shape_kji[0], empty.location.patch_shape_kji[0])

    def test_hpo_plan_and_fixed_baselines_are_minimize_and_optuna_free(self):
        self.assertEqual(p4.hpo_plan().direction, "minimize")
        self.assertEqual([item["model"] for item in p4.fixed_baseline_configs()], list(p4.MODEL_NAMES))
        with tempfile.TemporaryDirectory() as directory:
            results = p4.run_fixed_baseline_plan(
                output_dir=Path(directory),
                objective=lambda config, seed: {
                    "fold_scores": [float(len(config["model"])), float(len(config["model"]) + 1)]
                },
            )
            self.assertEqual(len(results), 3)
            self.assertTrue(all(item.metric_direction == "minimize" for item in results))
            self.assertTrue(all(item.worst == max(item.fold_scores) for item in results))

    def test_machine_readable_feasibility_report_keeps_scientific_limits(self):
        report = p4.feasibility_report(self.catalog, self.records)
        claims = {item["claim"] for item in report["not_feasible"]}
        self.assertIn("new blind cross-field generalization", claims)
        self.assertIn("cross-well generalization", claims)
        self.assertIn("fully constrained five-fold conditional IDW CV", claims)
        self.assertIn("strict P4 IDW from independent measured well porosity", claims)


class P4TrainingCheckpointAndVisualizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog, cls.records = p4.synthetic_catalog_and_records()

    def test_all_three_existing_models_complete_tiny_training_and_full_checkpoint(self):
        for model_name in p4.MODEL_NAMES:
            with self.subTest(model=model_name), tempfile.TemporaryDirectory() as directory:
                result = p4.tiny_smoke(Path(directory), mode="strict", model_name=model_name)
                self.assertEqual(result["epochs"], 3)
                self.assertTrue(result["finite_prediction"])
                checkpoint = p4.load_checkpoint(Path(result["checkpoint"]))
                for key in (
                    "model_state",
                    "optimizer_state",
                    "scheduler_state",
                    "scaler_state",
                    "trainer_state",
                    "seed_report",
                    "environment",
                ):
                    self.assertIn(key, checkpoint)
                self.assertEqual(checkpoint["extra"]["model"], model_name)

    def test_tiny_ridge_training_reduces_development_loss(self):
        active = p4.protocol("strict")
        development = [record for record in self.records if record.i_block in active.development_i_blocks]
        manifest = p4.build_spatial_manifest("strict", self.catalog)
        prepared = p4.prepare_fold("strict", manifest.folds[0], development)
        with tempfile.TemporaryDirectory() as directory:
            _, state, _ = p4.train_model(
                mode="strict",
                model_name="ridge_linear",
                prepared=prepared,
                output_dir=Path(directory),
                split_hash=manifest.stable_hash(),
                epochs=20,
                learning_rate=0.01,
                ridge_alpha=0.1,
                root_seed=2693,
            )
        self.assertLess(state.history[-1]["train_loss"], state.history[0]["train_loss"])
        self.assertTrue(all(math.isfinite(item["validation_loss"]) for item in state.history))

    def test_tiny_mlp_seed_reproduces_checkpoint_model_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = p4.tiny_smoke(
                root / "first", mode="strict", model_name="reconstruction_tiny_mlp"
            )
            second = p4.tiny_smoke(
                root / "second", mode="strict", model_name="reconstruction_tiny_mlp"
            )
            first_state = p4.load_checkpoint(Path(first["checkpoint"]))["model_state"]
            second_state = p4.load_checkpoint(Path(second["checkpoint"]))["model_state"]
            self.assertEqual(set(first_state), set(second_state))
            for name in first_state:
                if isinstance(first_state[name], np.ndarray):
                    np.testing.assert_array_equal(first_state[name], second_state[name])
                else:
                    self.assertEqual(first_state[name], second_state[name])

    def test_both_modes_complete_separate_pooled_oof_cv(self):
        for mode in p4.MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                active = p4.protocol(mode)
                development = [
                    record for record in self.records if record.i_block in active.development_i_blocks
                ]
                root = Path(directory) / mode
                summary = p4.run_development_baseline_cv(
                    mode=mode,
                    model_name="ridge_linear",
                    run_root=root,
                    catalog=self.catalog,
                    development_records=development,
                    epochs=2,
                    learning_rate=0.01,
                    ridge_alpha=0.1,
                    root_seed=2693,
                )
                self.assertEqual(summary["metric_direction"], "minimize")
                expected_manifest = p4.build_spatial_manifest(mode, self.catalog)
                self.assertEqual(summary["oof_sample_count"], len(expected_manifest.development_sample_ids))
                primary = p4.task_spec(mode).primary_metrics[0]
                self.assertTrue(math.isfinite(summary["pooled_oof_metrics"][primary]))
                self.assertEqual(
                    json.loads((root / "lifecycle.json").read_text())["state"],
                    ExperimentState.CV_COMPLETE.value,
                )
                self.assertTrue((root / "oof" / "predictions.npz").is_file())
                self.assertTrue((root / "manifest.json").is_file())

    def test_artifact_manifest_and_single_test_consumption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, life = p4.initialize_run(mode="strict", run_root=root, catalog=self.catalog)
            life.advance(ExperimentState.SMOKE_PASSED, {"tests": "ok"})
            life.advance(ExperimentState.CV_COMPLETE, {"oof_hash": "o"})
            life.advance(ExperimentState.CONFIG_FROZEN, {"config_hash": "c"})
            life.advance(ExperimentState.REFIT_COMPLETE, {"checkpoint_hash": "k"})
            p4._write_lifecycle(root, life)
            consumed = p4.consume_frozen_test_once(
                run_root=root,
                config_hash="c",
                checkpoint_hash="k",
                split_hash=manifest.stable_hash(),
            )
            self.assertEqual(consumed.state, ExperimentState.TEST_CONSUMED)
            with self.assertRaises(RuntimeError):
                p4.consume_frozen_test_once(
                    run_root=root,
                    config_hash="c",
                    checkpoint_hash="k",
                    split_hash=manifest.stable_hash(),
                )
            artifact_path = p4.refresh_artifact_manifest(root)
            self.assertTrue(artifact_path.is_file())
            self.assertIn("task_spec.json", json.loads(artifact_path.read_text())["artifacts"])

    def test_strict_end_to_end_refit_and_frozen_test_is_single_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            run_root = root / "run"
            write_synthetic_unified_hdf5(data_dir, self.records)
            catalog = p4.scan_patch_catalog(data_dir)
            development = p4.load_patch_records(
                p4.protocol("strict").development_i_blocks, data_dir
            )
            p4.run_development_baseline_cv(
                mode="strict",
                model_name="ridge_linear",
                run_root=run_root,
                catalog=catalog,
                development_records=development,
                epochs=1,
                learning_rate=0.01,
                ridge_alpha=0.1,
                root_seed=2693,
            )
            refit = p4.freeze_and_refit(
                mode="strict",
                model_name="ridge_linear",
                run_root=run_root,
                development_records=development,
                epochs=1,
                learning_rate=0.01,
                ridge_alpha=0.1,
                root_seed=2693,
            )
            self.assertTrue(math.isfinite(float(refit["epochs"])))
            result = p4.run_frozen_test_once(
                mode="strict", run_root=run_root, data_dir=data_dir
            )
            self.assertFalse(
                result["constraint_audit"]["strict_test_target_or_future_feature_used"]
            )
            self.assertEqual(result["constraint_audit"]["test_constraints_used"], 0)
            self.assertEqual(result["constraint_audit"]["development_constraints_used"], 0)
            self.assertFalse(
                result["constraint_audit"]["strict_reference_derived_well_value_used"]
            )
            self.assertTrue((run_root / "frozen_test" / "predictions.npz").is_file())
            self.assertEqual(
                json.loads((run_root / "lifecycle.json").read_text())["state"],
                ExperimentState.TEST_CONSUMED.value,
            )
            with self.assertRaises(RuntimeError):
                p4.run_frozen_test_once(mode="strict", run_root=run_root, data_dir=data_dir)

    def test_visualizer_reads_archives_only_and_never_mixes_modes(self):
        active = p4.protocol("strict")
        test_records = [record for record in self.records if record.i_block in active.test_i_blocks]
        cells = p4.flatten_records(test_records)
        prediction = cells.target + 0.005 * np.sin(np.arange(cells.target.size))
        metrics = p4.regression_metrics(
            "strict",
            cells.target,
            prediction,
            indices_kji=cells.indices_kji,
            volume_shape_kji=cells.volume_shape_kji,
            train_range=(float(cells.target.min()), float(cells.target.max())),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_path = p4.save_prediction_archive(
                root / "predictions.npz", mode="strict", cells=cells, prediction=prediction
            )
            metrics_path = atomic_write_json(root / "metrics.json", metrics)
            output_path = root / "strict_diagnostics.png"
            sidecar = p4_visualize.render_archived_visualization(
                prediction_path=prediction_path,
                metrics_path=metrics_path,
                output_path=output_path,
            )
            self.assertTrue(output_path.is_file())
            self.assertEqual(sidecar["evaluation_mode"], "strict")
            self.assertEqual(len(sidecar["panels"]), 6)
            mismatched = dict(metrics)
            mismatched["evaluation_mode"] = "conditional"
            bad_metrics = atomic_write_json(root / "bad_metrics.json", mismatched)
            with self.assertRaisesRegex(ValueError, "different mode/task"):
                p4_visualize.load_archived_prediction(prediction_path, bad_metrics)


if __name__ == "__main__":
    unittest.main()
