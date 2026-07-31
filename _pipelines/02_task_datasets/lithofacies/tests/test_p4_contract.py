from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from _code.ml_framework.checkpoint import load_checkpoint  # noqa: E402
from _code.ml_framework.lifecycle import ExperimentState  # noqa: E402
from p4_contract import (  # noqa: E402
    CLASS_NAMES,
    DEVELOPMENT_FAMILIES,
    EFFECTIVE_N_SPLITS,
    LOG_CHANNELS,
    TEST_FAMILY,
    apply_fold_preprocessor,
    build_lithofacies_split_manifest,
    classification_metrics_from_logits,
    fit_fold_preprocessor,
    lithofacies_hpo_plan,
    lithofacies_task_spec,
    model_output_from_logits,
    prediction_records,
    samples_to_model_batch,
    softmax_probabilities,
)
from p4_runner import (  # noqa: E402
    DEFAULT_CONFIG,
    _load_lifecycle,
    freeze_configuration,
    load_development_samples,
    prepare_run,
    refit_development,
    run_cv,
    run_frozen_test,
    run_real_smoke,
    tiny_overfit_probe,
)
from visualize_p4 import render_archived_visualizations  # noqa: E402


HAS_TORCH = importlib.util.find_spec("torch") is not None


FAMILY_LABELS = {
    "15/9-19": tuple(range(7)),
    "15/9-F-14": (1, 2, 3, 5, 6, 7),
    "15/9-F-15": tuple(range(9)),
    "15/9-F-4": (0, 1, 2, 3, 5, 6),
    "15/9-F-5": tuple(range(7)),
}


def synthetic_samples() -> list[dict]:
    samples: list[dict] = []
    family_offset = {family: index * 10.0 for index, family in enumerate(FAMILY_LABELS)}
    for family, labels in FAMILY_LABELS.items():
        well = family if family != "15/9-19" else "15/9-19 A"
        partition = "test" if family == TEST_FAMILY else ("guard" if family == "15/9-F-4" else "train")
        for index, label in enumerate(labels):
            center = 3000.0 + family_offset[family] + index * 2.0
            log_values = np.full(
                (len(LOG_CHANNELS), 4), family_offset[family] + label / 10.0, dtype=np.float32
            )
            masks = np.ones_like(log_values, dtype=np.float32)
            samples.append(
                {
                    "well_log_seq": np.concatenate((log_values, masks), axis=0),
                    "seismic_patch": np.full((2, 2, 4), label / 10.0, dtype=np.float32),
                    "label": np.int64(label),
                    "position": {
                        "inline": 100 + index,
                        "crossline": 200 + index,
                        "time_ms": 2000.0 + family_offset[family] + index,
                        "well_name": well,
                        "center_md_m": center,
                    },
                    "meta": {
                        "pipeline_version": "gm09_multimodal_v1",
                        "partition": partition,
                        "family_id": family,
                        "normalization_stats": {
                            "logs": {
                                channel: {
                                    "method": "zscore",
                                    "mean": 0.0,
                                    "std": 1.0,
                                    "vmin": None,
                                    "vmax": None,
                                }
                                for channel in LOG_CHANNELS
                            },
                            "seismic": {
                                "method": "zscore",
                                "mean": 0.0,
                                "std": 1.0,
                                "vmin": None,
                                "vmax": None,
                            },
                        },
                        "label_trace": {
                            "source": "GM09",
                            "curve_type": "GENETIC FACIES",
                            "class_name": CLASS_NAMES[label],
                            "member": f"{well}/Facies.xlsx",
                            "excel_row": index + 2,
                            "top_md_m": center - 1.0,
                            "base_md_m": center + 1.0,
                        },
                    },
                }
            )
    return samples


def _write_split(path: Path, samples: list[dict], split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["task"] = "lithofacies"
        handle.attrs["split"] = split
        handle.attrs["n_samples"] = len(samples)
        for index, sample in enumerate(samples):
            group = handle.create_group(f"sample_{index:07d}")
            group.create_dataset("seismic_patch", data=sample["seismic_patch"])
            group.create_dataset("well_log_seq", data=sample["well_log_seq"])
            group.create_dataset("label", data=sample["label"])
            group.attrs["position"] = json.dumps(sample["position"])
            group.attrs["meta"] = json.dumps(sample["meta"])


class P4PureContractTests(unittest.TestCase):
    def test_task_spec_freezes_nine_classes_and_declares_softmax(self) -> None:
        spec = lithofacies_task_spec()
        self.assertEqual(spec.track_id, "lithofacies")
        self.assertEqual(spec.task_type, "multiclass")
        self.assertEqual(tuple(spec.metadata["class_names"]), CLASS_NAMES)
        self.assertEqual(spec.metadata["class_count"], 9)
        self.assertEqual(spec.inference_transform["genetic_facies"], "softmax_then_argmax")
        self.assertEqual(tuple(spec.primary_metrics), ("fixed_schema_macro_f1",))
        self.assertEqual(
            tuple(spec.guardrail_metrics), ("worst_family_fixed_schema_macro_f1",)
        )
        self.assertIn("supported_class_macro_f1", spec.secondary_metrics)
        self.assertIn("Litho Class", spec.forbidden_inputs)
        self.assertNotIn("Litho Class", spec.input_whitelist)

    def test_f5_is_frozen_and_five_requested_folds_downgrade_to_logo4(self) -> None:
        manifest = build_lithofacies_split_manifest(synthetic_samples())
        self.assertEqual(manifest.requested_n_splits, 5)
        self.assertEqual(manifest.effective_n_splits, EFFECTIVE_N_SPLITS)
        self.assertEqual(manifest.test_groups, (TEST_FAMILY,))
        self.assertEqual(set(manifest.development_groups), set(DEVELOPMENT_FAMILIES))
        self.assertEqual(len(manifest.folds), 4)
        self.assertEqual(
            sorted(sample_id for fold in manifest.folds for sample_id in fold.validation_sample_ids),
            sorted(manifest.development_sample_ids),
        )
        for fold in manifest.folds:
            self.assertEqual(len(fold.validation_groups), 1)
            self.assertEqual(len(fold.support["train_class_support"]), 9)
            self.assertEqual(len(fold.support["validation_class_support"]), 9)

    def test_f15_holdout_reports_unseen_class_and_zero_weight(self) -> None:
        samples = synthetic_samples()
        manifest = build_lithofacies_split_manifest(samples)
        lookup = {
            "|".join(
                str(value)
                for value in (
                    sample["position"]["well_name"],
                    sample["position"]["center_md_m"],
                    sample["position"]["time_ms"],
                    sample["meta"]["label_trace"]["member"],
                    sample["meta"]["label_trace"]["excel_row"],
                )
            ): sample
            for sample in samples
        }
        fold = next(fold for fold in manifest.folds if fold.validation_groups == ("15/9-F-15",))
        self.assertEqual(fold.support["train_missing_class_ids"], [8])
        train = [lookup[sample_id] for sample_id in fold.train_sample_ids]
        validation = [lookup[sample_id] for sample_id in fold.validation_sample_ids]
        preprocessor = fit_fold_preprocessor(train)
        self.assertNotIn("15/9-F-15", preprocessor.fit_families)
        self.assertEqual(preprocessor.class_support[8], 0)
        self.assertEqual(preprocessor.class_weights[8], 0.0)
        transformed = apply_fold_preprocessor(validation, preprocessor)
        self.assertTrue(np.isfinite(transformed[0]["well_log_seq"]).all())
        self.assertTrue(np.isfinite(transformed[0]["seismic_patch"]).all())

    def test_logits_metrics_keep_fixed_and_supported_views(self) -> None:
        labels = np.asarray([0, 1, 1, 2], dtype=np.int64)
        logits = np.asarray(
            [[4, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 0],
             [0, 4, 0, 0, 0, 0, 0, 0, 0], [0, 0, 4, 0, 0, 0, 0, 0, 0]],
            dtype=np.float64,
        )
        output = softmax_probabilities(model_output_from_logits(logits))
        np.testing.assert_allclose(np.asarray(output.transformed["genetic_facies"]).sum(axis=1), 1.0)
        metrics = classification_metrics_from_logits(labels, logits)
        self.assertEqual(len(metrics["per_class"]), 9)
        self.assertEqual(metrics["supported_class_macro_f1"], 1.0)
        self.assertLess(metrics["fixed_schema_macro_f1"], 1.0)
        self.assertTrue(np.isfinite(metrics["negative_log_likelihood"]))
        self.assertTrue(np.isfinite(metrics["expected_calibration_error"]))

    def test_hpo_plan_is_development_only_and_maximizes_macro_f1(self) -> None:
        plan = lithofacies_hpo_plan()
        self.assertEqual(plan["direction"], "maximize")
        self.assertEqual(plan["primary_metric"], "fixed_schema_macro_f1")
        self.assertEqual(plan["supported_class_metric_role"], "diagnostic_only")
        self.assertEqual(plan["pruner"], "nop")
        self.assertEqual(plan["test_access"], "forbidden")
        self.assertEqual(plan["top_configs"], 3)
        self.assertEqual(plan["confirm_seeds"], 3)

    def test_cv_has_no_test_loader_and_run_id_cannot_be_reset(self) -> None:
        signature = inspect.signature(run_cv)
        self.assertNotIn("test", signature.parameters)
        self.assertNotIn("load_frozen_test_samples", inspect.getsource(run_cv))
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            (run / "lifecycle.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "never reset"):
                prepare_run(run, config=DEFAULT_CONFIG, dataset_root=Path(directory) / "missing")

    def test_visualizer_reads_archives_without_rewriting_them(self) -> None:
        samples = synthetic_samples()[:12]
        logits = np.eye(9, dtype=np.float64)[[int(sample["label"]) for sample in samples]] * 4.0
        metrics = classification_metrics_from_logits([int(sample["label"]) for sample in samples], logits)
        records = prediction_records(samples, logits)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_path = root / "predictions.json"
            metrics_path = root / "metrics.json"
            prediction_path.write_text(json.dumps({"records": records}), encoding="utf-8")
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            before = (prediction_path.read_bytes(), metrics_path.read_bytes())
            report = render_archived_visualizations(
                prediction_path=prediction_path,
                metrics_path=metrics_path,
                output_dir=root / "figures",
            )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(before, (prediction_path.read_bytes(), metrics_path.read_bytes()))
            self.assertEqual(len(report["figures"]), 4)

    def test_depth_track_refuses_interval_midpoint_fallback(self) -> None:
        samples = synthetic_samples()[:4]
        for sample in samples:
            sample["position"].pop("center_md_m")
        logits = np.zeros((len(samples), 9), dtype=np.float64)
        metrics = classification_metrics_from_logits([int(sample["label"]) for sample in samples], logits)
        records = prediction_records(samples, logits)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_path = root / "predictions.json"
            metrics_path = root / "metrics.json"
            prediction_path.write_text(json.dumps({"records": records}), encoding="utf-8")
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            report = render_archived_visualizations(
                prediction_path=prediction_path,
                metrics_path=metrics_path,
                output_dir=root / "figures",
            )
            self.assertEqual(report["status"], "PARTIAL_NOT_FEASIBLE")
            self.assertIn("depth_facies_track", report["not_feasible"])


@unittest.skipUnless(HAS_TORCH, "PyTorch is optional for portable contract tests")
class P4TorchContractTests(unittest.TestCase):
    def test_strict_batch_adapter_existing_models_and_ce_step(self) -> None:
        import torch

        samples = synthetic_samples()[:4]
        batch = samples_to_model_batch(samples)
        self.assertEqual(tuple(batch.inputs["well_log_seq"].shape), (4, 26, 4))
        self.assertEqual(tuple(batch.inputs["seismic_patch"].shape), (4, 2, 2, 4))
        for model_name in (
            "multimodal_mlp",
            "lithofacies_concat_linear",
            "lithofacies_late_fusion",
        ):
            from _code.ml_framework.model_registry import get_model

            model = get_model(
                model_name,
                models_package="models",
                num_classes=9,
                well_log_shape=(26, 4),
                seismic_shape=(2, 2, 4),
                hidden_size=8,
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            logits = model(batch.inputs["well_log_seq"], batch.inputs["seismic_patch"])
            output = model_output_from_logits(logits)
            loss = torch.nn.functional.cross_entropy(
                output.raw["genetic_facies"], batch.targets["genetic_facies"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            self.assertTrue(torch.isfinite(logits).all())

    def test_tiny_overfit_reduces_loss(self) -> None:
        samples = synthetic_samples()[:7]
        report = tiny_overfit_probe(samples, steps=20)
        self.assertEqual(report["status"], "PASS")
        self.assertLess(report["final_loss"], report["initial_loss"])

    def test_synthetic_full_lifecycle_oof_checkpoint_and_single_test(self) -> None:
        samples = synthetic_samples()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            run = root / "run"
            _write_split(
                data / "train.h5",
                [sample for sample in samples if sample["meta"]["family_id"] != TEST_FAMILY],
                "train",
            )
            _write_split(
                data / "test.h5",
                [sample for sample in samples if sample["meta"]["family_id"] == TEST_FAMILY],
                "test",
            )
            config = {
                **DEFAULT_CONFIG,
                "model_id": "lithofacies_concat_linear",
                "max_epochs": 2,
                "min_epochs": 1,
                "patience": 2,
                "batch_size": 16,
            }
            prepared = prepare_run(run, config=config, dataset_root=data)
            self.assertEqual(prepared["effective_n_splits"], 4)
            self.assertEqual(run_real_smoke(run, dataset_root=data)["status"], "PASS")
            cv = run_cv(run, dataset_root=data)
            self.assertEqual(cv["oof_sample_count"], len(load_development_samples(data)))
            frozen = freeze_configuration(run)
            self.assertFalse(frozen["test_metrics_seen"])
            self.assertEqual(refit_development(run, dataset_root=data)["status"], "REFIT_COMPLETE")
            checkpoint = load_checkpoint(run / "refit" / "checkpoint.pkl")
            self.assertEqual(tuple(checkpoint["extra"]["class_names"]), CLASS_NAMES)
            result = run_frozen_test(run, dataset_root=data)
            self.assertEqual(result["status"], "TEST_CONSUMED")
            self.assertEqual(_load_lifecycle(run).state, ExperimentState.TEST_CONSUMED)
            with self.assertRaises(RuntimeError):
                run_frozen_test(run, dataset_root=data)
            oof_ids = json.loads((run / "oof" / "sample_ids.json").read_text())
            self.assertEqual(len(oof_ids), len(set(oof_ids)))
            self.assertTrue((run / "manifest.json").is_file())


@unittest.skipUnless(
    HAS_TORCH and os.environ.get("LITHOFACIES_P4_REAL_SMOKE") == "1",
    "set LITHOFACIES_P4_REAL_SMOKE=1 with integration HDF5 to run the real-data smoke gate",
)
class P4RealDataSmokeTests(unittest.TestCase):
    def test_real_hdf5_one_step_smoke(self) -> None:
        dataset_root = Path(
            os.environ.get(
                "LITHOFACIES_P4_DATASET_ROOT",
                PROJECT_ROOT / "_data" / "processed" / "lithofacies",
            )
        ).resolve()
        if not (dataset_root / "train.h5").is_file() or not (dataset_root / "test.h5").is_file():
            self.skipTest(f"real lithofacies HDF5 is absent from integration worktree: {dataset_root}")
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            config = {
                **DEFAULT_CONFIG,
                "model_id": "lithofacies_concat_linear",
                "max_epochs": 1,
                "min_epochs": 1,
                "patience": 1,
            }
            prepare_run(run, config=config, dataset_root=dataset_root)
            report = run_real_smoke(run, dataset_root=dataset_root)
            self.assertEqual(report["status"], "PASS")
            self.assertTrue(report["finite_logits"])


if __name__ == "__main__":
    unittest.main()
