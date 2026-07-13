from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from _code.ml_framework.contracts import ModelBatch, TaskSpec
from _code.ml_framework.cv import run_development_cv
from _code.ml_framework.lifecycle import ExperimentLifecycle
from _code.ml_framework.splits import validate_manifest

from p4_data import ArchiveRecord, FaciesArchive
from p4_losses import LOSS_IDS, build_loss, softmax_probabilities
from p4_metrics import calibration_metrics, evaluate_probabilities
from p4_spatial import build_facies_spatial_manifest
from p4_tasks import TASK_IDS, get_task_spec, suggest_hparams
from p4_training import predict_consumed_frozen_test, run_development_fold
from p4_visualize import render_archived_diagnostics


def _record(task: str, split: str, line: int, classes: int) -> ArchiveRecord:
    return ArchiveRecord(
        task_id=task,
        split=split,
        sample_id=f"{task}:{split}:{line}",
        storage_key=f"sample_{line}",
        inline=line,
        crossline=0,
        time_ms=0.0,
        patch_shape=(4, classes),
        label_support=(tuple([4] * classes) if split == "train" else None),
        source="synthetic-contract",
    )


def _write_sample(
    archive: h5py.File,
    key: str,
    *,
    task: str,
    split: str,
    line: int,
    raw: np.ndarray,
    label: np.ndarray,
) -> None:
    group = archive.create_group(key)
    stored_stats = {"method": "zscore", "mean": 2.0, "std": 3.0, "vmin": None, "vmax": None}
    group.create_dataset("seismic_patch", data=((raw - 2.0) / 3.0).astype(np.float32))
    group.create_dataset("label", data=label.astype(np.uint8))
    group.attrs["position"] = json.dumps(
        {"inline": line, "crossline": 100, "time_ms": 4.0, "well_name": None}
    )
    group.attrs["meta"] = json.dumps(
        {
            "task": task,
            "split": split,
            "source": "synthetic-contract",
            "normalization_stats": stored_stats,
        }
    )


class TaskAndSplitContractTests(unittest.TestCase):
    def test_task_specs_are_strict_and_independent(self) -> None:
        f3 = get_task_spec("facies_f3")
        pen = get_task_spec("facies_penobscot")
        self.assertEqual(TaskSpec.from_dict(f3.to_dict()), f3)
        self.assertEqual(f3.metadata["valid_label_ids"], list(range(10)))
        self.assertEqual(pen.metadata["valid_label_ids"], list(range(8)))
        self.assertNotEqual(f3.label_version, pen.label_version)
        self.assertTrue(f3.metadata["label_spaces_must_not_be_combined"])
        self.assertEqual(f3.inference_transform["facies"]["name"], "softmax")
        self.assertEqual(f3.train_loss["facies"]["input"], "raw_logits")

    def test_buffered_spatial_cv_has_exact_oof_and_frozen_test(self) -> None:
        development = [_record("facies_f3", "train", line, 10) for line in range(100, 587)]
        frozen_test = [_record("facies_f3", "test", line, 10) for line in range(620, 751)]
        manifest = build_facies_spatial_manifest(
            "facies_f3", development, frozen_test, requested_n_splits=5
        )
        validate_manifest(manifest)
        self.assertEqual(manifest.effective_n_splits, 5)
        self.assertFalse(manifest.metadata["test_labels_read_during_split"])
        self.assertGreater(manifest.metadata["cv_excluded_buffer_sample_count"], 0)
        oof = [sample_id for fold in manifest.folds for sample_id in fold.validation_sample_ids]
        self.assertCountEqual(oof, manifest.development_sample_ids)
        for fold in manifest.folds:
            self.assertGreater(
                fold.purge["nearest_train_validation_inline_distance"],
                fold.purge["buffer_groups"],
            )

    def test_requested_five_can_downgrade_honestly(self) -> None:
        development = [_record("facies_f3", "train", line, 10) for line in range(100, 587)]
        frozen_test = [_record("facies_f3", "test", line, 10) for line in range(620, 751)]
        manifest = build_facies_spatial_manifest(
            "facies_f3",
            development,
            frozen_test,
            requested_n_splits=5,
            buffer_groups=121,
        )
        self.assertLess(manifest.effective_n_splits, 5)
        self.assertIn("support/buffer", manifest.downgrade_reason)

    def test_cv_hpo_interfaces_have_no_test_argument(self) -> None:
        self.assertNotIn("test", inspect.signature(run_development_cv).parameters)
        self.assertNotIn("test", inspect.signature(run_development_fold).parameters)
        self.assertNotIn("test", inspect.signature(suggest_hparams).parameters)

    def test_frozen_test_prediction_requires_consumed_lifecycle(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "TEST_CONSUMED"):
            predict_consumed_frozen_test(
                lifecycle=ExperimentLifecycle("not-consumed"),
                model=None,
                archive=None,
                records=(),
                preprocessor=None,
                batch_size=1,
                device=torch.device("cpu"),
                temperature=1.0,
            )


class AdapterAndLossTests(unittest.TestCase):
    def test_archive_recovers_raw_and_fits_fold_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_root = root / "facies_f3"
            task_root.mkdir()
            label = np.tile(np.arange(10, dtype=np.uint8), (4, 1))
            for split in ("train", "test"):
                with h5py.File(task_root / f"{split}.h5", "w") as archive:
                    archive.attrs["task"] = "facies_f3"
                    archive.attrs["split"] = split
                    _write_sample(
                        archive,
                        "sample_0000000",
                        task="facies_f3",
                        split=split,
                        line=100 if split == "train" else 620,
                        raw=np.arange(40, dtype=np.float32).reshape(4, 10),
                        label=(label if split == "train" else np.full_like(label, 255)),
                    )
            adapter = FaciesArchive("facies_f3", root)
            records = adapter.development_index()
            preprocessing = adapter.fit_preprocessor(records)
            batch = next(
                adapter.iter_model_batches(
                    records,
                    preprocessing,
                    batch_size=1,
                    shuffle=False,
                    seed=1,
                )
            )
            self.assertIsInstance(batch, ModelBatch)
            self.assertEqual(batch.inputs["seismic"].shape, (1, 1, 4, 10))
            self.assertLess(abs(float(batch.inputs["seismic"].mean())), 1e-5)
            self.assertLess(preprocessing.roundtrip_max_abs_error, 1e-2)
            metadata_only = adapter.frozen_test_index(labels_consumed=False)
            self.assertIsNone(metadata_only[0].label_support)
            with self.assertRaisesRegex(ValueError, "fixed schema"):
                adapter.frozen_test_index(labels_consumed=True)

    def test_all_losses_are_finite_and_softmax_is_inference_adapter(self) -> None:
        target = torch.arange(10).repeat(2, 4, 1)
        for loss_id in LOSS_IDS:
            with self.subTest(loss_id=loss_id):
                logits = torch.randn(2, 10, 4, 10, requires_grad=True)
                criterion = build_loss(
                    loss_id,
                    num_classes=10,
                    class_weights=torch.ones(10),
                )
                loss = criterion(logits, target)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                probabilities = softmax_probabilities(logits.detach())
                self.assertTrue(torch.allclose(probabilities.sum(1), torch.ones_like(target, dtype=torch.float32)))

    def test_metrics_and_calibration_are_finite(self) -> None:
        labels = np.tile(np.arange(10, dtype=np.uint8), (2, 4, 1))
        logits = np.random.default_rng(7).normal(size=(2, 10, 4, 10)).astype(np.float32)
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        metrics, matrix = evaluate_probabilities(probabilities, labels, num_classes=10)
        self.assertEqual(matrix.shape, (10, 10))
        self.assertTrue(np.isfinite(metrics["miou"]))
        self.assertEqual(len(metrics["per_class_f1"]), 10)
        self.assertEqual(len(metrics["reliability_bins"]), 15)
        self.assertTrue(np.isfinite(calibration_metrics(probabilities, labels)["ece"]))


class ReadOnlyVisualizationTests(unittest.TestCase):
    def test_visualizer_reads_archives_without_model_or_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.tile(np.arange(10, dtype=np.uint8), (2, 4, 1))
            probabilities = np.full((2, 10, 4, 10), 0.01, dtype=np.float32)
            for class_id in range(10):
                probabilities[:, class_id][labels == class_id] = 0.91
            metrics, _ = evaluate_probabilities(probabilities, labels, num_classes=10)
            prediction = probabilities.argmax(1).astype(np.uint8)
            confidence = probabilities.max(1).astype(np.float16)
            np.savez_compressed(
                root / "predictions.npz",
                sample_ids=np.asarray(["a", "b"]),
                inline=np.asarray([620, 621]),
                seismic=np.ones((2, 4, 10), dtype=np.float16),
                labels=labels,
                prediction=prediction,
                confidence=confidence,
                entropy=np.zeros((2, 4, 10), dtype=np.float16),
                error=(prediction != labels).astype(np.uint8),
            )
            (root / "metrics.json").write_text(json.dumps(metrics))
            render_archived_diagnostics(
                prediction_path=root / "predictions.npz",
                metrics_path=root / "metrics.json",
                output_path=root / "diagnostics.png",
                sidecar_path=root / "diagnostics.json",
            )
            self.assertTrue((root / "diagnostics.png").is_file())
            sidecar = json.loads((root / "diagnostics.json").read_text())
            self.assertTrue(sidecar["no_model_or_dataset_loaded"])


if __name__ == "__main__":
    unittest.main()
