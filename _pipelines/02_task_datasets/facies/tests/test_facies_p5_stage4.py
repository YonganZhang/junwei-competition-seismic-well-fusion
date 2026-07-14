from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from _code.ml_framework.artifacts import atomic_write_json, hash_file
from _code.ml_framework.splits import Fold, SplitManifest, validate_manifest

import facies_p5_stage2 as stage2
import facies_p5_stage3 as stage3
from p4_tasks import LABEL_VERSIONS, get_task_spec


def _load_track_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, TRACK_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename} as {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


stage4 = _load_track_module("facies_p5_stage4", "facies_p5_stage4.py")


def _synthetic_manifest(root: Path, task_id: str) -> tuple[Path, SplitManifest]:
    classes = int(get_task_spec(task_id).metadata["num_classes"])
    groups = (100, 200, 300, 400, 500)
    sample_ids = tuple(
        f"{task_id}:train:sample_{index:07d}" for index in range(len(groups))
    )
    folds = []
    for fold_id in range(5):
        folds.append(
            Fold(
                fold_id=fold_id,
                train_groups=tuple(str(group) for index, group in enumerate(groups) if index != fold_id),
                validation_groups=(str(groups[fold_id]),),
                train_sample_ids=tuple(value for index, value in enumerate(sample_ids) if index != fold_id),
                validation_sample_ids=(sample_ids[fold_id],),
                purge={
                    "strategy": "synthetic_spatial_buffer",
                    "buffer_groups": 10,
                    "nearest_train_validation_inline_distance": 100,
                },
                support={
                    "train_per_class_pixels": [100] * classes,
                    "validation_per_class_pixels": [25] * classes,
                },
            )
        )
    manifest = SplitManifest(
        manifest_version="facies-p4-v1",
        group_key="inline",
        requested_n_splits=5,
        effective_n_splits=5,
        downgrade_reason=None,
        test_groups=("520",),
        test_sample_ids=(f"{task_id}:test:sample_0000000",),
        development_groups=tuple(str(value) for value in groups),
        development_sample_ids=sample_ids,
        folds=tuple(folds),
        metadata={
            "track_id": "facies",
            "task_id": task_id,
            "label_version": LABEL_VERSIONS[task_id],
            "num_classes": classes,
            "outer_split": {
                "development_inline_range": [100, 500],
                "external_guard_inline_range": [501, 519],
                "test_inline_range": [520, 540],
            },
        },
    )
    validate_manifest(manifest)
    path = root / "split_manifest.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return path, manifest


def _synthetic_lifecycle(root: Path, task_id: str, split_hash: str) -> Path:
    path = root / "lifecycle.json"
    atomic_write_json(
        path,
        {
            "experiment_id": f"synthetic-{task_id}",
            "state": "TEST_CONSUMED",
            "test_consumed_at": "2026-07-13T00:00:00+00:00",
            "evidence": {
                "SPLIT_LOCKED": {
                    "task_id": task_id,
                    "split_hash": split_hash,
                    "test_labels_read": False,
                },
                "TEST_CONSUMED": {"split_hash": split_hash},
            },
        },
    )
    return path


def _synthetic_development_archive(
    processed_root: Path, manifest: SplitManifest, task_id: str
) -> None:
    classes = int(get_task_spec(task_id).metadata["num_classes"])
    task_root = processed_root / task_id
    task_root.mkdir(parents=True)
    with h5py.File(task_root / "train.h5", "w") as archive:
        archive.attrs["task"] = task_id
        archive.attrs["split"] = "train"
        for index, sample_id in enumerate(manifest.development_sample_ids):
            key = sample_id.rsplit(":", 1)[-1]
            group = archive.create_group(key)
            raw = np.linspace(-1, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
            raw = raw + index * 0.1
            label = np.resize(np.arange(classes, dtype=np.uint8), (32, 32))
            group.create_dataset("seismic_patch", data=raw)
            group.create_dataset("label", data=label)
            group.attrs["position"] = json.dumps(
                {"inline": int(manifest.development_groups[index]), "crossline": 1, "time_ms": 0.0}
            )
            group.attrs["meta"] = json.dumps(
                {"task": task_id, "split": "train", "source": "synthetic"}
            )


def _locked_task(task_id: str = "facies_f3"):
    spec = get_task_spec(task_id)
    return stage4.LockedTask(
        task_id=task_id,
        winner_model_id=stage4.FROZEN_WINNERS[task_id],
        label_version=spec.label_version,
        num_classes=int(spec.metadata["num_classes"]),
        leaderboard_sha256=stage4.EXPECTED_STAGE3_LEADERBOARD_SHA256[task_id],
        manifest_stable_hash=stage4.EXPECTED_MANIFEST_STABLE_HASH[task_id],
        manifest_file_sha256="a" * 64,
        development_sample_ids=(f"{task_id}:train:sample_0000000",),
        development_groups=("100",),
        test_sample_ids=(f"{task_id}:test:sample_0000000",),
        test_groups=("520",),
        prior_lifecycle_sha256="b" * 64,
        prior_test_consumed_at="2026-07-13T00:00:00+00:00",
    )


class FaciesP5Stage4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_track_prefixed_names_and_exact_frozen_winners(self) -> None:
        self.assertEqual(Path(__file__).name, "test_facies_p5_stage4.py")
        self.assertEqual(stage4.__name__, "facies_p5_stage4")
        self.assertEqual(Path(stage4.__file__).name, "facies_p5_stage4.py")
        self.assertEqual(
            stage4.FROZEN_WINNERS,
            {
                "facies_f3": "smp_fpn_r18",
                "facies_penobscot": "smp_deeplabv3plus_r18",
            },
        )
        self.assertEqual(stage4.EVIDENCE_CLASS, "previously_seen_reusable_holdout")

    def test_budget_is_exact_stage3_copy_and_mutations_fail(self) -> None:
        self.assertEqual(asdict(stage4.Stage4Budget()), asdict(stage3.Stage3Budget()))
        seed_report = stage4._seed_model(123)
        self.assertEqual(seed_report["root_seed"], 2693)
        self.assertTrue(seed_report["seed_tree"])
        with self.assertRaisesRegex(ValueError, "preserve"):
            stage4.Stage4Budget(max_updates=41)
        with self.assertRaisesRegex(ValueError, "preserve"):
            stage4.Stage4Budget(loss_id="focal")

    def test_committed_stage3_hashes_and_unique_winners_verify(self) -> None:
        evidence = stage4.validate_stage3(stage4.DEFAULT_STAGE3_ROOT)
        self.assertEqual(evidence["summary_sha256"], stage4.EXPECTED_STAGE3_SUMMARY_SHA256)
        for task_id, expected in stage4.FROZEN_WINNERS.items():
            self.assertEqual(evidence["tasks"][task_id]["winner_model_id"], expected)
            self.assertEqual(evidence["tasks"][task_id]["source_lock"]["allowed_lanes"], ["scratch"])

    def test_split_and_prior_consumption_are_bound_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_id = "facies_f3"
            manifest_path, manifest = _synthetic_manifest(root, task_id)
            stable_hash = manifest.stable_hash()
            lifecycle_path = _synthetic_lifecycle(root, task_id, stable_hash)
            before = hash_file(lifecycle_path)
            evidence = {
                "winner_model_id": stage4.FROZEN_WINNERS[task_id],
                "leaderboard_sha256": stage4.EXPECTED_STAGE3_LEADERBOARD_SHA256[task_id],
            }
            patched_hashes = {**stage2.LOCKED_MANIFEST_STABLE_HASHES, task_id: stable_hash}
            with (
                patch.object(stage2, "LOCKED_MANIFEST_STABLE_HASHES", patched_hashes),
                patch.object(stage4, "EXPECTED_MANIFEST_STABLE_HASH", patched_hashes),
            ):
                locked = stage4.lock_task(
                    task_id=task_id,
                    manifest_path=manifest_path,
                    lifecycle_path=lifecycle_path,
                    stage3_evidence=evidence,
                )
            self.assertEqual(locked.manifest_stable_hash, stable_hash)
            self.assertEqual(locked.prior_lifecycle_sha256, before)
            self.assertEqual(hash_file(lifecycle_path), before)
            self.assertFalse(set(locked.development_groups) & set(locked.test_groups))

    def test_one_task_full_development_preprocessor_and_frozen_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_id = "facies_f3"
            _, manifest = _synthetic_manifest(root, task_id)
            processed = root / "processed"
            _synthetic_development_archive(processed, manifest, task_id)
            spec = get_task_spec(task_id)
            locked = stage4.LockedTask(
                task_id=task_id,
                winner_model_id=stage4.FROZEN_WINNERS[task_id],
                label_version=spec.label_version,
                num_classes=int(spec.metadata["num_classes"]),
                leaderboard_sha256=stage4.EXPECTED_STAGE3_LEADERBOARD_SHA256[task_id],
                manifest_stable_hash=manifest.stable_hash(),
                manifest_file_sha256="a" * 64,
                development_sample_ids=tuple(manifest.development_sample_ids),
                development_groups=tuple(manifest.development_groups),
                test_sample_ids=tuple(manifest.test_sample_ids),
                test_groups=tuple(manifest.test_groups),
                prior_lifecycle_sha256="b" * 64,
                prior_test_consumed_at="2026-07-13T00:00:00+00:00",
            )
            prepared = stage4.prepare_refit(locked, processed, stage4.Stage4Budget())
            self.assertEqual(prepared.preprocessor.fit_sample_count, 5)
            self.assertEqual(prepared.images.shape, (5, 1, 32, 32))
            self.assertEqual(prepared.update_schedule.shape, (40, 2))
            self.assertLessEqual(prepared.preprocessor.roundtrip_max_abs_error, 1e-2)
            self.assertEqual(len(prepared.development_support), 10)
            self.assertTrue(set(prepared.sampled_sample_ids) <= set(locked.development_sample_ids))

    def test_firewall_requires_refit_bound_single_use_state_before_archive(self) -> None:
        locked = _locked_task()
        evidence = {
            "configuration_hash": "c" * 64,
            "checkpoint": {"sha256": "d" * 64},
            "test_archive_opened": False,
            "test_labels_read": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "stage4_state.json"
            with self.assertRaisesRegex(RuntimeError, "TEST_ACCESS_STARTED"):
                stage4._assert_access_started(state_path, locked, evidence)
            stage4._write_access_started(
                root,
                {task_id: (_locked_task(task_id)) for task_id in stage4.TASK_IDS},
                {
                    task_id: {
                        **evidence,
                        "configuration_hash": f"{index + 1}" * 64,
                        "checkpoint": {"sha256": f"{index + 3}" * 64},
                    }
                    for index, task_id in enumerate(stage4.TASK_IDS)
                },
            )
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                stage4._write_access_started(
                    root,
                    {task_id: (_locked_task(task_id)) for task_id in stage4.TASK_IDS},
                    {task_id: evidence for task_id in stage4.TASK_IDS},
                )

    def test_holdout_accessor_checks_state_before_constructing_archive(self) -> None:
        parameters = inspect.signature(stage4.consume_known_holdout).parameters
        self.assertIn("refit_evidence", parameters)
        self.assertIn("output_root", parameters)
        locked = _locked_task()
        evidence = {
            "configuration_hash": "c" * 64,
            "checkpoint": {"sha256": "d" * 64, "runtime_relative_path": "missing.ckpt"},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(stage4, "FaciesArchive") as archive:
                with self.assertRaisesRegex(RuntimeError, "TEST_ACCESS_STARTED"):
                    stage4.consume_known_holdout(
                        locked=locked,
                        refit_evidence=evidence,
                        processed_root=root,
                        output_root=root,
                        runtime_root=root,
                        device=torch.device("cpu"),
                        batch_size=2,
                    )
                archive.assert_not_called()

    def test_representative_figure_requires_informative_sample_when_available(self) -> None:
        classes = 8
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "predictions.npz"
            labels = np.zeros((2, 16, 16), dtype=np.uint8)
            labels[1, :, 8:] = 1
            predicted = labels.copy()
            predicted[1, 0, 0] = 1
            error = (predicted != labels).astype(np.uint8)
            stage4._atomic_save_npz(
                prediction,
                sample_ids=np.asarray(["uninformative", "informative"], dtype=str),
                inline=np.asarray([520, 521]),
                seismic=np.zeros((2, 16, 16), dtype=np.float16),
                labels=labels,
                probabilities=np.full((2, classes, 16, 16), 1 / classes, dtype=np.float16),
                prediction=predicted,
                confidence=np.ones((2, 16, 16), dtype=np.float16),
                entropy=np.zeros((2, 16, 16), dtype=np.float16),
                error=error,
            )
            bins = [
                {
                    "lower": index / 15,
                    "upper": (index + 1) / 15,
                    "count": 1 if index == 14 else 0,
                    "mean_confidence": 1.0 if index == 14 else None,
                    "accuracy": 0.9 if index == 14 else None,
                }
                for index in range(15)
            ]
            metrics = {
                "accuracy": 0.9,
                "miou": 0.4,
                "macro_f1": 0.5,
                "ece": 0.1,
                "nll": 1.0,
                "per_class_support": [10] * classes,
                "per_class_iou": [0.4] * classes,
                "per_class_f1": [0.5] * classes,
                "confusion_matrix": np.eye(classes, dtype=int).tolist(),
                "reliability_bins": bins,
            }
            metrics_path = atomic_write_json(root / "metrics.json", metrics)
            manifest = stage4.render_task_figure(
                task_id="facies_penobscot",
                prediction_path=prediction,
                metrics_path=metrics_path,
                output_path=root / "figure.png",
                manifest_path=root / "visualization_manifest.json",
            )
            self.assertEqual(manifest["selection"]["sample_id"], "informative")
            self.assertTrue(manifest["selection"]["eligible"])
            self.assertTrue((root / "figure.png").is_file())

    def test_gpu_contract_and_nonempty_output_fail_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "cuda:0"):
            stage4.validate_gpu_contract(torch.device("cpu"), str(stage4.EXPECTED_GPU_LOCK))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "portable"
            runtime = root / "runtime"
            output.mkdir()
            (output / "state.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "single-use"):
                stage4._validate_new_output(output, runtime)

    def test_completed_confirmation_cannot_use_recovery_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_write_json(
                root / "stage4_state.json",
                {"state": "CONFIRMATION_COMPLETE", "single_use": True},
            )
            with self.assertRaisesRegex(RuntimeError, "TEST_ACCESS_STARTED"):
                stage4.resume_incomplete_confirmation(
                    processed_root=root,
                    manifest_paths={},
                    lifecycle_paths={},
                    stage3_root=root,
                    output_root=root,
                    runtime_root=root,
                    device=torch.device("cpu"),
                )


if __name__ == "__main__":
    unittest.main()
