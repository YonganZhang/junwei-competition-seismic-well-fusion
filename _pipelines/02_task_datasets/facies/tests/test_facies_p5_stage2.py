from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch import nn

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from _code.ml_framework.splits import Fold, SplitManifest, validate_manifest

from p4_tasks import LABEL_VERSIONS, get_task_spec


def _load_track_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, TRACK_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename} as {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Both the basename and dynamic module name are track-prefixed so parallel
# track suites cannot collide in one interpreter.
_stage2 = _load_track_module("facies_p5_stage2", "facies_p5_stage2.py")


class _TinyModel(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.head = nn.Conv2d(1, classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(inputs)


class _FakeDiscovered:
    capabilities = {
        "task_types": ["multiclass"],
        "input_modalities": ["seismic_amplitude_2d"],
        "supports_missing_mask": False,
        "supports_uncertainty": False,
    }

    def build(self, task_spec, **config):
        return _TinyModel(int(task_spec.metadata["num_classes"]))


def _task_layout(task_id: str) -> tuple[list[int], list[int], int]:
    if task_id == "facies_f3":
        return list(range(100, 104)), list(range(200, 204)), 25
    return list(range(1000, 1004)), list(range(1100, 1104)), 23


def _write_archive_and_manifest(root: Path, task_id: str) -> Path:
    classes = int(get_task_spec(task_id).metadata["num_classes"])
    validation_groups, train_groups, buffer_groups = _task_layout(task_id)
    all_groups = validation_groups + train_groups
    task_root = root / "processed" / task_id
    task_root.mkdir(parents=True)
    sample_ids: list[str] = []
    group_ids: list[str] = []
    with h5py.File(task_root / "train.h5", "w") as archive:
        archive.attrs["task"] = task_id
        archive.attrs["split"] = "train"
        for index, inline in enumerate(all_groups):
            key = f"sample_{index:07d}"
            sample_id = f"{task_id}:train:{key}"
            sample_ids.append(sample_id)
            group_ids.append(str(inline))
            group = archive.create_group(key)
            # Validation deliberately has incomplete support.  Selection must
            # not inspect labels to manufacture all-class validation support.
            label_width = 2 if inline in validation_groups else classes
            label = np.tile(
                np.arange(label_width, dtype=np.uint8),
                (32, (32 + label_width - 1) // label_width),
            )[:, :32]
            raw = (
                np.linspace(-1.0, 1.0, num=32 * 32, dtype=np.float32).reshape(32, 32)
                + index * 0.01
            )
            group.create_dataset("seismic_patch", data=raw)
            group.create_dataset("label", data=label)
            group.attrs["position"] = json.dumps(
                {"inline": inline, "crossline": 300, "time_ms": 0.0}
            )
            group.attrs["meta"] = json.dumps(
                {"task": task_id, "split": "train", "source": "synthetic-development"}
            )

    validation_ids = tuple(sample_ids[: len(validation_groups)])
    train_ids = tuple(sample_ids[len(validation_groups) :])
    fold_support = {
        "train_per_class_pixels": [100] * classes,
        "validation_per_class_pixels": [100] * classes,
    }
    fold0 = Fold(
        fold_id=0,
        train_groups=tuple(map(str, train_groups)),
        validation_groups=tuple(map(str, validation_groups)),
        train_sample_ids=train_ids,
        validation_sample_ids=validation_ids,
        purge={
            "strategy": "permanent_contiguous_inline_buffer",
            "buffer_groups": buffer_groups,
            "nearest_train_validation_inline_distance": min(train_groups) - max(validation_groups),
        },
        support=fold_support,
    )
    fold1 = Fold(
        fold_id=1,
        train_groups=tuple(map(str, validation_groups)),
        validation_groups=tuple(map(str, train_groups)),
        train_sample_ids=validation_ids,
        validation_sample_ids=train_ids,
        purge={
            "strategy": "permanent_contiguous_inline_buffer",
            "buffer_groups": buffer_groups,
            "nearest_train_validation_inline_distance": min(train_groups) - max(validation_groups),
        },
        support=fold_support,
    )
    manifest = SplitManifest(
        manifest_version="facies-p4-v1",
        group_key="inline",
        requested_n_splits=2,
        effective_n_splits=2,
        downgrade_reason=None,
        test_groups=("9999",),
        test_sample_ids=(f"{task_id}:test:not_opened",),
        development_groups=tuple(group_ids),
        development_sample_ids=tuple(sample_ids),
        folds=(fold0, fold1),
        metadata={
            "track_id": "facies",
            "task_id": task_id,
            "label_version": LABEL_VERSIONS[task_id],
            "num_classes": classes,
        },
    )
    validate_manifest(manifest)
    manifest_path = root / f"{task_id}_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return manifest_path


def _manifest_hash(path: Path) -> str:
    return _stage2._manifest_from_dict(json.loads(path.read_text())).stable_hash()


class FaciesP5Stage2ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_module_and_test_basename_are_track_prefixed(self) -> None:
        self.assertEqual(Path(__file__).name, "test_facies_p5_stage2.py")
        self.assertEqual(_stage2.__name__, "facies_p5_stage2")
        self.assertEqual(Path(_stage2.__file__).name, "facies_p5_stage2.py")
        self.assertEqual(
            _stage2.DEFAULT_PORTABLE_OUTPUT,
            TRACK_DIR / "_outputs" / "p5_stage2",
        )
        self.assertEqual(
            _stage2.DEFAULT_RUNTIME_OUTPUT,
            TRACK_DIR / "_outputs" / "p5_stage2_runtime",
        )

    def test_budget_caps_are_frozen_below_stage2_maxima(self) -> None:
        budget = _stage2.PilotBudget()
        self.assertEqual(budget.fold_id, 0)
        self.assertEqual(budget.max_updates, 40)
        self.assertLessEqual(budget.max_updates, 200)
        self.assertLessEqual(budget.max_wall_seconds, 600)
        with self.assertRaises(ValueError):
            _stage2.PilotBudget(max_updates=201)
        with self.assertRaises(ValueError):
            _stage2.PilotBudget(max_wall_seconds=601)
        with self.assertRaises(ValueError):
            _stage2.PilotBudget(fold_id=1)

    def test_runner_surface_and_archive_fail_closed_on_test(self) -> None:
        parameters = inspect.signature(_stage2.run_stage2).parameters
        for forbidden in ("test", "test_loader", "frozen_test", "test_root"):
            self.assertNotIn(forbidden, parameters)
        with tempfile.TemporaryDirectory() as directory:
            archive = _stage2.Stage2DevelopmentArchive("facies_f3", Path(directory))
            with self.assertRaisesRegex(RuntimeError, "frozen-test"):
                archive.split_path("test")

    def test_label_independent_subset_and_schedule_are_deterministic(self) -> None:
        ids = tuple(f"sample-{index}" for index in range(20))
        first = _stage2.deterministic_subset(ids, count=7, seed=2693)
        second = _stage2.deterministic_subset(ids, count=7, seed=2693)
        self.assertEqual(first, second)
        budget = _stage2.PilotBudget(max_updates=3, validation_interval=1)
        schedule_a = _stage2.fixed_update_schedule(7, budget, seed=123)
        schedule_b = _stage2.fixed_update_schedule(7, budget, seed=123)
        np.testing.assert_array_equal(schedule_a, schedule_b)
        self.assertEqual(schedule_a.shape, (3, budget.batch_size))

    def test_manifest_hash_mismatch_fails_instead_of_resplitting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _write_archive_and_manifest(Path(directory), "facies_f3")
            with self.assertRaisesRegex(ValueError, "re-splitting is forbidden"):
                _stage2.load_locked_manifest("facies_f3", manifest)

    def test_locked_fold_prep_has_guard_and_records_missing_support(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_archive_and_manifest(root, "facies_f3")
            budget = _stage2.PilotBudget(
                max_updates=1,
                max_train_samples=4,
                max_validation_samples=2,
                batch_size=2,
                validation_interval=1,
            )
            with patch.object(
                _stage2,
                "LOCKED_MANIFEST_STABLE_HASHES",
                {"facies_f3": _manifest_hash(manifest)},
            ):
                prepared = _stage2.prepare_pilot(
                    task_id="facies_f3",
                    manifest_path=manifest,
                    processed_root=root / "processed",
                    budget=budget,
                )
            self.assertEqual(prepared.fold_id, 0)
            self.assertFalse(set(prepared.train_sample_ids) & set(prepared.validation_sample_ids))
            self.assertFalse(set(prepared.train_groups) & set(prepared.validation_groups))
            self.assertGreater(prepared.nearest_inline_distance, prepared.buffer_groups)
            self.assertEqual(prepared.validation_support[2:], (0,) * 8)
            self.assertLessEqual(prepared.preprocessor.roundtrip_max_abs_error, 1e-2)

    def test_all_ten_cells_per_task_and_separate_leaderboards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = {
                task_id: _write_archive_and_manifest(root, task_id)
                for task_id in ("facies_f3", "facies_penobscot")
            }
            budget = _stage2.PilotBudget(
                max_updates=1,
                max_wall_seconds=30,
                max_train_samples=4,
                max_validation_samples=2,
                batch_size=2,
                validation_interval=1,
            )
            locked_hashes = {
                task_id: _manifest_hash(path) for task_id, path in manifests.items()
            }
            with (
                patch.object(_stage2, "LOCKED_MANIFEST_STABLE_HASHES", locked_hashes),
                patch.object(_stage2, "discover_model", return_value=_FakeDiscovered()),
            ):
                summary = _stage2.run_stage2(
                    manifest_paths=manifests,
                    processed_root=root / "processed",
                    output_root=root / "portable" / "facies" / "p5_stage2",
                    runtime_root=root / "runtime" / "facies" / "p5_stage2",
                    device=torch.device("cpu"),
                    budget=budget,
                )
            self.assertEqual(summary["result_count"], 20)
            self.assertTrue(summary["tasks_are_independent"])
            self.assertTrue(summary["cross_task_ranking_forbidden"])
            results_path = root / "portable" / "facies" / "p5_stage2" / "p5_stage2_results.jsonl"
            records = [json.loads(line) for line in results_path.read_text().splitlines()]
            self.assertEqual(len(records), 20)
            for task_id, classes in (("facies_f3", 10), ("facies_penobscot", 8)):
                task_records = [record for record in records if record["task_id"] == task_id]
                self.assertEqual(len(task_records), 10)
                self.assertEqual({record["head_num_classes"] for record in task_records}, {classes})
                self.assertEqual(
                    sum(record["status"] == "development_piloted" for record in task_records),
                    6,
                )
                self.assertEqual(sum(record["status"] == "skipped" for record in task_records), 4)
                leaderboard = json.loads(
                    (
                        root
                        / "portable"
                        / "facies"
                        / "p5_stage2"
                        / f"{task_id}_scratch_leaderboard.json"
                    ).read_text()
                )
                self.assertEqual(leaderboard["task_id"], task_id)
                self.assertEqual(len(leaderboard["rows"]), 6)
                self.assertTrue(all(row["all_classes_supported"] is False for row in leaderboard["rows"]))
            portable_text = "\n".join(
                path.read_text()
                for path in (root / "portable" / "facies" / "p5_stage2").glob("*")
                if path.is_file()
            )
            self.assertNotIn("/mnt/", portable_text)
            self.assertNotIn(".claude/worktrees", portable_text)


if __name__ == "__main__":
    unittest.main()
