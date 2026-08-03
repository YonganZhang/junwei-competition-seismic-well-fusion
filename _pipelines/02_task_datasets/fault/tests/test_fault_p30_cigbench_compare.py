from __future__ import annotations

import pathlib
import unittest
from unittest import mock

import numpy as np


TRACK_DIR = pathlib.Path(__file__).resolve().parents[1]


def _load_module():
    import sys

    sys.path.insert(0, str(TRACK_DIR))
    sys.path.insert(0, str(TRACK_DIR.parents[2] / "_code"))
    import fault_p30_cigbench_compare as module

    return module


class P30CigBenchCompareTests(unittest.TestCase):
    def test_portable_audited_v2_checkpoint_replays_archived_metrics(self) -> None:
        module = _load_module()
        with np.load(module.P30_SUBVOLUME_PATH) as archive:
            dev = {key: archive[key] for key in archive.files}
        split = module.load_json(module.P30_SPLIT_MANIFEST_PATH)
        folds = {fold.name: fold for fold in module.parse_fold_views(dev, split)}
        # This regression owns the portable-coefficient replay path.  The
        # checked-in joblib checkpoint is validated separately by lifecycle
        # hashes and can differ by a few float32 ulps at the selected threshold.
        # Force the portable branch so the archived portable metrics are
        # compared to the implementation that actually produced them.
        missing_joblib = TRACK_DIR / "_outputs" / "missing_for_portable_replay.joblib"
        with mock.patch.object(module, "AUDITED_V2_MODEL_PATH", missing_joblib):
            fit_probabilities, info = module.predict_baseline_volume(folds["fit"].seismic)
            guard_probabilities, _ = module.predict_baseline_volume(folds["guard"].seismic)
        fit = module.evaluate_model_on_fold(folds["fit"], fit_probabilities)
        guard = module.evaluate_model_on_fold(
            folds["guard"],
            guard_probabilities,
            threshold=fit["threshold"],
            threshold_source="fit_reused",
        )
        archived = module.load_json(module.SCALED_OUTPUT_ROOT / "comparison.json")["models"]["fault_local_logistic"]
        for stage, actual in (("fit", fit["metrics"]), ("guard", guard["metrics"])):
            for metric in ("precision", "recall", "f1", "iou", "threshold"):
                self.assertAlmostEqual(actual[metric], archived[stage][metric], places=12)
        self.assertIn("portable_coefficients", info["model_class"])

    def test_compare_models_excludes_unknown_and_uses_fit_threshold(self) -> None:
        module = _load_module()
        seismic = np.zeros((2, 4, 2), dtype=np.float32)
        seismic[:, :2, :] = 1.0
        seismic[:, 2:, :] = 2.0
        positive = np.zeros_like(seismic, dtype=bool)
        positive[:, 0, 0] = True
        positive[:, 1, 1] = True
        positive[:, 2, 0] = True
        positive[:, 3, 1] = True
        unknown = np.zeros_like(seismic, dtype=bool)
        unknown[:, 1, 0] = True
        verified_background = ~(positive | unknown)
        dev = {
            "seismic": seismic,
            "positive_mask": positive,
            "unknown_mask": unknown,
            "verified_background_mask": verified_background,
            "iline": np.asarray([10, 11, 12, 13], dtype=np.int32),
            "time_idx": np.asarray([5, 6], dtype=np.int32),
            "xline": np.asarray([20, 21], dtype=np.int32),
            "tline_ms": np.asarray([100.0, 200.0], dtype=np.float32),
        }
        split_manifest = {
            "development_only": True,
            "group_isolated": True,
            "frozen_holdout_accessed": False,
            "coordinate_order": ["tline", "iline", "xline"],
            "subvolume": {"inline": [10, 13], "crossline": [20, 21], "time_idx": [5, 6], "time_ms": [100.0, 200.0]},
            "blocks": [
                {"name": "fit", "inline": [10, 11]},
                {"name": "guard", "inline": [12, 13]},
            ],
        }
        cig_fit = np.array(
            [
                [[0.95, 0.05], [0.05, 0.95]],
                [[0.95, 0.05], [0.05, 0.95]],
            ],
            dtype=np.float32,
        )
        cig_guard = np.array(
            [
                [[0.80, 0.20], [0.20, 0.80]],
                [[0.80, 0.20], [0.20, 0.80]],
            ],
            dtype=np.float32,
        )
        baseline_fit = np.array(
            [
                [[0.90, 0.10], [0.10, 0.90]],
                [[0.90, 0.10], [0.10, 0.90]],
            ],
            dtype=np.float32,
        )
        baseline_guard = np.array(
            [
                [[0.40, 0.60], [0.60, 0.40]],
                [[0.40, 0.60], [0.60, 0.40]],
            ],
            dtype=np.float32,
        )

        captured_scales: list[tuple[float, float, float]] = []

        def fake_cig(volume: np.ndarray, *, device: str, scale_t: float, scale_h: float, scale_w: float):
            _ = device
            captured_scales.append((scale_t, scale_h, scale_w))
            if volume.shape[1] == 2 and float(volume.mean()) == 1.0:
                return cig_fit, {"package": "cig_bench", "package_version": "x", "restore_path": "/tmp/cig", "restore_sha256": "a", "restore_bytes": 1}
            return cig_guard, {"package": "cig_bench", "package_version": "x", "restore_path": "/tmp/cig", "restore_sha256": "a", "restore_bytes": 1}

        def fake_baseline(volume: np.ndarray):
            if float(volume.mean()) == 1.0:
                return baseline_fit, {"model_class": "fake", "model_builder": "fake", "model_description": "fake", "elapsed_seconds": 0.0, "note": "fake"}
            return baseline_guard, {"model_class": "fake", "model_builder": "fake", "model_description": "fake", "elapsed_seconds": 0.0, "note": "fake"}

        with mock.patch.object(module, "git_head", return_value="deadbeef"), \
            mock.patch.object(module, "predict_cigbench_volume", side_effect=fake_cig), \
            mock.patch.object(module, "predict_baseline_volume", side_effect=fake_baseline):
            report = module.compare_models(dev, split_manifest, device="cpu")

        self.assertEqual(report["status"], "READY")
        self.assertAlmostEqual(report["comparison"]["fit_thresholds"]["cig_bench"], 0.95, places=6)
        self.assertAlmostEqual(report["comparison"]["fit_thresholds"]["fault_local_logistic"], 0.9, places=6)
        self.assertAlmostEqual(
            report["models"]["cig_bench_fault_predictor"]["guard"]["threshold"],
            report["models"]["cig_bench_fault_predictor"]["fit"]["threshold"],
            places=6,
        )
        self.assertAlmostEqual(
            report["models"]["fault_local_logistic"]["guard"]["threshold"],
            report["models"]["fault_local_logistic"]["fit"]["threshold"],
            places=6,
        )
        self.assertEqual(report["cig_bench_scale"], {"scale_t": 0.5, "scale_h": 0.85, "scale_w": 0.85})
        self.assertEqual(report["models"]["cig_bench_fault_predictor"]["guard"]["threshold_source"], "fit_reused")
        self.assertEqual(report["models"]["fault_local_logistic"]["guard"]["threshold_source"], "fit_reused")
        self.assertEqual(captured_scales, [(0.5, 0.85, 0.85), (0.5, 0.85, 0.85)])
        self.assertEqual(report["split"]["folds"][0]["unknown_voxels"], 2)
        self.assertGreaterEqual(report["models"]["cig_bench_fault_predictor"]["guard"]["scoreable_voxels"], 1)
        self.assertAlmostEqual(
            report["comparison"]["guard_delta"]["f1"],
            report["models"]["cig_bench_fault_predictor"]["guard"]["f1"] - report["models"]["fault_local_logistic"]["guard"]["f1"],
        )

    def test_render_evidence_mentions_p30_and_guard(self) -> None:
        module = _load_module()
        report = {
            "generated_at": "2026-08-01T00:00:00+00:00",
            "source_commit": "abc",
            "p30_manifest": "_pipelines/02_task_datasets/fault/_outputs/p30_3d_dev_gate/manifest.json",
            "p30_split_manifest": "_pipelines/02_task_datasets/fault/_outputs/p30_3d_dev_gate/split_manifest.json",
            "p30_gate": {"status": "READY", "reason_code": "OK"},
            "split": {
                "development_only": True,
                "group_isolated": True,
                "frozen_holdout_accessed": False,
                "folds": [
                    {"name": "fit", "inline": [10, 11], "shape": [2, 2, 2], "scoreable_voxels": 8, "positive_voxels": 2, "unknown_voxels": 1},
                ],
            },
            "cig_bench": {"package": "cig_bench", "package_version": "0.2.0", "restore_path": "/tmp/cig", "restore_sha256": "a"},
            "baseline_reference": {
                "audited_v2_model_path": "_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/baseline_model.joblib",
                "audited_v2_model_sha256": "b",
            },
            "models": {
                "cig_bench_fault_predictor": {"fit": {"threshold": 0.8}, "guard": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "iou": 0.75}},
                "fault_local_logistic": {"fit": {"threshold": 0.9}, "guard": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "iou": 0.65}},
            },
            "comparison": {"guard_delta": {"f1": 0.10, "precision": 0.10, "recall": 0.10, "iou": 0.10}},
        }
        text = module.render_evidence(report)
        self.assertIn("P30 CIG-Bench vs audited_v2 fault baseline comparison", text)
        self.assertIn("Guard deltas", text)
        self.assertIn("unknown_voxels", text)


if __name__ == "__main__":
    unittest.main()
