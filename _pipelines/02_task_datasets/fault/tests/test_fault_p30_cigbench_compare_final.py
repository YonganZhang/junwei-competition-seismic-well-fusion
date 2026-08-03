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
    import fault_p30_cigbench_compare_final as module

    return module


class FinalP30CigBenchCompareTests(unittest.TestCase):
    def test_compare_models_uses_single_full_volume_inference_and_reuses_fit_threshold(self) -> None:
        module = _load_module()
        seismic = np.zeros((2, 4, 2), dtype=np.float32)
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
                {"name": "guard", "inline": [12, 12]},
                {"name": "validation", "inline": [13, 13]},
            ],
        }
        cig_probs = np.array(
            [
                [[0.95, 0.05], [0.05, 0.95], [0.80, 0.20], [0.20, 0.80]],
                [[0.95, 0.05], [0.05, 0.95], [0.80, 0.20], [0.20, 0.80]],
            ],
            dtype=np.float32,
        )
        baseline_probs = np.array(
            [
                [[0.90, 0.10], [0.10, 0.90], [0.40, 0.60], [0.60, 0.40]],
                [[0.90, 0.10], [0.10, 0.90], [0.40, 0.60], [0.60, 0.40]],
            ],
            dtype=np.float32,
        )
        cig_calls: list[tuple[int, int, int]] = []
        baseline_calls: list[tuple[int, int, int]] = []

        def fake_cig(volume: np.ndarray, *, device: str, scale_t: float, scale_h: float, scale_w: float):
            _ = device
            cig_calls.append(tuple(int(v) for v in volume.shape))
            self.assertEqual((scale_t, scale_h, scale_w), (0.5, 0.85, 0.85))
            return cig_probs, {
                "package": "cig_bench",
                "package_version": "x",
                "restore_path": "/tmp/cig",
                "restore_sha256": "a",
                "restore_bytes": 1,
                "elapsed_seconds": 0.0,
                "scale_t": scale_t,
                "scale_h": scale_h,
                "scale_w": scale_w,
            }

        def fake_baseline(volume: np.ndarray):
            baseline_calls.append(tuple(int(v) for v in volume.shape))
            return baseline_probs, {
                "model_class": "fake",
                "model_builder": "fake",
                "model_description": "fake",
                "elapsed_seconds": 0.0,
                "note": "fake",
            }

        with mock.patch.object(module, "git_head", return_value="deadbeef"), \
            mock.patch.object(module, "predict_cigbench_volume", side_effect=fake_cig), \
            mock.patch.object(module, "predict_baseline_volume", side_effect=fake_baseline):
            report = module.compare_models(dev, split_manifest, device="cpu")

        self.assertEqual(cig_calls, [seismic.shape])
        self.assertEqual(baseline_calls, [seismic.shape])
        self.assertEqual(report["models"]["cig_bench_fault_predictor"]["guard"]["threshold_source"], "fit_reused")
        self.assertEqual(report["models"]["cig_bench_fault_predictor"]["validation"]["threshold_source"], "fit_reused")
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

    def test_render_mentions_st10010_full_volume(self) -> None:
        module = _load_module()
        report = {
            "generated_at": "2026-08-03T00:00:00+00:00",
            "source_commit": "abc",
            "asset_root": "_pipelines/02_task_datasets/fault/_outputs/p30_3d_dev_gate_st10010",
            "p30_manifest": "_pipelines/02_task_datasets/fault/_outputs/p30_3d_dev_gate_st10010/manifest.json",
            "p30_split_manifest": "_pipelines/02_task_datasets/fault/_outputs/p30_3d_dev_gate_st10010/split_manifest.json",
            "p30_subvolume": "_pipelines/02_task_datasets/fault/_outputs/p30_3d_dev_gate_st10010/dev_subvolume.npz",
            "split": {
                "development_only": True,
                "group_isolated": True,
                "frozen_holdout_accessed": False,
            },
            "cig_bench": {"package": "cig_bench", "package_version": "0.2.0", "restore_path": "/tmp/cig", "restore_sha256": "a", "restore_bytes": 1, "scale_t": 0.5, "scale_h": 0.85, "scale_w": 0.85},
            "cig_bench_scale": {"scale_t": 0.5, "scale_h": 0.85, "scale_w": 0.85},
            "baseline_reference": {
                "audited_v2_model_path": "_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/baseline_model.joblib",
                "audited_v2_model_sha256": "b",
                "audited_v2_metrics_path": "_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/baseline_metrics.json",
                "audited_v2_metrics_sha256": "c",
                "audited_v2_old_metrics": {},
            },
            "models": {
                "cig_bench_fault_predictor": {
                    "fit": {"threshold": 0.8},
                    "guard": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "iou": 0.75, "threshold_source": "fit_reused"},
                    "validation": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "iou": 0.65, "threshold_source": "fit_reused"},
                },
                "fault_local_logistic": {
                    "fit": {"threshold": 0.9},
                    "guard": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "iou": 0.65, "threshold_source": "fit_reused"},
                    "validation": {"precision": 0.7, "recall": 0.6, "f1": 0.65, "iou": 0.55, "threshold_source": "fit_reused"},
                },
            },
            "comparison": {"guard_delta": {"f1": 0.10, "precision": 0.10, "recall": 0.10, "iou": 0.10}, "primary_metric": "f1", "fit_thresholds": {"cig_bench": 0.8, "fault_local_logistic": 0.9}},
            "minimum_unblock_contract": ["full volume"],
        }
        text = module.render_evidence(report)
        self.assertIn("ST10010", text)
        self.assertIn("full ST10010 cube", text)


if __name__ == "__main__":
    unittest.main()
