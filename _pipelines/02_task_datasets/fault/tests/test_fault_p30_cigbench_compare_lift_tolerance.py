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
    import fault_p30_cigbench_compare_lift_tolerance as module

    return module


class P30CigBenchLiftToleranceTests(unittest.TestCase):
    def test_tolerance_scores_math_exact_shift_and_empty_cases(self) -> None:
        module = _load_module()
        truth = np.zeros((3, 3, 4), dtype=bool)
        truth[1, 1, 1] = True
        score_mask = np.ones_like(truth, dtype=bool)

        exact_pred = np.zeros_like(truth)
        exact_pred[1, 1, 1] = True
        exact = module.tolerance_scores(truth, exact_pred, score_mask, radius=0)
        self.assertEqual((exact["precision"], exact["recall"], exact["f1"]), (1.0, 1.0, 1.0))

        shifted_pred = np.zeros_like(truth)
        shifted_pred[1, 1, 2] = True
        strict = module.tolerance_scores(truth, shifted_pred, score_mask, radius=0)
        tolerant = module.tolerance_scores(truth, shifted_pred, score_mask, radius=1)
        self.assertEqual((strict["precision"], strict["recall"], strict["f1"]), (0.0, 0.0, 0.0))
        self.assertEqual((tolerant["precision"], tolerant["recall"], tolerant["f1"]), (1.0, 1.0, 1.0))

        farther_pred = np.zeros_like(truth)
        farther_pred[1, 1, 3] = True
        beyond = module.tolerance_scores(truth, farther_pred, score_mask, radius=1)
        self.assertEqual((beyond["precision"], beyond["recall"], beyond["f1"]), (0.0, 0.0, 0.0))

        empty_pred = np.zeros_like(truth)
        empty_pred_scores = module.tolerance_scores(truth, empty_pred, score_mask, radius=1)
        self.assertEqual((empty_pred_scores["precision"], empty_pred_scores["recall"], empty_pred_scores["f1"]), (0.0, 0.0, 0.0))

        empty_truth = np.zeros_like(truth)
        with self.assertRaisesRegex(ValueError, "positive truth"):
            module.tolerance_scores(empty_truth, exact_pred, score_mask, radius=1)

    def test_micro_aggregate_tolerance_summaries_sums_per_fold_counts(self) -> None:
        module = _load_module()
        summaries = [
            {
                "radius_1": {
                    "radius": 1,
                    "predicted_positive_voxels": 4,
                    "truth_positive_voxels": 2,
                    "matched_prediction_voxels": 1,
                    "matched_truth_voxels": 1,
                    "precision": 0.25,
                    "recall": 0.5,
                    "f1": 1 / 3,
                }
            },
            {
                "radius_1": {
                    "radius": 1,
                    "predicted_positive_voxels": 2,
                    "truth_positive_voxels": 4,
                    "matched_prediction_voxels": 1,
                    "matched_truth_voxels": 2,
                    "precision": 0.5,
                    "recall": 0.5,
                    "f1": 0.5,
                }
            },
        ]
        aggregated = module.aggregate_tolerance_summaries(summaries, radii=(1,))
        radius_1 = aggregated["radius_1"]
        self.assertEqual(radius_1["predicted_positive_voxels"], 6)
        self.assertEqual(radius_1["truth_positive_voxels"], 6)
        self.assertEqual(radius_1["matched_prediction_voxels"], 2)
        self.assertEqual(radius_1["matched_truth_voxels"], 3)
        self.assertAlmostEqual(radius_1["precision"], 2 / 6)
        self.assertAlmostEqual(radius_1["recall"], 3 / 6)
        self.assertAlmostEqual(radius_1["f1"], 0.4)

    def test_predicted_positive_fraction_and_coverage_ratio_use_tp_fp(self) -> None:
        module = _load_module()
        positive = np.zeros((2, 2, 1), dtype=bool)
        positive[0, 0, 0] = True
        positive[1, 1, 0] = True
        verified_background = ~positive
        fold = module.base.FoldView(
            name="fit",
            inline_start=10,
            inline_end=11,
            selection=np.asarray([0, 1], dtype=np.int32),
            positive_mask=positive,
            unknown_mask=np.zeros_like(positive, dtype=bool),
            verified_background_mask=verified_background,
        )
        probabilities = np.zeros((2, 2, 1), dtype=np.float32)
        probabilities[0, 0, 0] = 1.0
        probabilities[1, 1, 0] = 1.0

        fake_result = {
            "metrics": {
                "tp": 3,
                "fp": 7,
                "fn": 2,
                "tn": 8,
                "precision": 0.3,
                "recall": 0.6,
                "dice": 0.4,
                "iou": 0.25,
                "f1": 0.4,
                "average_precision": 0.5,
                "pr_auc": 0.5,
                "threshold": 0.5,
                "threshold_source": "fit_reused",
                "fit_selected_f1": 0.4,
                "scoreable_voxels": 20,
                "positive_voxels": 5,
                "unknown_voxels_excluded": 0,
            },
            "truth": np.asarray([1, 0, 1, 0], dtype=np.uint8),
            "probabilities": np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float64),
        }

        with mock.patch.object(module.base, "evaluate_fold", return_value=fake_result):
            enriched = module.enrich_fold_metrics(fold, probabilities, threshold=0.5, radii=(1,))

        self.assertAlmostEqual(enriched["metrics"]["predicted_positive_fraction"], 0.5)
        self.assertAlmostEqual(enriched["metrics"]["coverage_ratio"], 2.0)
        union = module.enrich_union_metrics(
            {
                "tp": 3,
                "fp": 7,
                "fn": 2,
                "tn": 8,
                "precision": 0.3,
                "recall": 0.6,
                "dice": 0.4,
                "iou": 0.25,
                "f1": 0.4,
                "average_precision": 0.5,
                "pr_auc": 0.5,
                "threshold": 0.5,
                "scoreable_voxels": 20,
                "positive_voxels": 5,
            },
            [enriched],
            prior=0.25,
            radii=(1,),
        )
        self.assertAlmostEqual(union["predicted_positive_fraction"], 0.5)
        self.assertAlmostEqual(union["coverage_ratio"], 2.0)
        self.assertNotEqual(union["coverage_ratio"], 1.0)

    def test_report_contains_lift_tolerance_and_verdict_fields(self) -> None:
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

        def fake_cig(volume: np.ndarray, *, device: str, scale_t: float, scale_h: float, scale_w: float):
            _ = device
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
            return baseline_probs, {
                "model_class": "fake",
                "model_builder": "fake",
                "model_description": "fake",
                "elapsed_seconds": 0.0,
                "note": "fake",
            }

        with mock.patch.object(module.base, "git_head", return_value="deadbeef"), \
            mock.patch.object(module.base, "predict_cigbench_volume", side_effect=fake_cig), \
            mock.patch.object(module.base, "predict_baseline_volume", side_effect=fake_baseline):
            report = module.compare_with_lift_and_tolerance(dev, split_manifest, device="cpu", radii=(1, 2))

        guard = report["comparison"]["guard_lift"]
        self.assertIn("precision_lift", guard)
        self.assertIn("average_precision_lift", guard)
        self.assertIn("predicted_positive_fraction", guard)
        self.assertIn("coverage_ratio", guard)
        self.assertIn("guard_tolerance_sweep", report["comparison"])
        self.assertIn("development_union_tolerance_sweep", report["comparison"])
        self.assertEqual(sorted(report["comparison"]["guard_tolerance_sweep"]["cig_bench"].keys()), ["radius_1", "radius_2"])
        self.assertEqual(sorted(report["comparison"]["development_union_tolerance_sweep"]["baseline"].keys()), ["radius_1", "radius_2"])
        self.assertEqual(report["decision"]["default_recommendation"], "do_not_advance")
        self.assertEqual(report["decision"]["model_classification"], "diagnostic_high_recall_proposal_only")
        self.assertEqual(report["models"]["cig_bench_fault_predictor"]["guard"]["threshold_source"], "fit_reused")

    def test_render_mentions_tolerance_and_prior(self) -> None:
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
            "tolerance_policy": {
                "distance_metric": "euclidean_voxel",
                "radii_voxels": [1, 2, 3],
                "primary_radius_voxels": 2,
                "rationale": ["x"],
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
                    "guard": {
                        "precision": 0.9,
                        "recall": 0.8,
                        "average_precision": 0.7,
                        "f1": 0.85,
                        "threshold_source": "fit_reused",
                        "precision_lift": 1.0,
                        "average_precision_lift": 1.0,
                        "recall_to_prior_ratio": 1.0,
                        "positive_prior": 0.01,
                        "predicted_positive_fraction": 0.694,
                        "coverage_ratio": 69.4,
                    },
                    "validation": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "iou": 0.65, "threshold_source": "fit_reused"},
                    "development_union": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "iou": 0.65},
                },
                "fault_local_logistic": {
                    "fit": {"threshold": 0.9},
                    "guard": {
                        "precision": 0.8,
                        "recall": 0.7,
                        "average_precision": 0.6,
                        "f1": 0.75,
                        "threshold_source": "fit_reused",
                        "precision_lift": 1.0,
                        "average_precision_lift": 1.0,
                        "recall_to_prior_ratio": 1.0,
                        "positive_prior": 0.01,
                        "predicted_positive_fraction": 0.04,
                        "coverage_ratio": 4.0,
                    },
                    "validation": {"precision": 0.7, "recall": 0.6, "f1": 0.65, "iou": 0.55, "threshold_source": "fit_reused"},
                    "development_union": {"precision": 0.7, "recall": 0.6, "f1": 0.65, "iou": 0.55},
                },
            },
            "comparison": {
                "primary_metric": "tolerance_f1_radius_2",
                "guard_delta": {"precision": 0.10, "recall": 0.10, "f1": 0.10, "iou": 0.10},
                "guard_lift": {"precision_lift": 1.0, "baseline_precision_lift": 1.0, "average_precision_lift": 1.0, "baseline_average_precision_lift": 1.0, "recall_to_prior_ratio": 1.0, "baseline_recall_to_prior_ratio": 1.0, "positive_prior": 0.01, "predicted_positive_fraction": 0.694, "baseline_predicted_positive_fraction": 0.04, "coverage_ratio": 69.4, "baseline_coverage_ratio": 4.0, "threshold": 0.0},
                "guard_tolerance_sweep": {"cig_bench": {"radius_1": {"precision": 0.8, "recall": 0.7, "f1": 0.75}, "radius_2": {"precision": 0.9, "recall": 0.8, "f1": 0.85}, "radius_3": {"precision": 1.0, "recall": 0.9, "f1": 0.95}}, "baseline": {"radius_1": {"precision": 0.6, "recall": 0.5, "f1": 0.55}, "radius_2": {"precision": 0.8, "recall": 0.7, "f1": 0.75}, "radius_3": {"precision": 0.9, "recall": 0.8, "f1": 0.85}}},
                "development_union_tolerance_sweep": {"cig_bench": {"radius_1": {"precision": 0.8, "recall": 0.7, "f1": 0.75}, "radius_2": {"precision": 0.9, "recall": 0.8, "f1": 0.85}, "radius_3": {"precision": 1.0, "recall": 0.9, "f1": 0.95}}, "baseline": {"radius_1": {"precision": 0.6, "recall": 0.5, "f1": 0.55}, "radius_2": {"precision": 0.8, "recall": 0.7, "f1": 0.75}, "radius_3": {"precision": 0.9, "recall": 0.8, "f1": 0.85}}},
                "tolerance_radius_2": {"cig_bench": {"radius": 2, "precision": 0.9, "recall": 0.8, "f1": 0.85}, "baseline": {"radius": 2, "precision": 0.8, "recall": 0.7, "f1": 0.75}},
                "fit_thresholds": {"cig_bench": 0.8, "fault_local_logistic": 0.9},
            },
            "decision": {"default_recommendation": "do_not_advance", "model_classification": "diagnostic_high_recall_proposal_only", "reason_codes": [], "summary": "x", "minimum_advancement_conditions": ["a"]},
            "minimum_unblock_contract": ["full volume"],
        }
        text = module.render_evidence(report)
        self.assertIn("precision lift", text)
        self.assertIn("Tolerance radius 2 voxels", text)
        self.assertIn("Default recommendation", text)


if __name__ == "__main__":
    unittest.main()
