#!/usr/bin/env python3
"""Unit, contract, tiny-overfit, and real-data smoke tests for fault P4."""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TRACK_DIR = Path(__file__).resolve().parent
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

from _code.ml_framework.artifacts import atomic_write_json, hash_payload
from _code.ml_framework.contracts import TaskSpec
from _code.ml_framework.hpo import rank_trials
from _code.ml_framework.trainer import TrainerState
from p4_contract import adapt_fault_arrays, fault_task_spec, validate_fault_batch
from p4_smoke import (
    AUDITED_SPLIT,
    BUILD_SUMMARY,
    FAULT_POINTS,
    SEISMIC_INDEX,
    run_preflight,
)
from p4_split import (
    SpatialSample,
    audit_blind_test,
    build_buffered_spatial_cv,
    run_buffered_development_cv,
)
from p4_visualization import render_fault_visualizations
from p4_workflow import (
    FaultRunContext,
    LegacyFaultBaselineAdapter,
    assert_development_interfaces_have_no_test_argument,
    fault_hpo_plan,
    fixed_baseline_configs,
    masked_fault_metrics,
    run_fault_fixed_trials,
    run_tiny_baseline_smoke,
)


def supported_spatial_samples(n_blocks: int = 5) -> list[SpatialSample]:
    if n_blocks == 5:
        inlines = (2, 12, 25, 35, 45, 55, 65, 75, 88, 98)
    elif n_blocks == 3:
        inlines = (10, 50, 90)
    else:
        raise ValueError(n_blocks)
    return [
        SpatialSample(
            sample_id=f"sample-{index}",
            inline=inline,
            positive_count=1,
            verified_negative_count=1,
        )
        for index, inline in enumerate(inlines)
    ]


class FaultTaskAndMaskTests(unittest.TestCase):
    def test_strict_task_spec_roundtrip(self) -> None:
        spec = fault_task_spec()
        self.assertEqual(TaskSpec.from_dict(spec.to_dict()), spec)
        self.assertEqual(spec.track_id, "fault")
        self.assertEqual(spec.metric_directions["average_precision"], "maximize")
        self.assertEqual(spec.inference_transform["fault"], "sigmoid")
        self.assertTrue(spec.hpo["test_loader_allowed"] is False)
        self.assertEqual(len(spec.required_figures), 4)

    def test_unknown_is_never_valid_negative_and_proxy_is_separate(self) -> None:
        amplitude = np.zeros((2, 1, 2, 3), dtype=np.float32)
        labels = np.zeros((2, 2, 3), dtype=np.uint8)
        labels[0, 0, 0] = 1
        verified_negative = np.zeros_like(labels, dtype=bool)
        verified_negative[0, 1, 1] = True
        batch = adapt_fault_arrays(
            amplitude,
            labels,
            [
                {"inline": 10, "crossline": 20, "time_index": 30},
                {"inline": 11, "crossline": 20, "time_index": 30},
            ],
            ["fault", "non_fault"],
            verified_negative_mask=verified_negative,
        )
        validate_fault_batch(batch)
        valid = np.asarray(batch.target_masks["fault"])
        proxy = np.asarray(batch.input_masks["proxy_mask"])
        unknown = np.asarray(batch.input_masks["unknown_mask"])
        self.assertEqual(int(valid.sum()), 2)
        self.assertEqual(int(proxy[1].sum()), 6)
        self.assertFalse(np.any(proxy & valid))
        np.testing.assert_array_equal(unknown, ~valid)

    def test_legacy_baseline_fails_closed_without_verified_negatives(self) -> None:
        amplitudes = np.zeros((2, 1, 3, 3), dtype=np.float32)
        amplitudes[0, 0, 1, 1] = 2.0
        labels = np.zeros((2, 3, 3), dtype=np.uint8)
        labels[0, 1, 1] = 1
        batch = adapt_fault_arrays(
            amplitudes,
            labels,
            [
                {"inline": 10, "crossline": 20, "time_index": 30},
                {"inline": 30, "crossline": 20, "time_index": 30},
            ],
            ["fault", "non_fault"],
        )
        adapter = LegacyFaultBaselineAdapter("fault_raw_logistic", seed=7)
        with self.assertRaisesRegex(RuntimeError, "positive and negative"):
            adapter.train_batch(batch)
        proxy_result = adapter.train_batch(batch, supervision_mode="proxy_regression")
        self.assertEqual(proxy_result["metric_role"], "proxy_regression_only")
        self.assertGreater(proxy_result["unknown_zero_weight"], 0)

    def test_formal_metrics_ignore_unknown_and_keep_proxy_separate(self) -> None:
        amplitudes = np.zeros((2, 1, 2, 2), dtype=np.float32)
        labels = np.zeros((2, 2, 2), dtype=np.uint8)
        labels[0, 0, 0] = 1
        verified_negative = np.zeros_like(labels, dtype=bool)
        verified_negative[0, 0, 1] = True
        batch = adapt_fault_arrays(
            amplitudes,
            labels,
            [
                {"inline": 10, "crossline": 20, "time_index": 30},
                {"inline": 30, "crossline": 20, "time_index": 30},
            ],
            ["fault", "non_fault"],
            verified_negative_mask=verified_negative,
        )
        first = np.full((2, 1, 2, 2), 0.2, dtype=np.float32)
        first[0, 0, 0, 0] = 0.9
        second = first.copy()
        unknown = np.asarray(batch.input_masks["unknown_mask"])
        second[unknown] = 1.0 - second[unknown]
        first_metrics = masked_fault_metrics(batch, first, threshold=0.5)
        second_metrics = masked_fault_metrics(batch, second, threshold=0.5)
        self.assertEqual(first_metrics["formal"], second_metrics["formal"])
        self.assertEqual(first_metrics["proxy"]["role"], "proxy_regression_only")


class FaultSplitAndBlindTests(unittest.TestCase):
    def test_requested_five_buffered_cv_and_honest_downgrade(self) -> None:
        plan = build_buffered_spatial_cv(
            supported_spatial_samples(5),
            requested_n_splits=5,
            buffer_inlines=1,
        )
        self.assertEqual(plan.status, "ready")
        self.assertEqual(plan.effective_n_splits, 5)
        self.assertEqual(
            sorted(sample_id for fold in plan.folds for sample_id in fold.validation_sample_ids),
            sorted(plan.development_sample_ids),
        )
        for fold in plan.folds:
            val_low, val_high = fold.validation_inline_range
            for train_low, train_high in fold.train_inline_ranges:
                self.assertTrue(train_high < val_low or train_low > val_high)

        downgraded = build_buffered_spatial_cv(
            supported_spatial_samples(3),
            requested_n_splits=5,
            buffer_inlines=1,
        )
        self.assertEqual(downgraded.effective_n_splits, 3)
        self.assertIn("reduced to 3", downgraded.downgrade_reason)

    def test_proxy_labels_cannot_enable_formal_cv(self) -> None:
        samples = [
            SpatialSample(f"s-{index}", index * 20, int(index % 2 == 0), 0, 10)
            for index in range(10)
        ]
        plan = build_buffered_spatial_cv(samples, requested_n_splits=5, buffer_inlines=1)
        self.assertEqual(plan.status, "not_feasible")
        self.assertEqual(plan.effective_n_splits, 0)
        self.assertIn("no audited verified-negative", plan.downgrade_reason)

    def test_development_api_archives_exact_oof_without_test_argument(self) -> None:
        plan = build_buffered_spatial_cv(
            supported_spatial_samples(5),
            requested_n_splits=5,
            buffer_inlines=1,
        )
        self.assertNotIn("test", inspect.signature(run_buffered_development_cv).parameters)
        with tempfile.TemporaryDirectory() as directory:
            def runner(fold):
                return {
                    "validation_sample_ids": fold.validation_sample_ids,
                    "metrics": {"average_precision": 0.60 + 0.01 * fold.fold_id},
                    "valid_label_count": len(fold.validation_sample_ids) * 2,
                    "oof_predictions": [
                        {"sample_id": sample_id, "probability": 0.5}
                        for sample_id in fold.validation_sample_ids
                    ],
                }

            summary = run_buffered_development_cv(
                plan,
                runner,
                output_dir=Path(directory),
            )
            self.assertEqual(summary["oof_sample_count"], len(plan.development_sample_ids))
            archived = json.loads((Path(directory) / "oof" / "predictions.json").read_text())
            self.assertEqual(len(archived), len(plan.development_sample_ids))
            with self.assertRaisesRegex(ValueError, "frozen"):
                run_buffered_development_cv(
                    plan,
                    runner,
                    output_dir=Path(directory) / "wrong-direction",
                    metric_direction="minimize",
                )

    def test_real_blind_audit_is_not_feasible_without_coverage_evidence(self) -> None:
        if not all(path.is_file() for path in (FAULT_POINTS, SEISMIC_INDEX, BUILD_SUMMARY, AUDITED_SPLIT)):
            self.skipTest("real fault audit inputs are unavailable")
        result = audit_blind_test(
            fault_points_path=FAULT_POINTS,
            seismic_index_path=SEISMIC_INDEX,
            audited_build_summary_path=BUILD_SUMMARY,
            audited_split_manifest_path=AUDITED_SPLIT,
        )
        self.assertEqual(result["status"], "not_feasible")
        self.assertEqual(result["reason_code"], "NO_UNCONSUMED_COMPLETE_ANNOTATION_BLOCK")
        self.assertEqual(result["consumed_regression_blocks"][1]["role"], "regression_evidence_only")
        archived_hash = result.pop("audit_hash")
        self.assertEqual(archived_hash, hash_payload(result))

    def test_complete_unconsumed_guard_block_can_be_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coverage_path = Path(directory) / "coverage.json"
            atomic_write_json(
                coverage_path,
                {
                    "blocks": [
                        {
                            "block_id": "guard-audited",
                            "inline_range": [10286, 10290],
                            "annotation_status": "complete",
                            "contiguous": True,
                            "positive_labels": 10,
                            "verified_negative_labels": 20,
                        }
                    ]
                },
            )
            result = audit_blind_test(
                fault_points_path=FAULT_POINTS,
                seismic_index_path=SEISMIC_INDEX,
                audited_build_summary_path=BUILD_SUMMARY,
                audited_split_manifest_path=AUDITED_SPLIT,
                annotation_coverage_path=coverage_path,
            )
            self.assertEqual(result["status"], "frozen")
            self.assertEqual(result["blind_test_block"]["block_id"], "guard-audited")


class FaultWorkflowTests(unittest.TestCase):
    def test_tiny_verified_baseline_outputs_finite_logits_and_probabilities(self) -> None:
        result = run_tiny_baseline_smoke(seed=2693, epochs=8)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["output_shape"], [10, 1, 3, 3])
        self.assertEqual(result["unknown_voxels_used_for_training"], 0)
        self.assertGreater(
            result["positive_probability_mean"],
            result["verified_negative_probability_mean"],
        )

    def test_fixed_baseline_hpo_is_maximize_and_development_only(self) -> None:
        self.assertEqual(fault_hpo_plan().direction, "maximize")
        self.assertEqual(len(fixed_baseline_configs()), 3)
        self.assertNotIn("test", inspect.signature(run_fault_fixed_trials).parameters)
        assert_development_interfaces_have_no_test_argument()
        with tempfile.TemporaryDirectory() as directory:
            results = run_fault_fixed_trials(
                [{"score": 0.4}, {"score": 0.8}],
                lambda params, seed: {"fold_scores": [params["score"], params["score"] - 0.1]},
                root_seed=2693,
                output_dir=Path(directory),
            )
            self.assertEqual(rank_trials(results)[0].params["score"], 0.8)
            archived = json.loads((Path(directory) / "trials.json").read_text())
            self.assertEqual(archived[0]["metric_direction"], "maximize")

    def test_complete_checkpoint_artifacts_and_single_test_lifecycle(self) -> None:
        plan = build_buffered_spatial_cv(
            supported_spatial_samples(3),
            requested_n_splits=5,
            buffer_inlines=1,
        )
        blind = {
            "status": "frozen",
            "blind_test_block": {"inline_range": [200, 210]},
            "split_hash": "blind",
        }
        with tempfile.TemporaryDirectory() as directory:
            context = FaultRunContext.initialize(
                Path(directory),
                run_id="fault-unit-run",
                split_plan=plan,
                blind_audit=blind,
            )
            context.mark_smoke_passed({"unit": "ok"})
            context.mark_cv_complete({"oof_hash": "oof"})
            config_hash = context.freeze_config({"model_id": "fault_raw_logistic"})
            trainer_state = TrainerState(
                next_epoch=2,
                global_step=4,
                best_epoch=1,
                best_val_loss=0.2,
                epochs_without_improvement=0,
                stopped_early=False,
                history=[{"epoch": 1, "validation_loss": 0.2}],
            )
            checkpoint, checkpoint_hash = context.save_complete_checkpoint(
                model_state={"weight": [1.0]},
                optimizer_state={"lr": 0.1},
                scheduler_state={"step": 2},
                scaler_state=None,
                trainer_state=trainer_state,
                config_hash=config_hash,
            )
            self.assertTrue(checkpoint.is_file())
            context.mark_refit_complete(checkpoint_hash)
            context.consume_test_once(config_hash=config_hash, checkpoint_hash=checkpoint_hash)
            with self.assertRaises(RuntimeError):
                context.consume_test_once(config_hash=config_hash, checkpoint_hash=checkpoint_hash)
            context.verify_artifacts()


class FaultVisualizationAndRealSmokeTests(unittest.TestCase):
    def test_visualizer_reads_archived_prediction_only(self) -> None:
        self.assertNotIn("model", inspect.signature(render_fault_visualizations).parameters)
        volume = np.zeros((3, 5, 7), dtype=np.float32)
        target = np.zeros_like(volume, dtype=np.uint8)
        target[1, 2, 2:5] = 1
        valid = np.zeros_like(volume, dtype=bool)
        valid[1, 1:4, 1:6] = True
        proxy = np.zeros_like(volume, dtype=bool)
        proxy[0] = True
        probability = np.full_like(volume, 0.1, dtype=np.float32)
        probability[target.astype(bool)] = 0.9
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "predictions.npz"
            metrics = root / "metrics.json"
            np.savez_compressed(
                prediction,
                amplitude=volume,
                target=target,
                valid_label_mask=valid,
                proxy_mask=proxy,
                probability=probability,
            )
            atomic_write_json(
                metrics,
                {
                    "threshold": 0.5,
                    "threshold_source": "pooled_oof",
                    "config_hash": "config",
                    "split_hash": "split",
                    "checkpoint_hash": "checkpoint",
                    "prediction_role": "synthetic_contract",
                },
            )
            report = render_fault_visualizations(
                prediction_path=prediction,
                metrics_path=metrics,
                output_dir=root / "figures",
            )
            self.assertFalse(report["selection_performed"])
            self.assertEqual(len(report["figures"]), 4)
            self.assertTrue(all(Path(path).is_file() for path in report["figures"].values()))

    def test_real_data_preflight_smoke_is_fast_and_fail_closed(self) -> None:
        if not all(path.is_file() for path in (FAULT_POINTS, SEISMIC_INDEX, BUILD_SUMMARY, AUDITED_SPLIT)):
            self.skipTest("real fault audit inputs are unavailable")
        with tempfile.TemporaryDirectory() as directory:
            report = run_preflight(Path(directory), root_seed=2693)
            self.assertEqual(report["real_data_smoke"]["status"], "passed")
            self.assertFalse(report["real_data_smoke"]["training_performed"])
            self.assertEqual(report["blind_test_status"], "not_feasible")
            self.assertEqual(report["effective_n_splits"], 0)
            self.assertFalse(report["hpo_executed"])
            self.assertTrue((Path(directory) / "blind_test_not_feasible.json").is_file())


if __name__ == "__main__":
    unittest.main()
