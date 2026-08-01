from __future__ import annotations

import io
import inspect
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

import lithofacies_p28_agentic_optimization as p28  # noqa: E402


class P28AgenticOptimizationTests(unittest.TestCase):
    def test_frozen_a0_and_five_action_allowlist(self) -> None:
        self.assertEqual(p28.A0.max_depth, 3)
        self.assertEqual(p28.A0.eta, 0.1)
        self.assertEqual(p28.A0.rounds, 60)
        self.assertEqual(len(p28.ACTIONS), 5)
        self.assertEqual(
            set(p28.ACTION_IDS),
            {
                "ACT_DEPTH4_ETA0075_ROUNDS80",
                "ACT_WEIGHT_EXP05_MEAN1",
                "ACT_WEIGHT_EXP075_MEAN1",
                "ACT_WELL_MASK_ONLY_858",
                "ACT_PRIOR_SHRINK010",
            },
        )
        self.assertEqual(p28.TRIAL_BUDGET, 3)
        self.assertEqual(p28.EXPECTED_A0_DISPLAY, 0.2133487970)

    def test_nested_rotations_are_disjoint_and_complete(self) -> None:
        plan = p28.nested_rotation_plan()
        self.assertEqual(len(plan), 4)
        for item in plan:
            promotion = item["promotion_fold_id"]
            selection = item["selection_fold_ids"]
            self.assertEqual(len(selection), 3)
            self.assertNotIn(promotion, selection)
            self.assertEqual(set(selection) | {promotion}, set(p28.OUTER_FOLDS))
            self.assertTrue(item["disjoint"])

    def test_policy_payload_has_only_coarse_observables(self) -> None:
        observation = p28.build_policy_observation(
            support_buckets={
                "absent": 1,
                "very_low_1_to_5": 2,
                "low_6_to_20": 1,
                "medium_21_to_60": 2,
                "high_over_60": 3,
            },
            fit_state="overfit",
            history=[
                {
                    "action_id": "ACT_WEIGHT_EXP05_MEAN1",
                    "feedback": "flat",
                }
            ],
            remaining_actions=[
                action_id
                for action_id in p28.ACTION_IDS
                if action_id != "ACT_WEIGHT_EXP05_MEAN1"
            ],
            trial_index=1,
        )
        p28.assert_policy_payload_safe(observation)
        serialized = json.dumps(observation, sort_keys=True)
        for forbidden in (
            "fixed_schema_macro_f1",
            "sample_id",
            "family_id",
            "residual",
            "promotion",
        ):
            self.assertNotIn(forbidden, serialized)
        with self.assertRaisesRegex(ValueError, "forbidden policy field"):
            p28.assert_policy_payload_safe({"promotion_metric": 0.2})

    def test_mean_one_weights_and_well_mask_action(self) -> None:
        counts = np.asarray([0, 1, 4, 9, 16, 25, 36, 49, 64])
        weights = p28.class_weight_vector(
            counts,
            exponent=0.75,
            normalize_mean_one=True,
        )
        self.assertEqual(weights[0], 0.0)
        self.assertAlmostEqual(float(weights[counts > 0].mean()), 1.0)
        action = p28.ACTION_BY_ID["ACT_WELL_MASK_ONLY_858"]
        self.assertEqual(action.features, "well_mask_858")

    def test_live_policy_missing_credential_fails_closed(self) -> None:
        observation = p28.build_policy_observation(
            support_buckets={
                "absent": 0,
                "very_low_1_to_5": 1,
                "low_6_to_20": 2,
                "medium_21_to_60": 3,
                "high_over_60": 3,
            },
            fit_state="balanced",
            history=[],
            remaining_actions=list(p28.ACTION_IDS),
            trial_index=0,
        )
        with self.assertRaisesRegex(p28.CredentialUnavailable, "fails closed"):
            p28.call_deepseek_action(observation=observation, api_key="")

    def test_live_policy_accepts_only_strict_allowlisted_json(self) -> None:
        observation = p28.build_policy_observation(
            support_buckets={
                "absent": 0,
                "very_low_1_to_5": 1,
                "low_6_to_20": 2,
                "medium_21_to_60": 3,
                "high_over_60": 3,
            },
            fit_state="overfit",
            history=[],
            remaining_actions=list(p28.ACTION_IDS),
            trial_index=0,
        )
        response = {
            "id": "live-response",
            "model": "deepseek-chat",
            "usage": {"total_tokens": 12},
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "action_id": "ACT_WELL_MASK_ONLY_858",
                                "reason_code": "remove_spatial_noise",
                                "stop": False,
                            }
                        )
                    }
                }
            ],
        }
        stream = io.BytesIO(json.dumps(response).encode("utf-8"))
        with mock.patch.object(p28.urllib.request, "urlopen", return_value=stream):
            decision, metadata = p28.call_deepseek_action(
                observation=observation,
                api_key="not-persisted",
                attempts=1,
            )
        self.assertEqual(decision["action_id"], "ACT_WELL_MASK_ONLY_858")
        self.assertTrue(metadata["valid"])
        self.assertNotIn("not-persisted", json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "unavailable action"):
            p28._validate_deepseek_decision(
                json.dumps(
                    {
                        "action_id": "NOT_ALLOWED",
                        "reason_code": "explore_alternative",
                        "stop": False,
                    }
                ),
                remaining_actions=p28.ACTION_IDS,
            )

    def test_auc_and_moment_lane_are_separate(self) -> None:
        self.assertAlmostEqual(p28.normalized_auc([-0.1, 0.03, 0.02]), 0.02)
        self.assertFalse(p28.MOMENT_PAIRED_LANE["main_effect_member"])
        self.assertNotIn("MOMENT", " ".join(p28.ACTION_IDS).upper())

    def test_a1_identity_is_an_actual_same_entrypoint_replay(self) -> None:
        def cell(outer_id: int, repeat_id: int) -> dict[str, object]:
            return {
                "outer_rollout_id": outer_id,
                "repeat_id": repeat_id,
                "seed": p28.REPEAT_SEEDS[repeat_id],
                "metrics": {
                    "fixed_schema_macro_f1": 0.2,
                    "per_class_f1": [0.0] * 9,
                },
                "prediction_sha256": f"prediction-{outer_id}-{repeat_id}",
            }

        a0_by_outer = {
            outer_id: [
                cell(outer_id, repeat_id)
                for repeat_id in range(len(p28.REPEAT_SEEDS))
            ]
            for outer_id in p28.OUTER_FOLDS
        }

        def replay(_arrays, outer_id, action):
            self.assertEqual(action, p28.A0)
            return [
                cell(outer_id, repeat_id)
                for repeat_id in range(len(p28.REPEAT_SEEDS))
            ]

        with mock.patch.object(
            p28, "_evaluate_promotion_action", side_effect=replay
        ) as execute:
            control, rows = p28._actual_a1_replay({}, a0_by_outer)
        self.assertEqual(execute.call_count, len(p28.OUTER_FOLDS))
        self.assertEqual(len(rows), 12)
        self.assertTrue(control["actual_replay_executed"])
        self.assertTrue(control["all_rosters_equal"])
        self.assertTrue(control["all_prediction_hashes_equal"])
        self.assertTrue(control["all_metric_hashes_equal"])
        self.assertEqual(
            control["executor_entrypoint"], "_evaluate_promotion_action"
        )
        source = inspect.getsource(p28.run_pilot)
        self.assertNotIn("a1_prediction_hash = a0_prediction_hash", source)
        self.assertNotIn("a1_metric_hash = a0_metric_hash", source)

    def test_exhaustive_ceiling_is_inner_only_and_never_promotes(self) -> None:
        class InnerOnlyEvaluator:
            def __init__(self) -> None:
                self.calls: list[tuple[int, str]] = []

            def inner(self, outer_id: int, action_id: str):
                self.calls.append((outer_id, action_id))
                offset = p28.ACTION_IDS.index(action_id) + 1
                score = 0.2 + 0.002 * offset
                return {
                    "selection_mean": score,
                    "per_class_selection_mean": [score] * 9,
                    "cells": [
                        {
                            "inner_fold_id": inner_id,
                            "repeat_id": repeat_id,
                            "seed": p28.REPEAT_SEEDS[repeat_id],
                            "selection_metrics": {
                                "fixed_schema_macro_f1": score,
                                "per_class_f1": [score] * 9,
                            },
                        }
                        for inner_id in range(3)
                        for repeat_id in range(3)
                    ],
                }

            def promotion(self, *_args, **_kwargs):
                raise AssertionError("inner ceiling must never call promotion")

        evaluator = InnerOnlyEvaluator()
        a0_inner = {
            outer_id: {"selection_mean": 0.2}
            for outer_id in p28.OUTER_FOLDS
        }
        rollouts = [
            {
                "outer_rollout_id": outer_id,
                "selected_for_promotion": p28.A0.action_id,
                "selected_inner_mean_local_evaluator_only": 0.2,
            }
            for outer_id in p28.OUTER_FOLDS
        ]
        control, records = p28._exhaustive_inner_ceiling(
            evaluator=evaluator,
            a0_inner=a0_inner,
            a2l_rollouts=rollouts,
        )
        self.assertEqual(len(evaluator.calls), 20)
        self.assertEqual(len(records), 20)
        self.assertEqual(control["actions_per_outer_rollout"], 5)
        self.assertFalse(control["used_for_policy_feedback"])
        self.assertFalse(control["used_for_promotion_selection"])
        self.assertFalse(control["promotion_metrics_computed"])
        self.assertEqual(control["failure_diagnosis"], "POLICY_SEARCH_LIMIT")
        self.assertTrue(
            all(not row["used_for_promotion_selection"] for row in records)
        )

    def test_a2l_and_a2d_can_differ_in_trajectory_but_share_endpoint(self) -> None:
        same = [
            "ACT_WELL_MASK_ONLY_858",
            "ACT_WEIGHT_EXP05_MEAN1",
            "ACT_PRIOR_SHRINK010",
        ]
        alternate = [
            "ACT_WELL_MASK_ONLY_858",
            "ACT_WEIGHT_EXP075_MEAN1",
            "ACT_PRIOR_SHRINK010",
        ]

        def rollout(outer_id: int, sequence: list[str]) -> dict[str, object]:
            return {
                "outer_rollout_id": outer_id,
                "trials": [{"action_id": action_id} for action_id in sequence],
                "selected_for_promotion": p28.A0.action_id,
            }

        a2l = [
            rollout(outer_id, alternate if outer_id in {1, 2} else same)
            for outer_id in p28.OUTER_FOLDS
        ]
        a2d = [rollout(outer_id, same) for outer_id in p28.OUTER_FOLDS]
        promotion = {
            "fixed_schema_macro_f1_mean": 0.2,
            "outer_fold_deltas": [0.0] * 4,
        }
        comparison = p28._trajectory_comparison(
            a2l, a2d, promotion, promotion
        )
        self.assertEqual(comparison["differing_trajectory_outer_folds"], [1, 2])
        self.assertEqual(comparison["differing_trajectory_count"], 2)
        self.assertTrue(comparison["promotion_endpoint_identical"])

    def test_holdout_like_inputs_are_rejected(self) -> None:
        for path in (
            Path("data/test.h5"),
            Path("known-holdout/data.npz"),
            Path("frozen/results.json"),
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "forbidden non-development"):
                    p28.ensure_development_only_paths((path,))
        p28.ensure_development_only_paths((Path("runtime/development_logo4.npz"),))


if __name__ == "__main__":
    unittest.main()
