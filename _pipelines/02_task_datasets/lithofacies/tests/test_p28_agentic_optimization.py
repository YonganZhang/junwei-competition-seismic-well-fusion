from __future__ import annotations

import io
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
