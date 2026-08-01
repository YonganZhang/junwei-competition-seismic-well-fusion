from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


TRACK_DIR = Path(__file__).resolve().parents[1]
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

import lithofacies_p29_agent_action_effect as p29


def _metric(value: float, per_class: list[float] | None = None) -> dict:
    return {
        p29.PRIMARY_METRIC: float(value),
        "per_class_f1": list(per_class or [float(value)] * p29.NUM_CLASSES),
    }


def _inner_result(
    action_id: str,
    values: list[float],
    *,
    train_offset: float = 0.05,
    prediction_prefix: str | None = None,
) -> dict:
    cells = []
    for inner_id, value in enumerate(values):
        cells.append(
            {
                "inner_fold_id": inner_id,
                "selection_metrics": _metric(value),
                "train_metrics": _metric(value + train_offset),
                "prediction_sha256": f"{prediction_prefix or action_id}-i{inner_id}",
                "executor_state_sha256": f"state-{action_id}-i{inner_id}",
            }
        )
    return {
        "action_id": action_id,
        "config_sha256": f"config-{action_id}",
        "cells": cells,
        "selection_mean": float(np.mean(values)),
        "train_mean": float(np.mean(values) + train_offset),
        "prediction_sha256": f"inner-{prediction_prefix or action_id}",
        "executor_state_sha256": f"inner-state-{action_id}",
    }


def _outer_result(action_id: str, outer_id: int, value: float) -> dict:
    return {
        "outer_rollout_id": outer_id,
        "model_seed": p29.MODEL_SEED,
        "action_id": action_id,
        "config_sha256": f"config-{action_id}",
        "executor_state_sha256": f"outer-state-{action_id}-{outer_id}",
        "metrics": _metric(value),
        "prediction_sha256": f"outer-{action_id}-{outer_id}",
    }


class FakeEvaluator:
    def __init__(self) -> None:
        self.arrays = {}
        self.inner_deltas = {
            action_id: (index + 1) * 0.001
            for index, action_id in enumerate(p29.PILOT_ACTION_IDS)
        }
        self.outer_deltas = {
            action_id: (index - 1) * 0.002
            for index, action_id in enumerate(p29.PILOT_ACTION_IDS)
        }

    def inner(self, outer_id: int, action_id: str) -> dict:
        base = 0.15 + outer_id * 0.01
        delta = 0.0 if action_id == p29.A0.action_id else self.inner_deltas[action_id]
        return _inner_result(
            action_id,
            [base + delta - 0.001, base + delta, base + delta + 0.001],
        )

    def outer(self, outer_id: int, action_id: str) -> dict:
        base = 0.20 + outer_id * 0.01
        delta = 0.0 if action_id == p29.A0.action_id else self.outer_deltas[action_id]
        return _outer_result(action_id, outer_id, base + delta)


class P29ObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arrays = {}
        for inner_id in range(3):
            self.arrays[f"o0_i{inner_id}_class_counts"] = np.asarray(
                [1 + inner_id, 30, 2, 10, 5, 40, 45, 1, 0],
                dtype=np.int64,
            )
        self.baseline = _inner_result(
            p29.A0.action_id,
            [0.14, 0.15, 0.16],
        )
        candidate = _inner_result(
            p29.PILOT_ACTION_IDS[0],
            [0.145, 0.158, 0.154],
        )
        self.effect = p29.summarize_inner_effect(candidate, self.baseline)

    def test_enhanced_observation_has_only_safe_normalized_information(self) -> None:
        observation = p29.build_enhanced_observation(
            arrays=self.arrays,
            outer_id=0,
            baseline=self.baseline,
            history=[
                {
                    "action_id": p29.PILOT_ACTION_IDS[0],
                    "effect": self.effect,
                    "feedback": "improved",
                }
            ],
            remaining_actions=p29.PILOT_ACTION_IDS[1:],
            trial_index=1,
        )
        p29.assert_policy_payload_safe(observation)
        self.assertEqual(observation["observation_schema"], p29.OBSERVATION_SCHEMA)
        self.assertEqual(len(observation["class_support"]), 9)
        self.assertIn("normalized_delta_units", observation["history"][0]["effect"])
        self.assertIn("uncertainty_units", observation["history"][0]["effect"])
        serialized = json.dumps(observation, sort_keys=True).lower()
        for forbidden in ("raw_metric", "label", "residual", "promotion"):
            self.assertNotIn(forbidden, serialized)

    def test_categorical_prompt_ablation_removes_all_float_observables(self) -> None:
        observation = p29.build_categorical_ablation_observation(
            arrays=self.arrays,
            outer_id=0,
            baseline=self.baseline,
            history=[
                {
                    "action_id": p29.PILOT_ACTION_IDS[0],
                    "effect": self.effect,
                    "feedback": "improved",
                }
            ],
            remaining_actions=p29.PILOT_ACTION_IDS[1:],
            trial_index=1,
        )
        self.assertEqual(
            observation["observation_schema"], p29.CATEGORICAL_ABLATION_SCHEMA
        )
        self.assertFalse(
            any(isinstance(value, float) for value in p29._walk_values(observation))
        )

    def test_firewall_rejects_raw_unbounded_identity_and_path_fields(self) -> None:
        with self.assertRaises(ValueError):
            p29.assert_policy_payload_safe({"raw_metric": 0.2})
        with self.assertRaises(ValueError):
            p29.assert_policy_payload_safe({"safe_effect": 5.0})
        with self.assertRaises(ValueError):
            p29.assert_policy_payload_safe({"safe_effect": "/tmp/value"})
        with self.assertRaises(ValueError):
            p29.ensure_development_only_paths([Path("test.h5")])


class P29EffectTests(unittest.TestCase):
    def test_fold_uncertainty_uses_three_real_inner_folds(self) -> None:
        baseline = _inner_result(p29.A0.action_id, [0.10, 0.20, 0.30])
        action = _inner_result(p29.PILOT_ACTION_IDS[0], [0.11, 0.19, 0.33])
        effect = p29.summarize_inner_effect(action, baseline)
        expected = [0.01, -0.01, 0.03]
        self.assertTrue(np.allclose(effect["inner_fold_deltas"], expected))
        self.assertGreater(effect["standard_error"], 0.0)
        self.assertEqual(sum(effect["inner_fold_outcomes"].values()), 3)

    def test_action_noop_detector_flags_same_prediction_hashes(self) -> None:
        entries = [
            {
                "action_id": p29.A0.action_id,
                "config_sha256": "a0-config",
                "inner_prediction_sha256": "same-inner",
                "outer_prediction_sha256": "same-outer",
            },
            {
                "action_id": p29.PILOT_ACTION_IDS[0],
                "config_sha256": "different-config",
                "inner_prediction_sha256": "same-inner",
                "outer_prediction_sha256": "same-outer",
            },
        ]
        flags = p29.mark_action_effect_flags(entries)
        self.assertTrue(entries[1]["effective_action_noop"])
        self.assertTrue(flags["all_nonbaseline_configs_differ"])
        self.assertFalse(flags["all_nonbaseline_actions_change_prediction"])

    def test_action_effect_hashes_and_transfer_matrix_are_diagnostic_only(self) -> None:
        evaluator = FakeEvaluator()
        a0_inner = {
            outer_id: evaluator.inner(outer_id, p29.A0.action_id)
            for outer_id in p29.OUTER_FOLDS
        }
        a0_outer = {
            outer_id: evaluator.outer(outer_id, p29.A0.action_id)
            for outer_id in p29.OUTER_FOLDS
        }
        rollouts = [
            {
                "outer_rollout_id": outer_id,
                "selected_for_promotion": p29.A0.action_id,
                "selected_inner_delta": 0.0,
            }
            for outer_id in p29.OUTER_FOLDS
        ]
        effects, transfer, oracle, rows = p29._action_effects_and_transfer(
            evaluator=evaluator,
            a0_inner=a0_inner,
            a0_outer=a0_outer,
            a2l_rollouts=rollouts,
        )
        self.assertEqual(len(effects["actions"]), 5)
        self.assertEqual(len(rows), 16)
        self.assertTrue(transfer["computed_after_all_policy_calls"])
        self.assertFalse(transfer["used_for_policy_feedback"])
        self.assertFalse(transfer["used_for_legal_promotion_selection"])
        self.assertEqual(oracle["layer"], "inner_selection_only")


class P29ContractTests(unittest.TestCase):
    def test_protocol_preserves_metric_split_and_collapses_model_seed(self) -> None:
        protocol = p29._protocol_payload("f" * 64)
        self.assertEqual(protocol["primary_metric"], "fixed_schema_macro_f1")
        self.assertEqual(protocol["split_hash"], p29.EXPECTED_SPLIT_HASH)
        self.assertEqual(protocol["model_seeds"], [p29.MODEL_SEED])
        self.assertEqual(protocol["pilot_action_count"], 4)
        self.assertEqual(protocol["trial_budget_per_outer_rollout"], 3)
        self.assertEqual(
            protocol["uncertainty_unit"], "three_disjoint_inner_logo_folds"
        )

    def test_actual_a1_replay_calls_real_entrypoint_once_per_outer_fold(self) -> None:
        a0_outer = {
            outer_id: _outer_result(p29.A0.action_id, outer_id, 0.2 + outer_id * 0.01)
            for outer_id in p29.OUTER_FOLDS
        }
        with mock.patch.object(
            p29,
            "_evaluate_outer_action",
            side_effect=[a0_outer[outer_id] for outer_id in p29.OUTER_FOLDS],
        ) as entrypoint:
            control, rows = p29._actual_a1_replay({}, a0_outer)
        self.assertEqual(entrypoint.call_count, 4)
        self.assertEqual(len(rows), 4)
        self.assertTrue(control["all_equal"])

    def test_missing_credential_fails_closed(self) -> None:
        with self.assertRaises(p29.CredentialUnavailable):
            p29.call_deepseek_action(
                observation={
                    "observation_schema": p29.CATEGORICAL_ABLATION_SCHEMA,
                    "available_actions": [
                        {"action_id": p29.PILOT_ACTION_IDS[0], "description": "safe"}
                    ],
                },
                api_key="",
            )

    def test_pilot_action_configs_are_distinct_from_a0(self) -> None:
        a0_hash = p29._stable_hash(p29._config_payload(p29.A0))
        action_hashes = {
            p29._stable_hash(p29._config_payload(action))
            for action in p29.PILOT_ACTIONS
        }
        self.assertEqual(len(action_hashes), 4)
        self.assertNotIn(a0_hash, action_hashes)

    def test_normalized_auc_uses_exact_budget(self) -> None:
        self.assertTrue(math.isclose(p29.normalized_auc([0.0, 0.005, 0.01]), 1.0))
        with self.assertRaises(ValueError):
            p29.normalized_auc([0.0, 0.005])


if __name__ == "__main__":
    unittest.main()
