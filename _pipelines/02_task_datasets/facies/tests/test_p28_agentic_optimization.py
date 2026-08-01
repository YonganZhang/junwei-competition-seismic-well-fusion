from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

import p28_agentic_optimization as p28  # noqa: E402


class P28ProtocolTests(unittest.TestCase):
    def test_allowlist_is_pretrained_single_factor_and_excludes_control(self) -> None:
        self.assertEqual(len(p28.ACTION_ALLOWLIST), 5)
        self.assertNotIn("continued_cnn_control", p28.ACTION_ALLOWLIST)
        baseline = p28.asdict(p28.A0_CONFIG)
        for action_id, config in p28.ACTION_ALLOWLIST.items():
            changed = {
                key
                for key, value in p28.asdict(config).items()
                if key not in {"action_id", "changed_factor", "description"}
                and value != baseline[key]
            }
            self.assertEqual(changed, {config.changed_factor})
            self.assertEqual(config.action_id, action_id)

    def test_observation_rejects_raw_metric_and_paths(self) -> None:
        p28._assert_safe_observation({"loss_level": "moderate"})
        with self.assertRaisesRegex(ValueError, "denied observation key"):
            p28._assert_safe_observation({"raw_miou": 0.2})
        with self.assertRaisesRegex(ValueError, "path-like"):
            p28._assert_safe_observation({"source": "/tmp/dev.json"})

    def test_strict_decision_and_fixed_four_trial_budget(self) -> None:
        decision = p28._validate_decision(
            {
                "action_id": "FAC_GATE_035",
                "confidence": 0.7,
                "rationale": "Moderate gate diagnostic.",
                "stop": False,
            },
            allowed=p28.ACTION_IDS,
            round_id=1,
        )
        self.assertEqual(decision["action_id"], "FAC_GATE_035")
        with self.assertRaisesRegex(ValueError, "outside"):
            p28._validate_decision(
                {
                    "action_id": "ARBITRARY_SHELL",
                    "confidence": 1.0,
                    "rationale": "invalid",
                    "stop": False,
                },
                allowed=p28.ACTION_IDS,
                round_id=1,
            )
        self.assertEqual(p28.TRIALS_PER_POLICY, 4)

    def test_missing_provider_fails_closed(self) -> None:
        observation = {
            "available_action_ids": list(p28.ACTION_IDS),
            "round": 1,
        }
        with mock.patch.dict(os.environ, {}, clear=True):
            response = p28._call_deepseek(observation, advice_only=False)
        self.assertEqual(response["status"], "BLOCKED_PROVIDER")
        self.assertFalse(response["credential_persisted"])
        self.assertNotIn("decision", response)

    def test_pcg64_is_without_replacement(self) -> None:
        first = p28.np.random.Generator(p28.np.random.PCG64(p28.ROOT_SEED))
        second = p28.np.random.Generator(p28.np.random.PCG64(p28.ROOT_SEED))
        left = [
            str(value)
            for value in first.choice(
                p28.ACTION_IDS,
                size=p28.TRIALS_PER_POLICY,
                replace=False,
            )
        ]
        right = [
            str(value)
            for value in second.choice(
                p28.ACTION_IDS,
                size=p28.TRIALS_PER_POLICY,
                replace=False,
            )
        ]
        self.assertEqual(left, right)
        self.assertEqual(len(set(left)), p28.TRIALS_PER_POLICY)

    def test_a1_is_a_true_same_config_replay(self) -> None:
        def package(phase: str) -> dict[str, object]:
            return {
                "phase": phase,
                "tasks": {
                    "F3": {"miou": 0.2, "prediction_hash": f"f3-{phase}"},
                    "Penobscot": {
                        "miou": 0.1,
                        "prediction_hash": f"pen-{phase}",
                    },
                },
                "equal_mean": 0.15,
            }

        a0 = {
            phase: package(phase)
            for phase in ("selection", "promotion")
        }
        diagnostics = {
            task: {
                "loss_level": "moderate",
                "gradient_norm": "moderate",
                "sam2_update": "moderate",
                "attention_entropy": "moderate",
                "fusion_scale_state": "stuck",
            }
            for task in ("F3", "Penobscot")
        }
        with (
            mock.patch.object(
                p28,
                "_call_deepseek",
                return_value={"status": "OK", "decision": None},
            ),
            mock.patch.object(
                p28,
                "_run_config_package",
                side_effect=lambda **kwargs: package(kwargs["phase"]),
            ) as replay,
        ):
            result = p28._a1_advice(
                a0=a0,
                a0_diagnostics=diagnostics,
                states={},
                device="cpu",
            )
        self.assertEqual(replay.call_count, 2)
        self.assertEqual(
            result["replay_kind"],
            "fresh_same_config_seed_fold_reexecution",
        )
        self.assertTrue(result["prediction_hash_equal_a0"])
        self.assertTrue(result["metrics_equal_a0"])

    def test_path_score_is_running_max_mean_miou_not_endpoint(self) -> None:
        trials = [
            {"equal_mean": value}
            for value in (0.1, 0.3, 0.25, 0.4)
        ]
        score = p28._optimization_path_score_running_max_mean_miou(
            trials,
            baseline=0.2,
        )
        self.assertAlmostEqual(score, 0.3)


class P28EvidenceTests(unittest.TestCase):
    def test_generated_pilot_evidence_verifies(self) -> None:
        if not p28.OUTPUT_ROOT.is_dir():
            self.skipTest("P28 evidence is generated by the bounded pilot")
        verified = p28.verify()
        self.assertEqual(verified["artifacts"], 4)
        self.assertFalse(verified["frozen_test_accessed"])
        self.assertFalse(verified["credential_persisted"])
        summary = json.loads(
            (p28.OUTPUT_ROOT / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            summary["runtime"]["runner_sha256"],
            p28.p11._sha256(Path(p28.__file__)),
        )
        self.assertEqual(
            summary["a1"]["prediction_hashes"],
            summary["a1"]["a0_prediction_hashes"],
        )
        self.assertTrue(summary["a1"]["metrics_equal_a0"])
        self.assertEqual(
            summary["a1"]["replay_kind"],
            "fresh_same_config_seed_fold_reexecution",
        )
        for policy_key in ("a2l", "a2d", "a3"):
            self.assertEqual(
                len(summary[policy_key]["selection_trials"]),
                p28.TRIALS_PER_POLICY,
            )
            for trial in summary[policy_key]["selection_trials"]:
                self.assertLessEqual(
                    trial["runtime_s"],
                    p28.ACTION_WALL_CLOCK_BUDGET_S,
                )
            policy = summary[policy_key]
            self.assertNotIn("optimization_auc", policy)
            self.assertIn(
                "optimization_path_score_running_max_mean_mIoU",
                policy,
            )
            self.assertEqual(
                policy["endpoint_promotion_mean_mIoU"],
                policy["promotion"]["equal_mean"],
            )


if __name__ == "__main__":
    unittest.main()
