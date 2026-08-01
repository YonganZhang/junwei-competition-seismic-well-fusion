from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

import p29_agent_action_effect as p29  # noqa: E402


def _task(
    miou: float,
    prediction_hash: str,
    *,
    gradient: float = 4.0,
    initial: float = 0.2,
    final: float = 0.201,
) -> dict[str, object]:
    return {
        "miou": miou,
        "macro_f1": miou,
        "accuracy": miou,
        "prediction_hash": prediction_hash,
        "train_loss_mean": 1.0,
        "train_loss_last": 1.0,
        "last_grad_norm": gradient,
        "sam2_update_l2": 0.2,
        "sam2_trainable_parameters": 1,
        "sam2_trainable_blocks": [22, 23],
        "attention_entropy": 0.9,
        "train_attention_entropy": 0.9,
        "fusion_scale_initial": initial,
        "fusion_scale": final,
    }


def _package(
    action_id: str,
    f3: float,
    pen: float,
    *,
    round_id: int = 1,
) -> dict[str, object]:
    config = (
        p29.A0_CONFIG
        if action_id == p29.A0_CONFIG.action_id
        else p29.ACTION_ALLOWLIST[action_id]
    )
    return {
        "policy_id": "test",
        "round": round_id,
        "phase": "selection",
        "action_id": action_id,
        "config": p29.asdict(config),
        "config_hash": p29._hash_payload(p29.asdict(config)),
        "tasks": {
            "F3": _task(f3, f"f3-{action_id}", gradient=6.0),
            "Penobscot": _task(pen, f"pen-{action_id}", gradient=3.0),
        },
        "equal_mean": (f3 + pen) / 2.0,
        "runtime_s": 0.1,
        "wall_clock_budget_s": p29.ACTION_WALL_CLOCK_BUDGET_S,
        "exit_status": "OK",
        "frozen_test_accessed": False,
    }


class P29ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        selection = _package(p29.A0_CONFIG.action_id, 0.30, 0.25, round_id=0)
        promotion = _package(p29.A0_CONFIG.action_id, 0.31, 0.26, round_id=0)
        promotion["phase"] = "promotion"
        self.a0 = {"selection": selection, "promotion": promotion}
        self.diagnostics = p29._baseline_diagnostics(self.a0)

    def test_registry_is_the_same_five_single_factor_actions_as_p28(self) -> None:
        self.assertEqual(p29.ACTION_ALLOWLIST, p29.p28.ACTION_ALLOWLIST)
        self.assertEqual(len(p29.ACTION_ALLOWLIST), 5)
        baseline = p29.asdict(p29.A0_CONFIG)
        for action_id, config in p29.ACTION_ALLOWLIST.items():
            changed = {
                key
                for key, value in p29.asdict(config).items()
                if key not in {"action_id", "changed_factor", "description"}
                and value != baseline[key]
            }
            self.assertEqual(changed, {config.changed_factor})
            self.assertEqual(action_id, config.action_id)

    def test_observation_v2_has_previous_action_full_history_and_safe_effects(self) -> None:
        candidate = _package("FAC_GATE_050", 0.35, 0.24)
        history = [
            p29._history_entry(
                candidate,
                self.a0["selection"],
                stop_requested=False,
            )
        ]
        observation = p29._build_observation_v2(
            policy_id="A2L_llm_agent_execute",
            round_id=2,
            available_action_ids=("FAC_GATE_035",),
            baseline_diagnostics=self.diagnostics,
            history=history,
        )
        self.assertEqual(observation["schema_version"], p29.OBSERVATION_SCHEMA_VERSION)
        self.assertEqual(observation["previous_action"]["action_id"], "FAC_GATE_050")
        self.assertEqual(observation["full_action_history"], history)
        effect = history[0]["per_dataset_effects"]["F3"]
        self.assertIn("optimizer_gradient_movement", effect)
        self.assertIn("fusion_gate_movement", effect)
        self.assertGreater(effect["effect_signal"]["normalized_signed_delta"], 0)
        p29._assert_safe_observation(observation)

    def test_observation_firewall_rejects_raw_score_label_and_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "denied observation key"):
            p29._assert_safe_observation({"raw_metric": 0.3})
        with self.assertRaisesRegex(ValueError, "denied observation key"):
            p29._assert_safe_observation({"labels": [1, 2]})
        with self.assertRaisesRegex(ValueError, "path-like"):
            p29._assert_safe_observation({"source": "/tmp/fold.json"})
        with self.assertRaisesRegex(ValueError, "safe range"):
            p29._assert_safe_observation({"normalized_signed_delta": 3.0})

    def test_stop_executes_selected_action_then_terminates(self) -> None:
        candidate = _package("FAC_GATE_050", 0.35, 0.24)
        endpoint = {
            "policy_id": "A2L_llm_agent_execute",
            "action_by_dataset": {
                "F3": "FAC_GATE_050",
                "Penobscot": p29.A0_CONFIG.action_id,
            },
            "tasks": {
                "F3": _task(0.36, "f3-end"),
                "Penobscot": _task(0.26, "pen-end"),
            },
            "equal_mean": 0.31,
        }
        response = {
            "status": "OK",
            "decision": {
                "action_id": "FAC_GATE_050",
                "confidence": 0.8,
                "rationale": "Execute then stop.",
                "stop": True,
            },
            "provider": "deepseek",
            "model_requested": "deepseek-chat",
            "model_returned": "deepseek-chat",
            "response_id": "response",
            "usage": {},
        }
        with (
            mock.patch.object(p29, "_call_deepseek", return_value=response),
            mock.patch.object(
                p29.p28,
                "_run_config_package",
                return_value=candidate,
            ) as executor,
            mock.patch.object(
                p29,
                "_run_per_dataset_package",
                return_value=endpoint,
            ),
        ):
            result = p29._run_policy(
                policy_id="A2L_llm_agent_execute",
                states={},
                device="cpu",
                a0=self.a0,
                baseline_diagnostics=self.diagnostics,
            )
        self.assertEqual(executor.call_count, 1)
        self.assertEqual(result["executed_action_count"], 1)
        self.assertEqual(result["stop_after_round"], 1)
        self.assertEqual(result["status"], "STOPPED_AFTER_EXECUTION")
        self.assertEqual(len(result["decisions"]), 1)

    def test_dataset_package_can_keep_a0_for_penobscot(self) -> None:
        gate = _package("FAC_GATE_050", 0.36, 0.24)
        dice = _package("FAC_DICE_050", 0.32, 0.23, round_id=2)
        selected = p29._select_per_dataset_actions(
            a0_selection=self.a0["selection"], trials=(gate, dice)
        )
        self.assertEqual(selected["F3"], "FAC_GATE_050")
        self.assertEqual(selected["Penobscot"], p29.A0_CONFIG.action_id)

    def test_dataset_package_keeps_a0_for_subthreshold_positive_delta(self) -> None:
        gate = _package("FAC_GATE_050", 0.36, 0.253)
        selected = p29._select_per_dataset_actions(
            a0_selection=self.a0["selection"], trials=(gate,)
        )
        self.assertEqual(selected["F3"], "FAC_GATE_050")
        self.assertEqual(selected["Penobscot"], p29.A0_CONFIG.action_id)

    def test_two_step_sample_efficiency_is_per_dataset_and_padded_on_stop(self) -> None:
        gate = _package("FAC_GATE_050", 0.36, 0.24)
        score, trace = p29._sample_efficiency_path_score(
            a0_selection=self.a0["selection"], trials=(gate,)
        )
        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[0], trace[1])
        self.assertAlmostEqual(score, (0.36 + 0.25) / 2.0)

    def test_promotion_guards_apply_per_dataset_and_to_continued_cnn(self) -> None:
        policy = {
            "status": "OK",
            "promotion": {
                "tasks": {
                    "F3": {"miou": 0.36},
                    "Penobscot": {"miou": 0.26},
                },
                "equal_mean": 0.31,
            },
        }
        control = {
            "promotion": {
                "tasks": {
                    "F3": {"miou": 0.20},
                    "Penobscot": {"miou": 0.20},
                },
                "equal_mean": 0.20,
            }
        }
        diagnostics = p29._promotion_guard_diagnostics(
            policy=policy,
            a0=self.a0,
            control=control,
        )
        self.assertTrue(diagnostics["passed"])
        self.assertTrue(all(diagnostics["checks"].values()))

    def test_prompt_ablation_removes_numeric_history_and_movement(self) -> None:
        candidate = _package("FAC_GATE_050", 0.35, 0.24)
        history = [
            p29._history_entry(candidate, self.a0["selection"], stop_requested=False)
        ]
        ablated = p29._build_observation_ablation(
            policy_id="A2L_v1_prompt_ablation",
            round_id=2,
            available_action_ids=("FAC_GATE_035",),
            baseline_diagnostics=self.diagnostics,
            history=history,
        )
        serialized = json.dumps(ablated)
        self.assertNotIn("full_action_history", serialized)
        self.assertNotIn("normalized_signed_delta", serialized)
        self.assertNotIn("optimizer_gradient_movement", serialized)
        self.assertEqual(ablated["most_recent_direction_only"]["F3"], "improved")

    def test_formal_metric_support_records_all_configured_classes(self) -> None:
        states = {}
        for phase, fold in (("selection", 0), ("promotion", 4)):
            for task_id in p29.TASKS:
                prepared = SimpleNamespace(
                    validation_labels=np.asarray([[[0, 1], [1, 0]]]),
                    num_classes=2,
                    fold_id=fold,
                )
                states[(phase, task_id)] = (prepared, object())
        support = p29._formal_metric_support(states)
        self.assertEqual(len(support["cells"]), 4)
        self.assertTrue(all(cell["all_classes_supported"] for cell in support["cells"].values()))
        self.assertTrue(support["formal_metric_support_hash"])

    def test_missing_provider_fails_closed(self) -> None:
        observation = {
            "available_action_ids": list(p29.ACTION_IDS),
            "round": 1,
        }
        with mock.patch.dict(os.environ, {}, clear=True):
            response = p29._call_deepseek(observation, advice_only=False)
        self.assertEqual(response["status"], "BLOCKED_PROVIDER")
        self.assertFalse(response["credential_persisted"])
        self.assertNotIn("decision", response)

    def test_pcg64_two_of_five_is_deterministic_without_replacement(self) -> None:
        first = np.random.Generator(np.random.PCG64(p29.ROOT_SEED))
        second = np.random.Generator(np.random.PCG64(p29.ROOT_SEED))
        left = first.choice(p29.ACTION_IDS, size=2, replace=False).tolist()
        right = second.choice(p29.ACTION_IDS, size=2, replace=False).tolist()
        self.assertEqual(left, right)
        self.assertEqual(len(set(left)), 2)


class P29EvidenceTests(unittest.TestCase):
    def test_generated_p29_evidence_verifies(self) -> None:
        if not p29.OUTPUT_ROOT.is_dir():
            self.skipTest("P29 evidence is generated by the bounded live pilot")
        verified = p29.verify()
        self.assertEqual(verified["artifacts"], 6)
        self.assertFalse(verified["frozen_test_accessed"])
        self.assertFalse(verified["credential_persisted"])
        summary = json.loads(
            (p29.OUTPUT_ROOT / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            summary["runtime"]["runner_sha256"],
            p29.p11._sha256(Path(p29.__file__)),
        )
        self.assertEqual(set(summary["action_effects"]["actions"]), set(p29.ACTION_IDS))
        self.assertTrue(summary["action_noop_check_passed"])
        self.assertEqual(
            summary["a2l"]["promotion"]["action_by_dataset"],
            summary["a2l"]["selected_per_dataset_package"],
        )


if __name__ == "__main__":
    unittest.main()
