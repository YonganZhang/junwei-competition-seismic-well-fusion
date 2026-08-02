from __future__ import annotations

import json
import hashlib
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

TRACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACK_DIR))

from fault_p29_agent_action_effect import (  # noqa: E402
    PROJECT_ROOT,
    SpyPredictorExecutor,
    build_scenarios,
    build_source_hashes,
    canonical_decision_hash,
    canonical_decision_hash_from_fields,
    run_p29,
    validate_action_stop_contract,
    validate_a2l_response,
    validate_evidence_tokens,
)


class FaultP29AgentActionEffectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_hashes = build_source_hashes()
        self.scenarios = build_scenarios(self.source_hashes)

    def tearDown(self) -> None:
        temp_root = TRACK_DIR / "_outputs" / "p29_agent_action_effect_test"
        if temp_root.exists():
            shutil.rmtree(temp_root)

    def test_allowlist_and_stop_validation_fail_closed(self) -> None:
        scenario = self.scenarios[0]
        validate_evidence_tokens(scenario, [])
        with self.assertRaises(RuntimeError):
            validate_evidence_tokens(scenario, ["hallucinated_token"])
        validate_action_stop_contract("PROCEED", False)
        validate_action_stop_contract("STOP_DATA_GATE", True)
        with self.assertRaises(RuntimeError):
            validate_action_stop_contract("PROCEED", True)
        with self.assertRaises(RuntimeError):
            validate_action_stop_contract("VERIFY_HASHES", False)

    def test_canonical_hash_replays_a0_and_is_order_stable(self) -> None:
        scenario = self.scenarios[0]
        a0 = {
            "scenario_id": scenario.scenario_id,
            "scenario_split": scenario.split,
            "policy_id": "A0_static_baseline",
            "selected_action_id": "STOP_DATA_GATE",
            "necessary_evidence": list(scenario.gold_necessary_evidence),
            "stop_requested": True,
        }
        shuffled = {
            "stop_requested": True,
            "necessary_evidence": list(scenario.gold_necessary_evidence),
            "selected_action_id": "STOP_DATA_GATE",
            "policy_id": "A1_advice_only",
            "scenario_split": scenario.split,
            "scenario_id": scenario.scenario_id,
        }
        self.assertEqual(canonical_decision_hash(a0), canonical_decision_hash(shuffled))
        self.assertEqual(
            canonical_decision_hash_from_fields(
                scenario_id=scenario.scenario_id,
                scenario_split=scenario.split,
                policy_id="A0_static_baseline",
                selected_action_id="STOP_DATA_GATE",
                necessary_evidence=scenario.gold_necessary_evidence,
                stop_requested=True,
            ),
            canonical_decision_hash(a0),
        )

    def test_spy_executor_contract(self) -> None:
        spy = SpyPredictorExecutor()
        self.assertEqual(spy.dispatch("PROCEED").dispatch_count, 1)
        self.assertEqual(spy.dispatch("PROCEED").dispatched_predictors, ("fault_gate_predictor_v1",))
        self.assertEqual(spy.dispatch("STOP_DATA_GATE").dispatch_count, 0)
        self.assertEqual(spy.dispatch("REQUEST_EVIDENCE").dispatch_count, 0)
        self.assertEqual(spy.dispatch("VERIFY_HASHES").dispatch_count, 0)

    def test_validate_a2l_response_rejects_bad_tokens_or_stop_flags(self) -> None:
        scenario = self.scenarios[1]
        with self.assertRaises(RuntimeError):
            validate_a2l_response(
                scenario,
                {
                    "action_id": "REQUEST_EVIDENCE",
                    "necessary_evidence": ["hallucinated_token"],
                    "stop_requested": True,
                    "confidence": 0.9,
                    "rationale": "bad token",
                },
            )
        with self.assertRaises(RuntimeError):
            validate_a2l_response(
                scenario,
                {
                    "action_id": "PROCEED",
                    "necessary_evidence": [],
                    "stop_requested": True,
                    "confidence": 0.9,
                    "rationale": "bad stop flag",
                },
            )

    def test_run_p29_writes_outputs_and_manifest_and_spy_effects(self) -> None:
        output_root = TRACK_DIR / "_outputs" / "p29_agent_action_effect_test"
        if output_root.exists():
            shutil.rmtree(output_root)
        with patch.dict(os.environ, {"DEEPSEEK_KEY": "present"}, clear=True), patch(
            "fault_p29_agent_action_effect._deepseek_chat_completion",
            side_effect=[
                {
                    "action_id": "STOP_DATA_GATE",
                    "necessary_evidence": [],
                    "stop_requested": True,
                    "confidence": 0.99,
                    "rationale": "stop",
                },
                {
                    "action_id": "REQUEST_EVIDENCE",
                    "necessary_evidence": [
                        "hash:_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/baseline_metrics.json",
                        "hash:_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/build_summary.json",
                    ],
                    "stop_requested": True,
                    "confidence": 0.91,
                    "rationale": "request",
                },
                {
                    "action_id": "VERIFY_HASHES",
                    "necessary_evidence": [
                        "verify:_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/baseline_metrics.json",
                    ],
                    "stop_requested": True,
                    "confidence": 0.93,
                    "rationale": "verify",
                },
                {
                    "action_id": "PROCEED",
                    "necessary_evidence": [],
                    "stop_requested": False,
                    "confidence": 0.95,
                    "rationale": "proceed",
                },
            ],
        ):
            result = run_p29(output_root)
        manifest_path = output_root / "manifest.json"
        action_effects_path = output_root / "action_effects.json"
        root_cause_path = output_root / "root_cause.md"
        for path in (
            output_root / "protocol.jsonl",
            output_root / "results.jsonl",
            output_root / "summary.json",
            action_effects_path,
            root_cause_path,
            manifest_path,
        ):
            self.assertTrue(path.is_file(), path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["outputs"]), 5)
        self.assertTrue(manifest["data_gate_blocked"])
        self.assertEqual(manifest["selection_promotion_intersection"], [])
        self.assertEqual(manifest["retained_policy"], "A2D_deterministic_agent")
        self.assertEqual(manifest["rejected_policy"], "A2L_llm_agent_execute")
        for entry in manifest["inputs"] + manifest["outputs"]:
            resolved = PROJECT_ROOT / entry["path"]
            self.assertTrue(resolved.is_file(), resolved)
            self.assertEqual(entry["sha256"], hashlib.sha256(resolved.read_bytes()).hexdigest())
        effects = json.loads(action_effects_path.read_text(encoding="utf-8"))
        self.assertEqual(effects["registered_predictors"], ["fault_gate_predictor_v1"])
        self.assertTrue(all(item["dispatch_count"] == 1 if item["selected_action_id"] == "PROCEED" else item["dispatch_count"] == 0 for item in effects["records"]))
        self.assertEqual(result["summary"]["retained_policy"], "A2D_deterministic_agent")
        self.assertEqual(result["summary"]["rejected_policy"], "A2L_llm_agent_execute")
        self.assertTrue(result["summary"]["data_gate_blocked"])


if __name__ == "__main__":
    unittest.main()
