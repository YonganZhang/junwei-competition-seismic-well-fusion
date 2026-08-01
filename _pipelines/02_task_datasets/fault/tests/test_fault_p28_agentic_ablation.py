from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

TRACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACK_DIR))

from fault_p28_agentic_ablation import (  # noqa: E402
    ACTION_REGISTRY,
    advice_only_policy,
    build_scenarios,
    conservative_policy,
    PROJECT_ROOT,
    git_head,
    random_policy,
    run_ablation,
    select_retained_policy,
    sha256_file,
)


class FaultP28AgenticAblationTests(unittest.TestCase):
    def test_scenarios_are_fixed_auditable_and_split_disjoint(self) -> None:
        source_hashes = {
            name: {"path": f"_fake/{name}", "sha256": f"sha256-{name}"}
            for name in (
                "fault_p18_cigbench.py",
                "test_fault_p18_cigbench.py",
                "tests/test_fault_p28_agentic_ablation.py",
                "baseline_metrics.json",
                "build_summary.json",
                "p18_evidence.md",
                "historical_baseline_manifest.json",
            )
        }
        scenarios = build_scenarios(source_hashes)
        self.assertEqual(len(scenarios), 4)
        self.assertEqual(
            [scenario.scenario_id for scenario in scenarios],
            [
                "observed_blocked_current",
                "packet_hash_missing",
                "packet_hash_mismatch",
                "counterfactual_contract_green",
            ],
        )
        self.assertEqual(
            {scenario.scenario_id for scenario in scenarios if scenario.split == "selection"},
            {"observed_blocked_current", "packet_hash_missing"},
        )
        self.assertEqual(
            {scenario.scenario_id for scenario in scenarios if scenario.split == "promotion"},
            {"packet_hash_mismatch", "counterfactual_contract_green"},
        )
        for scenario in scenarios:
            self.assertEqual(
                set(scenario.gates),
                {
                    "contiguous_3d_development_blocks",
                    "coverage_audited_verified_background",
                    "explicit_unknown_mask_provenance",
                    "group_isolated_development_split",
                },
            )
            self.assertEqual(len(scenario.packet_items), 4)

    def test_conservative_a0_and_advice_only_a1_share_decision_hash(self) -> None:
        source_hashes = {
            name: {"path": f"_fake/{name}", "sha256": f"sha256-{name}"}
            for name in (
                "fault_p18_cigbench.py",
                "test_fault_p18_cigbench.py",
                "tests/test_fault_p28_agentic_ablation.py",
                "baseline_metrics.json",
                "build_summary.json",
                "p18_evidence.md",
                "historical_baseline_manifest.json",
            )
        }
        scenario = build_scenarios(source_hashes)[0]
        a0 = conservative_policy(scenario)
        a1 = advice_only_policy(scenario, a0)
        self.assertEqual(a0["decision_hash"], a1["decision_hash"])
        self.assertEqual(a0["selected_action_id"], "STOP_DATA_GATE")
        self.assertEqual(a0["decision_status"], "OK")

    def test_random_policy_uses_registry_and_is_different_from_deterministic_choice(self) -> None:
        source_hashes = {
            name: {"path": f"_fake/{name}", "sha256": f"sha256-{name}"}
            for name in (
                "fault_p18_cigbench.py",
                "test_fault_p18_cigbench.py",
                "tests/test_fault_p28_agentic_ablation.py",
                "baseline_metrics.json",
                "build_summary.json",
                "p18_evidence.md",
                "historical_baseline_manifest.json",
            )
        }
        scenario = build_scenarios(source_hashes)[1]
        deterministic = conservative_policy(scenario)
        random_decision = random_policy(scenario, np.random.Generator(np.random.PCG64(2693)))
        self.assertIn(random_decision["selected_action_id"], ACTION_REGISTRY)
        self.assertNotEqual(random_decision["decision_hash"], deterministic["decision_hash"])

    def test_blocked_provider_when_key_is_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            source_hashes = {
                name: {"path": f"_fake/{name}", "sha256": f"sha256-{name}"}
                for name in (
                    "fault_p18_cigbench.py",
                    "test_fault_p18_cigbench.py",
                    "tests/test_fault_p28_agentic_ablation.py",
                    "baseline_metrics.json",
                    "build_summary.json",
                    "p18_evidence.md",
                    "historical_baseline_manifest.json",
                )
            }
            scenario = build_scenarios(source_hashes)[0]
            from fault_p28_agentic_ablation import _run_a2l

            result = _run_a2l(scenario)
            self.assertEqual(result["decision_status"], "BLOCKED_PROVIDER")
            self.assertIsNone(result["selected_action_id"])
            self.assertEqual(result["decision_hash"], None)

    def test_a2l_validates_strict_json_when_provider_is_mocked(self) -> None:
        source_hashes = {
            name: {"path": f"_fake/{name}", "sha256": f"sha256-{name}"}
            for name in (
                "fault_p18_cigbench.py",
                "test_fault_p18_cigbench.py",
                "tests/test_fault_p28_agentic_ablation.py",
                "baseline_metrics.json",
                "build_summary.json",
                "p18_evidence.md",
                "historical_baseline_manifest.json",
            )
        }
        scenario = build_scenarios(source_hashes)[2]
        fake_payload = {
            "action_id": "VERIFY_HASHES",
            "necessary_evidence": ["verify:_fake/baseline_metrics.json"],
            "stop_requested": True,
            "confidence": 0.87,
            "rationale": "hash mismatch must be verified",
        }
        with patch("fault_p28_agentic_ablation._deepseek_chat_completion", return_value=fake_payload), patch.dict(
            os.environ,
            {"DEEPSEEK_KEY": "present"},
            clear=True,
        ):
            from fault_p28_agentic_ablation import _run_a2l

            result = _run_a2l(scenario)
            self.assertEqual(result["decision_status"], "OK")
            self.assertEqual(result["selected_action_id"], "VERIFY_HASHES")
            self.assertEqual(result["necessary_evidence"], ["verify:_fake/baseline_metrics.json"])
            self.assertTrue(result["stop_requested"])

    def test_retention_prefers_a2d_when_it_noninferiorly_dominates_a2l(self) -> None:
        policy_summaries = {
            "A2L_llm_agent_execute": {
                "decision_accuracy": 1.0,
                "necessary_evidence_f1": 0.0,
                "dangerous_false_release_rate": 0.0,
            },
            "A2D_deterministic_agent": {
                "decision_accuracy": 1.0,
                "necessary_evidence_f1": 1.0,
                "dangerous_false_release_rate": 0.0,
            },
        }
        retain_policy, reject_policy = select_retained_policy(policy_summaries)
        self.assertEqual(retain_policy, "A2D_deterministic_agent")
        self.assertEqual(reject_policy, "A2L_llm_agent_execute")

    def test_run_ablation_writes_protocol_results_summary_evidence_and_manifest(self) -> None:
        output_root = Path("_outputs") / "p28_agentic_ablation_test"
        if output_root.exists():
            for child in output_root.iterdir():
                if child.is_file():
                    child.unlink()
            output_root.rmdir()
        with patch.dict(
            os.environ,
            {"DEEPSEEK_KEY": "present"},
            clear=True,
        ), patch(
            "fault_p28_agentic_ablation._deepseek_chat_completion",
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
                    "necessary_evidence": [],
                    "stop_requested": True,
                    "confidence": 0.91,
                    "rationale": "request",
                },
                {
                    "action_id": "VERIFY_HASHES",
                    "necessary_evidence": [],
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
            result = run_ablation(output_root)
            protocol_path = output_root / "protocol.jsonl"
            results_path = output_root / "results.jsonl"
            summary_path = output_root / "summary.json"
            evidence_path = output_root / "evidence.md"
            manifest_path = output_root / "manifest.json"
            for path in (protocol_path, results_path, summary_path, evidence_path, manifest_path):
                self.assertTrue(path.is_file(), path)
            protocol = [json.loads(line) for line in protocol_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(protocol), 4)
            results = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(results), 20)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "fault_p28_agentic_ablation/v1")
            self.assertEqual(manifest["track_id"], "fault")
            self.assertEqual(len(manifest["inputs"]), 7)
            self.assertEqual(len(manifest["outputs"]), 4)
            self.assertEqual(result["summary"]["retained_policy"], "A2D_deterministic_agent")
            self.assertEqual(result["summary"]["rejected_policy"], "A2L_llm_agent_execute")
            self.assertFalse(result["summary"]["frozen_test_accessed"])
            self.assertEqual(
                json.loads(summary_path.read_text(encoding="utf-8"))["selection_scenario_ids"],
                ["observed_blocked_current", "packet_hash_missing"],
            )
            self.assertIn("Retain:", evidence_path.read_text(encoding="utf-8"))
            for entry in manifest["inputs"] + manifest["outputs"]:
                resolved = PROJECT_ROOT / entry["path"]
                self.assertTrue(resolved.is_file(), resolved)
                self.assertEqual(entry["sha256"], sha256_file(resolved))
            self.assertEqual(git_head(), manifest["source_commit"])
        if output_root.exists():
            for child in output_root.iterdir():
                if child.is_file():
                    child.unlink()
            output_root.rmdir()

    def test_git_head_helper_returns_current_commit(self) -> None:
        self.assertRegex(git_head(), r"^[0-9a-f]{40}$|^unknown$")


if __name__ == "__main__":
    unittest.main()
