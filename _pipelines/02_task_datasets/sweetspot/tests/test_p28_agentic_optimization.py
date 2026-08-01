from __future__ import annotations

import ast
import importlib
import json
import tempfile
import unittest
from pathlib import Path


p28 = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p28_agentic_optimization"
)


class SweetspotP28AgenticOptimizationTests(unittest.TestCase):
    def test_source_rejects_sibling_worktree_python_imports(self) -> None:
        source = Path(p28.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        hits: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if ".claude.worktrees" in node.module or "p10-results-sweetspot" in node.module:
                    hits.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if ".claude.worktrees" in alias.name or "p10-results-sweetspot" in alias.name:
                        hits.append(alias.name)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        value = node.args[0].value
                        if ".claude.worktrees" in value or "p10-results-sweetspot" in value:
                            hits.append(value)

        self.assertEqual(hits, [], f"sibling-worktree runtime imports found: {hits}")

    def test_prompt_excludes_promotion_feedback(self) -> None:
        prompt = p28._build_prompt(
            observation=p28._observation(),
            candidate_feedback=[
                {
                    "action_id": p28.XGB_ACTION_IDS[0],
                    "selection_feedback": "flat",
                    "budget": 64,
                }
            ],
        )
        prompt_text = json.dumps(prompt, ensure_ascii=False)
        self.assertIn("selection_feedback", prompt_text)
        self.assertNotIn("promotion_feedback", prompt_text)
        self.assertNotIn("promotion_mae", prompt_text)

    def test_generate_report_writes_local_pilot_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = p28.generate_report(Path(tmp), call_deepseek=False)

            protocol = Path(tmp) / "protocol.json"
            protocol_jsonl = Path(tmp) / "protocol.jsonl"
            summary_path = Path(tmp) / "summary.json"
            evidence = Path(tmp) / "evidence.md"
            manifest = Path(tmp) / "manifest.json"

            for path in (protocol, protocol_jsonl, summary_path, evidence, manifest):
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 0)

            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest_data["track_id"], "sweetspot")
            self.assertEqual(manifest_data["schema_version"], p28.SCHEMA_VERSION)
            self.assertEqual(manifest_data["renderer"]["path"], "_pipelines/02_task_datasets/sweetspot/p28_agentic_optimization.py")
            self.assertFalse(manifest_data["manual_review"]["reviewed"])
            self.assertEqual(manifest_data["artifact_count"], 4)
            self.assertGreaterEqual(len(manifest_data["inputs"]), 8)
            self.assertEqual(summary["executor_available"], True)
            self.assertEqual(summary["verdict"], "blocked")
            self.assertFalse(summary["retain_llm"])
            self.assertEqual(summary["gates"], {"T5": "not_feasible", "T6": "blocked", "T7": "blocked"})
            self.assertEqual(summary["selection_fold_ids"], [0, 1, 2])
            self.assertEqual(summary["promotion_fold_ids"], [3])
            self.assertEqual(summary["a2d"]["status"], "EXECUTED")
            self.assertEqual(summary["a3"]["status"], "EXECUTED")
            self.assertEqual(summary["a2l"]["status"], "BLOCKED")
            self.assertTrue(summary["a1_same_hash"] is True)
            self.assertEqual(summary["a0_prediction_hash"], summary["a1_prediction_hash"])
            protocol_data = json.loads(protocol.read_text(encoding="utf-8"))
            protocol_rows = protocol_jsonl.read_text(encoding="utf-8").splitlines()
            self.assertEqual(protocol_data["action_allowlist"]["xgboost"], list(p28.XGB_ACTION_IDS))
            self.assertTrue(protocol_data["leakage_firewall"]["sibling_worktree_python_imports"] is False)
            self.assertTrue(protocol_data["leakage_firewall"]["foundation_route_fixed_or_factorized"])
            self.assertEqual(protocol_data["promotion_gate"]["equal_trial_budget"], 4)
            self.assertNotIn("promotion_dev_feedback_improved_flat_worse", protocol_data["observation_schema"]["visible_to_llm"])
            self.assertEqual(protocol_data["task_spec"]["route_family"], "xgboost")
            self.assertIn('"status": "EXECUTED_IDENTITY_CHECK"', protocol_rows[1])

    def test_summary_jsonl_aligns_with_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            summary = p28.generate_report(output_dir, call_deepseek=False)
            manifest_data = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            protocol_row = next(row for row in manifest_data["outputs"] if row["role"] == "protocol")
            summary_row = next(row for row in manifest_data["outputs"] if row["role"] == "summary")

            self.assertEqual(summary["protocol_sha256"], protocol_row["sha256"])
            self.assertEqual(summary["executor_available"], True)
            self.assertEqual(summary["a4"]["status"], "EXECUTED")
            self.assertEqual(summary["a4"]["selected_action_id"], "T3_XGB_D4_ETA_0_05_ROUNDS_96")
            self.assertEqual(summary_row["sha256"], p28._sha256_file(output_dir / "summary.json"))


if __name__ == "__main__":
    unittest.main()
