from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from _code.six_track_pipeline.contracts import STAGES, TRACKS, TRACK_DIRS, PipelineContractError
from _code.six_track_pipeline.runner import (
    VerificationFailed,
    build_plan,
    parse_parameters,
    preflight_project,
    verify_pipeline,
)


def _stage(stage_id: str, index: int, *, execution: str = "evidence") -> dict:
    result = {
        "id": stage_id,
        "needs": [] if index == 0 else [STAGES[index - 1]],
        "execution": execution,
        "entrypoint": "shared/entry.py",
        "argv": [],
        "required_parameters": [],
        "required_inputs": ["shared/input.json"],
        "expected_outputs": [f"outputs/{stage_id}.json"],
        "description": f"{stage_id} stage",
    }
    if stage_id == "optimize":
        result["agent"] = {
            "role": "candidate selector",
            "decision_owner": "development evidence",
            "candidate_source": "fixed action bank",
            "promotion_guard": "incumbent non-degradation",
            "fallback": "retain incumbent",
        }
    return result


class ProjectFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "shared").mkdir(parents=True)
        (self.root / "shared" / "entry.py").write_text("pass\n", encoding="utf-8")
        (self.root / "shared" / "input.json").write_text("{}\n", encoding="utf-8")
        lifecycle = self.root / "_pipelines" / "02_task_datasets" / "track_lifecycle.py"
        lifecycle.parent.mkdir(parents=True)
        lifecycle.write_text("pass\n", encoding="utf-8")
        for track in TRACKS:
            task_dir = self.root / "_pipelines" / "02_task_datasets" / TRACK_DIRS[track]
            task_dir.mkdir(parents=True)
            manifest = self.root / "_pipelines" / f"{track}_agentic_optimization.yml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "pipeline": f"{track}_agentic_optimization",
                        "steps": [
                            {
                                "id": stage,
                                "ref": f"code:{track}_pipeline_adapter",
                                **({} if index == 0 else {"needs": [STAGES[index - 1]]}),
                                "params": {"track": track, "stage": stage},
                            }
                            for index, stage in enumerate(STAGES)
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            adapter = {
                "schema_version": "six_track_adapter/v1",
                "track": track,
                "task_dir": f"_pipelines/02_task_datasets/{TRACK_DIRS[track]}",
                "manifest": f"_pipelines/{track}_agentic_optimization.yml",
                "stage_order": list(STAGES),
                "stages": {stage: _stage(stage, index) for index, stage in enumerate(STAGES)},
            }
            self.write_adapter(track, adapter)

    def write_adapter(self, track: str, adapter: dict) -> None:
        path = self.root / "_pipelines" / "02_task_datasets" / TRACK_DIRS[track] / "pipeline_adapter.py"
        path.write_text("ADAPTER = " + repr(adapter) + "\n", encoding="utf-8")

    def read_adapter(self, track: str) -> dict:
        namespace: dict = {}
        path = self.root / "_pipelines" / "02_task_datasets" / TRACK_DIRS[track] / "pipeline_adapter.py"
        exec(path.read_text(encoding="utf-8"), namespace)
        return namespace["ADAPTER"]

    def close(self) -> None:
        self.temporary.cleanup()


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project = ProjectFixture()

    def tearDown(self) -> None:
        self.project.close()

    def test_preflight_loads_exactly_six_adapters(self) -> None:
        adapters, report = preflight_project(self.project.root)
        self.assertEqual(tuple(adapters), TRACKS)
        self.assertEqual(report["adapter_count"], 6)

    def test_extra_adapter_is_rejected(self) -> None:
        extra = self.project.root / "_pipelines" / "02_task_datasets" / "seventh"
        extra.mkdir()
        (extra / "pipeline_adapter.py").write_text("ADAPTER = {}\n", encoding="utf-8")
        with self.assertRaisesRegex(PipelineContractError, "unexpected adapter"):
            preflight_project(self.project.root)

    def test_plan_is_dependency_closed_prefix(self) -> None:
        plan = build_plan("facies", through_stage="promote", project_root=self.project.root)
        self.assertEqual([item["id"] for item in plan["stages"]], list(STAGES[:5]))
        self.assertEqual(plan["stages"][-1]["needs"], ["optimize"])

    def test_all_plan_has_six_fixed_order_pipelines(self) -> None:
        plan = build_plan("all", through_stage="prepare", project_root=self.project.root)
        self.assertEqual([item["track"] for item in plan["pipelines"]], list(TRACKS))
        self.assertTrue(all(len(item["stages"]) == 2 for item in plan["pipelines"]))

    def test_prepare_requires_a_real_module_entrypoint(self) -> None:
        adapter = self.project.read_adapter("fault")
        adapter["stages"]["prepare"]["entrypoint"] = "missing.py"
        self.project.write_adapter("fault", adapter)
        with self.assertRaisesRegex(PipelineContractError, "prepare.entrypoint"):
            preflight_project(self.project.root)

    def test_optimize_requires_complete_agent_metadata(self) -> None:
        adapter = self.project.read_adapter("fault")
        del adapter["stages"]["optimize"]["agent"]["fallback"]
        self.project.write_adapter("fault", adapter)
        with self.assertRaisesRegex(PipelineContractError, "agent missing non-empty field: fallback"):
            preflight_project(self.project.root)

    def test_execute_preflight_fails_before_action_for_missing_parameter(self) -> None:
        adapter = self.project.read_adapter("fault")
        prepare = adapter["stages"]["prepare"]
        prepare["execution"] = "command"
        prepare["required_parameters"] = ["dataset"]
        prepare["argv"] = ["{python}", "{project_root}/shared/entry.py", "--dataset", "{dataset}"]
        prepare["required_inputs"] = ["{dataset}"]
        self.project.write_adapter("fault", adapter)
        with patch("_code.six_track_pipeline.runner.subprocess.run") as run:
            with self.assertRaisesRegex(PipelineContractError, "missing required parameters"):
                preflight_project(
                    self.project.root,
                    intent="execute",
                    track="fault",
                    through_stage="prepare",
                )
            run.assert_not_called()

    def test_execute_preflight_rejects_manual_stage(self) -> None:
        adapter = self.project.read_adapter("fault")
        adapter["stages"]["baseline"]["execution"] = "manual"
        adapter["stages"]["baseline"]["entrypoint"] = None
        adapter["stages"]["baseline"]["block_reason"] = "human approval required"
        self.project.write_adapter("fault", adapter)
        with self.assertRaisesRegex(PipelineContractError, "is manual and cannot execute"):
            preflight_project(
                self.project.root,
                intent="execute",
                track="fault",
                through_stage="baseline",
            )

    def test_manifest_ref_drift_fails_loud(self) -> None:
        path = self.project.root / "_pipelines" / "fault_agentic_optimization.yml"
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        payload["steps"][2]["ref"] = "code:track_lifecycle_verify"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with self.assertRaisesRegex(PipelineContractError, "manifest:baseline.ref"):
            preflight_project(self.project.root)

    def test_manifest_track_drift_fails_loud(self) -> None:
        path = self.project.root / "_pipelines" / "fault_agentic_optimization.yml"
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        payload["steps"][3]["params"]["track"] = "facies"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with self.assertRaisesRegex(PipelineContractError, "params.track"):
            preflight_project(self.project.root)

    def test_verify_calls_every_stage_and_writes_requested_trace(self) -> None:
        outputs = [json.dumps({"status": "PASS", "stage": stage}).encode() for stage in STAGES[:4]]
        completed = [subprocess.CompletedProcess([], 0, stdout=value, stderr=b"") for value in outputs]
        trace_path = Path("traces/fault.json")
        with patch("_code.six_track_pipeline.runner.subprocess.run", side_effect=completed) as run:
            trace = verify_pipeline(
                "fault",
                through_stage="optimize",
                project_root=self.project.root,
                output=trace_path,
            )
        self.assertEqual(run.call_count, 4)
        self.assertEqual([item["stage"] for item in trace["stages"]], list(STAGES[:4]))
        self.assertEqual(trace["stages"][0]["stdout_sha256"], hashlib.sha256(outputs[0]).hexdigest())
        persisted = json.loads((self.project.root / trace_path).read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], "PASS")

    def test_verify_stops_at_first_failure(self) -> None:
        completed = [
            subprocess.CompletedProcess([], 0, stdout=b"ok", stderr=b""),
            subprocess.CompletedProcess([], 7, stdout=b"bad", stderr=b"failure"),
        ]
        with patch("_code.six_track_pipeline.runner.subprocess.run", side_effect=completed) as run:
            with self.assertRaises(VerificationFailed) as caught:
                verify_pipeline("fault", project_root=self.project.root)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(caught.exception.trace["failed_stage"], "prepare")

    def test_verify_all_has_six_records(self) -> None:
        completed = [subprocess.CompletedProcess([], 0, stdout=b"ok", stderr=b"")] * (
            len(TRACKS) * 2
        )
        with patch("_code.six_track_pipeline.runner.subprocess.run", side_effect=completed) as run:
            trace = verify_pipeline(
                "all", through_stage="prepare", project_root=self.project.root
            )
        self.assertEqual(run.call_count, 12)
        self.assertEqual([item["track"] for item in trace["pipelines"]], list(TRACKS))

    def test_parameter_parser_is_strict(self) -> None:
        self.assertEqual(parse_parameters(["fold=2", "seed=7"]), {"fold": "2", "seed": "7"})
        with self.assertRaises(PipelineContractError):
            parse_parameters(["fold=2", "fold=3"])


if __name__ == "__main__":
    unittest.main()
