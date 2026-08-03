"""Fail-closed planning and evidence verification for all six tracks."""

from __future__ import annotations

import hashlib
import json
import os
import string
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .contracts import (
    EXECUTION_MODES,
    SCHEMA_VERSION,
    STAGES,
    TRACKS,
    TRACK_DIRS,
    PipelineAdapter,
    PipelineContractError,
    StageSpec,
)
from .loader import default_project_root, load_adapter_dicts


PREFLIGHT_SCHEMA_VERSION = "six_track_pipeline_preflight/v1"
TRACE_SCHEMA_VERSION = "six_track_pipeline_trace/v1"
AGENT_FIELDS = ("role", "decision_owner", "candidate_source", "promotion_guard", "fallback")
STAGE_FIELDS = (
    "id",
    "needs",
    "execution",
    "entrypoint",
    "argv",
    "required_parameters",
    "required_inputs",
    "expected_outputs",
    "description",
)


def parse_parameters(items: Iterable[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    errors: list[str] = []
    for item in items:
        if "=" not in item:
            errors.append(f"parameter must use key=value syntax: {item!r}")
            continue
        key, value = item.split("=", 1)
        if not key or key.strip() != key:
            errors.append(f"invalid parameter name: {key!r}")
            continue
        if key in params:
            errors.append(f"duplicate parameter: {key}")
            continue
        params[key] = value
    if errors:
        raise PipelineContractError(errors)
    return params


def _as_string_list(value: Any, label: str, errors: list[str]) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        errors.append(f"{label} must be a list")
        return ()
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            errors.append(f"{label}[{index}] must be a non-empty string")
        else:
            result.append(item)
    return tuple(result)


def _repo_path(project_root: Path, value: str, label: str, errors: list[str]) -> Path | None:
    path = Path(value)
    if path.is_absolute():
        errors.append(f"{label} must be repository-relative: {value}")
        return None
    try:
        resolved = (project_root / path).resolve()
        resolved.relative_to(project_root)
    except ValueError:
        errors.append(f"{label} escapes the project root: {value}")
        return None
    return resolved


def _rendered_repo_path(project_root: Path, value: str, label: str, errors: list[str]) -> Path | None:
    """Resolve rendered inputs, accepting an absolute path only inside the project."""

    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError:
        errors.append(f"{label} escapes the project root: {value}")
        return None
    return resolved


def _template_fields(template: str) -> set[str]:
    fields: set[str] = set()
    try:
        parsed = string.Formatter().parse(template)
        for _, field, _, _ in parsed:
            if field:
                fields.add(field.split(".", 1)[0].split("[", 1)[0])
    except ValueError as exc:
        raise PipelineContractError([f"invalid template {template!r}: {exc}"]) from exc
    return fields


def _render(template: str, context: Mapping[str, str], label: str, errors: list[str]) -> str | None:
    try:
        return template.format_map(context)
    except (KeyError, ValueError) as exc:
        errors.append(f"{label} cannot be rendered: {exc}")
        return None


def _validate_stage(
    *,
    raw: Any,
    expected_id: str,
    previous_id: str | None,
    track: str,
    project_root: Path,
    errors: list[str],
) -> StageSpec | None:
    label = f"{track}:{expected_id}"
    if not isinstance(raw, dict):
        errors.append(f"{label} stage must be a dict")
        return None
    for field in STAGE_FIELDS:
        if field not in raw:
            errors.append(f"{label} missing stage field: {field}")

    stage_id = raw.get("id")
    if stage_id != expected_id:
        errors.append(f"{label} id must be {expected_id!r}, got {stage_id!r}")
    needs = _as_string_list(raw.get("needs"), f"{label}.needs", errors)
    expected_needs = () if previous_id is None else (previous_id,)
    if needs != expected_needs:
        errors.append(f"{label}.needs must be {list(expected_needs)!r}, got {list(needs)!r}")

    execution = raw.get("execution")
    if execution not in EXECUTION_MODES:
        errors.append(f"{label}.execution must be one of {EXECUTION_MODES}, got {execution!r}")

    entrypoint = raw.get("entrypoint")
    if entrypoint is not None and (not isinstance(entrypoint, str) or not entrypoint):
        errors.append(f"{label}.entrypoint must be a non-empty string or null")
        entrypoint = None
    if execution in ("command", "evidence") and not entrypoint:
        errors.append(f"{label} {execution} stage requires an entrypoint")
    if entrypoint:
        resolved = _repo_path(project_root, entrypoint, f"{label}.entrypoint", errors)
        if resolved is not None and (not resolved.is_file() or resolved.suffix != ".py"):
            errors.append(f"{label}.entrypoint is not an existing Python module: {entrypoint}")

    argv = _as_string_list(raw.get("argv"), f"{label}.argv", errors)
    required_parameters = _as_string_list(
        raw.get("required_parameters"), f"{label}.required_parameters", errors
    )
    required_inputs = _as_string_list(raw.get("required_inputs"), f"{label}.required_inputs", errors)
    expected_outputs = _as_string_list(raw.get("expected_outputs"), f"{label}.expected_outputs", errors)
    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append(f"{label}.description must be a non-empty string")
        description = ""

    required_set = set(required_parameters)
    template_fields = set().union(
        *(_template_fields(item) for item in (*argv, *required_inputs, *expected_outputs))
    )
    builtins = {"project_root", "task_dir", "track", "python"}
    undeclared = template_fields - required_set - builtins
    if undeclared:
        errors.append(f"{label} templates use undeclared parameters: {sorted(undeclared)}")
    if execution == "command" and entrypoint:
        expected_program = "{project_root}/" + entrypoint
        if len(argv) < 2 or argv[0] != "{python}" or argv[1] != expected_program:
            errors.append(
                f"{label}.argv must be complete tokens beginning "
                f"['{{python}}', {expected_program!r}]"
            )

    included_in = raw.get("included_in")
    if execution == "included" and (not isinstance(included_in, str) or not included_in):
        errors.append(f"{label} included stage requires included_in")
        included_in = None
    block_reason = raw.get("block_reason")
    if execution == "manual" and (not isinstance(block_reason, str) or not block_reason.strip()):
        errors.append(f"{label} manual stage requires block_reason")
        block_reason = None

    agent = raw.get("agent")
    if expected_id == "optimize":
        if not isinstance(agent, dict):
            errors.append(f"{label}.agent must be a dict")
            agent = None
        else:
            for field in AGENT_FIELDS:
                if field not in agent or agent[field] in (None, "", [], {}):
                    errors.append(f"{label}.agent missing non-empty field: {field}")

    return StageSpec(
        id=expected_id,
        needs=needs,
        execution=str(execution),
        entrypoint=entrypoint,
        argv=argv,
        required_parameters=required_parameters,
        required_inputs=required_inputs,
        expected_outputs=expected_outputs,
        description=description,
        included_in=included_in,
        block_reason=block_reason,
        agent=agent,
    )


def _validate_manifest_alignment(
    project_root: Path,
    adapter: PipelineAdapter,
    errors: list[str],
) -> None:
    label = adapter.track
    manifest_path = project_root / adapter.manifest
    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{label}.manifest cannot be parsed as YAML: {exc}")
        return
    if not isinstance(payload, dict):
        errors.append(f"{label}.manifest must contain a YAML mapping")
        return
    expected_pipeline = f"{label}_agentic_optimization"
    if payload.get("pipeline") != expected_pipeline:
        errors.append(f"{label}.manifest pipeline must be {expected_pipeline!r}")
    steps = payload.get("steps")
    if not isinstance(steps, list) or len(steps) != len(STAGES):
        errors.append(f"{label}.manifest must contain exactly seven steps")
        return
    expected_ref = f"code:{label}_pipeline_adapter"
    for index, stage_id in enumerate(STAGES):
        step = steps[index]
        step_label = f"{label}.manifest:{stage_id}"
        if not isinstance(step, dict):
            errors.append(f"{step_label} must be a mapping")
            continue
        if step.get("id") != stage_id:
            errors.append(f"{step_label}.id must be {stage_id!r}")
        needs = step.get("needs", [])
        expected_needs = [] if index == 0 else [STAGES[index - 1]]
        if needs != expected_needs or needs != list(adapter.stages[index].needs):
            errors.append(f"{step_label}.needs drifted from adapter: expected {expected_needs!r}")
        if step.get("ref") != expected_ref:
            errors.append(f"{step_label}.ref must be {expected_ref!r}")
        params = step.get("params")
        if not isinstance(params, dict):
            errors.append(f"{step_label}.params must be a mapping")
        else:
            if params.get("track") != label:
                errors.append(f"{step_label}.params.track must be {label!r}")
            if params.get("stage") != stage_id:
                errors.append(f"{step_label}.params.stage must be {stage_id!r}")


def _validate_adapter(
    project_root: Path, track: str, source_path: Path, raw: dict[str, Any], errors: list[str]
) -> PipelineAdapter | None:
    label = track
    if raw.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"{label}.schema_version must be {SCHEMA_VERSION!r}")
    if raw.get("track") != track:
        errors.append(f"{label}.track must match its canonical directory")

    task_dir = raw.get("task_dir")
    if not isinstance(task_dir, str) or not task_dir:
        errors.append(f"{label}.task_dir must be a repository-relative path")
        task_dir = ""
    else:
        task_path = _repo_path(project_root, task_dir, f"{label}.task_dir", errors)
        expected_task = (project_root / "_pipelines" / "02_task_datasets" / TRACK_DIRS[track]).resolve()
        if task_path != expected_task:
            errors.append(f"{label}.task_dir must be _pipelines/02_task_datasets/{TRACK_DIRS[track]}")
        elif not task_path.is_dir():
            errors.append(f"{label}.task_dir does not exist: {task_dir}")

    manifest = raw.get("manifest")
    expected_manifest = f"_pipelines/{track}_agentic_optimization.yml"
    if manifest != expected_manifest:
        errors.append(f"{label}.manifest must be {expected_manifest}")
        manifest = manifest if isinstance(manifest, str) else ""
    manifest_path = _repo_path(project_root, manifest, f"{label}.manifest", errors) if manifest else None
    if manifest_path is not None and not manifest_path.is_file():
        errors.append(f"{label}.manifest does not exist: {manifest}")

    stage_order = _as_string_list(raw.get("stage_order"), f"{label}.stage_order", errors)
    if stage_order != STAGES:
        errors.append(f"{label}.stage_order must be exactly {list(STAGES)!r}")
    raw_stages = raw.get("stages")
    if not isinstance(raw_stages, dict) or set(raw_stages) != set(STAGES):
        errors.append(f"{label}.stages must be a mapping with exactly {list(STAGES)!r}")
        raw_stages = raw_stages if isinstance(raw_stages, dict) else {}

    stages: list[StageSpec] = []
    for index, stage_id in enumerate(STAGES):
        raw_stage = raw_stages.get(stage_id)
        stage = _validate_stage(
            raw=raw_stage,
            expected_id=stage_id,
            previous_id=STAGES[index - 1] if index else None,
            track=track,
            project_root=project_root,
            errors=errors,
        )
        if stage is not None:
            stages.append(stage)

    by_id = {stage.id: stage for stage in stages}
    for stage in stages:
        if stage.execution == "included":
            if stage.included_in == stage.id or stage.included_in not in by_id:
                errors.append(f"{track}:{stage.id}.included_in must reference another canonical stage")

    prepare = by_id.get("prepare")
    if prepare is not None and not prepare.entrypoint:
        errors.append(f"{track}:prepare must expose a real module entrypoint")

    adapter = PipelineAdapter(
        schema_version=SCHEMA_VERSION,
        track=track,
        task_dir=task_dir,
        manifest=manifest,
        stage_order=stage_order,
        stages=tuple(stages),
        source_path=source_path,
    )
    _validate_manifest_alignment(project_root, adapter, errors)
    return adapter


def preflight_project(
    project_root: Path | None = None,
    *,
    intent: str = "verify",
    parameters: Mapping[str, str] | None = None,
    track: str | None = None,
    through_stage: str = "verify",
) -> tuple[dict[str, PipelineAdapter], dict[str, Any]]:
    """Validate all six adapters before verification or execution can start."""

    root = (project_root or default_project_root()).resolve()
    if intent not in ("verify", "execute"):
        raise PipelineContractError([f"unknown preflight intent: {intent}"])
    if track is not None and track not in (*TRACKS, "all"):
        raise PipelineContractError([f"unknown track: {track}"])
    if through_stage not in STAGES:
        raise PipelineContractError([f"unknown through stage: {through_stage}"])

    raw_adapters = load_adapter_dicts(root)
    errors: list[str] = []
    adapters: dict[str, PipelineAdapter] = {}
    for canonical_track in TRACKS:
        source_path, raw = raw_adapters[canonical_track]
        adapter = _validate_adapter(root, canonical_track, source_path, raw, errors)
        if adapter is not None:
            adapters[canonical_track] = adapter

    params = dict(parameters or {})
    if intent == "execute":
        selected_tracks = (track,) if track and track != "all" else TRACKS
        last_index = STAGES.index(through_stage)
        for selected_track in selected_tracks:
            adapter = adapters.get(selected_track)
            if adapter is None:
                continue
            produced_outputs: set[Path] = set()
            context = {
                "project_root": str(root),
                "task_dir": adapter.task_dir,
                "track": adapter.track,
                "python": sys.executable,
                **params,
            }
            for stage in adapter.stages[: last_index + 1]:
                label = f"{selected_track}:{stage.id}"
                if stage.execution == "manual":
                    errors.append(f"{label} is manual and cannot execute: {stage.block_reason}")
                    continue
                missing = sorted(set(stage.required_parameters) - params.keys())
                if stage.execution == "command" and missing:
                    errors.append(f"{label} missing required parameters: {missing}")
                if stage.execution != "command":
                    continue
                if missing:
                    continue
                for index, template in enumerate(stage.argv):
                    _render(template, context, f"{label}.argv[{index}]", errors)
                for index, template in enumerate(stage.required_inputs):
                    rendered = _render(template, context, f"{label}.required_inputs[{index}]", errors)
                    if rendered is None:
                        continue
                    input_path = _rendered_repo_path(
                        root, rendered, f"{label}.required_inputs[{index}]", errors
                    )
                    if (
                        input_path is not None
                        and not input_path.exists()
                        and input_path not in produced_outputs
                    ):
                        errors.append(f"{label} required input does not exist: {rendered}")
                for index, template in enumerate(stage.expected_outputs):
                    rendered = _render(template, context, f"{label}.expected_outputs[{index}]", errors)
                    if rendered is None:
                        continue
                    output_path = _rendered_repo_path(
                        root, rendered, f"{label}.expected_outputs[{index}]", errors
                    )
                    if output_path is not None:
                        produced_outputs.add(output_path)

    if errors:
        raise PipelineContractError(errors)
    report = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "PASS",
        "intent": intent,
        "adapter_count": len(adapters),
        "tracks": list(TRACKS),
        "selected_track": track or "all",
        "through_stage": through_stage,
    }
    return adapters, report


def build_plan(
    track: str,
    *,
    through_stage: str = "verify",
    project_root: Path | None = None,
    parameters: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    adapters, preflight = preflight_project(
        project_root, intent="verify", parameters=parameters, track=track, through_stage=through_stage
    )
    selected_tracks = TRACKS if track == "all" else (track,)

    def one(adapter: PipelineAdapter) -> dict[str, Any]:
        selected = adapter.stages[: STAGES.index(through_stage) + 1]
        return {
            "track": adapter.track,
            "stages": [
                {
                    "id": stage.id,
                    "needs": list(stage.needs),
                    "execution": stage.execution,
                    "entrypoint": stage.entrypoint,
                    "included_in": stage.included_in,
                    "description": stage.description,
                }
                for stage in selected
            ],
        }

    plans = [one(adapters[selected_track]) for selected_track in selected_tracks]
    result: dict[str, Any] = {
        "schema_version": "six_track_pipeline_plan/v1",
        "track": track,
        "through_stage": through_stage,
        "preflight": preflight,
    }
    if track == "all":
        result["pipelines"] = plans
    else:
        result["stages"] = plans[0]["stages"]
    return result


def _write_json(path: Path, project_root: Path, payload: dict[str, Any]) -> None:
    destination = path if path.is_absolute() else project_root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class VerificationFailed(RuntimeError):
    def __init__(self, trace: dict[str, Any]):
        self.trace = trace
        failed_track = trace.get("failed_track", trace.get("track", "unknown"))
        super().__init__(f"{failed_track} verification failed at {trace.get('failed_stage', 'preflight')}")


def _verify_one(
    root: Path,
    track: str,
    through_stage: str,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    lifecycle = root / "_pipelines" / "02_task_datasets" / "track_lifecycle.py"
    selected_stages = STAGES[: STAGES.index(through_stage) + 1]
    trace: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "status": "RUNNING",
        "track": track,
        "through_stage": through_stage,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "preflight": preflight,
        "stages": [],
    }
    for stage in selected_stages:
        actual_command = [
            sys.executable,
            os.fspath(lifecycle.relative_to(root)),
            "--track",
            track,
            "--stage",
            stage,
        ]
        portable_command = ["{python}", *actual_command[1:]]
        started = time.perf_counter()
        completed = subprocess.run(
            actual_command, cwd=root, capture_output=True, check=False, shell=False
        )
        duration = time.perf_counter() - started
        stdout = completed.stdout if isinstance(completed.stdout, bytes) else completed.stdout.encode("utf-8")
        trace["stages"].append(
            {
                "stage": stage,
                "command": portable_command,
                "exit_code": completed.returncode,
                "duration_seconds": round(duration, 6),
                "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            }
        )
        if completed.returncode != 0:
            trace["status"] = "FAIL"
            trace["failed_stage"] = stage
            trace["finished_at"] = datetime.now(timezone.utc).isoformat()
            raise VerificationFailed(trace)
    trace["status"] = "PASS"
    trace["finished_at"] = datetime.now(timezone.utc).isoformat()
    return trace


def verify_pipeline(
    track: str,
    *,
    through_stage: str = "verify",
    project_root: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    """Run lifecycle verification for every prefix stage, stopping on first failure."""

    root = (project_root or default_project_root()).resolve()
    _, preflight = preflight_project(root, intent="verify", track=track, through_stage=through_stage)
    lifecycle = root / "_pipelines" / "02_task_datasets" / "track_lifecycle.py"
    if not lifecycle.is_file():
        raise PipelineContractError(["missing shared lifecycle verifier: _pipelines/02_task_datasets/track_lifecycle.py"])

    if track != "all":
        try:
            trace = _verify_one(root, track, through_stage, preflight)
        except VerificationFailed as exc:
            if output is not None:
                _write_json(output, root, exc.trace)
            raise
        if output is not None:
            _write_json(output, root, trace)
        return trace

    batch: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "status": "RUNNING",
        "track": "all",
        "through_stage": through_stage,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "preflight": preflight,
        "pipelines": [{"track": item, "status": "NOT_RUN", "stages": []} for item in TRACKS],
    }
    for index, selected_track in enumerate(TRACKS):
        try:
            batch["pipelines"][index] = _verify_one(root, selected_track, through_stage, preflight)
        except VerificationFailed as exc:
            batch["pipelines"][index] = exc.trace
            batch["status"] = "FAIL"
            batch["failed_track"] = selected_track
            batch["failed_stage"] = exc.trace["failed_stage"]
            batch["finished_at"] = datetime.now(timezone.utc).isoformat()
            if output is not None:
                _write_json(output, root, batch)
            raise VerificationFailed(batch) from exc
    batch["status"] = "PASS"
    batch["finished_at"] = datetime.now(timezone.utc).isoformat()
    if output is not None:
        _write_json(output, root, batch)
    return batch
