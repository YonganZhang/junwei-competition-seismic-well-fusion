"""Stable contracts shared by the six archived research pipelines.

The adapters deliberately expose plain dictionaries.  This module turns those
dictionaries into immutable, typed values only after the complete project has
passed a fail-closed preflight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "six_track_adapter/v1"
TRACKS = (
    "fault",
    "facies",
    "property",
    "lithofacies",
    "sweetspot",
    "reconstruction",
)
TRACK_DIRS = {track: ("reservoir" if track == "property" else track) for track in TRACKS}
STAGES = ("validate", "prepare", "baseline", "optimize", "promote", "refit", "verify")
EXECUTION_MODES = ("command", "evidence", "included", "manual")


class PipelineContractError(RuntimeError):
    """Raised before execution when one or more adapter contracts are unsafe."""

    def __init__(self, errors: list[str] | tuple[str, ...]):
        self.errors = tuple(errors)
        super().__init__("pipeline preflight failed:\n- " + "\n- ".join(self.errors))


@dataclass(frozen=True)
class StageSpec:
    id: str
    needs: tuple[str, ...]
    execution: str
    entrypoint: str | None
    argv: tuple[str, ...]
    required_parameters: tuple[str, ...]
    required_inputs: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    description: str
    included_in: str | None = None
    block_reason: str | None = None
    agent: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PipelineAdapter:
    schema_version: str
    track: str
    task_dir: str
    manifest: str
    stage_order: tuple[str, ...]
    stages: tuple[StageSpec, ...]
    source_path: Path

    def stage(self, stage_id: str) -> StageSpec:
        for stage in self.stages:
            if stage.id == stage_id:
                return stage
        raise KeyError(stage_id)
