"""Public interface for the six-track research pipeline runtime."""

from .contracts import STAGES, TRACKS, TRACK_DIRS, PipelineAdapter, PipelineContractError, StageSpec
from .runner import build_plan, parse_parameters, preflight_project, verify_pipeline

__all__ = [
    "STAGES",
    "TRACKS",
    "TRACK_DIRS",
    "PipelineAdapter",
    "PipelineContractError",
    "StageSpec",
    "build_plan",
    "parse_parameters",
    "preflight_project",
    "verify_pipeline",
]
