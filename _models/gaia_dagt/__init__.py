"""Portable offline Gaia/DAGT public layer."""

from .agents import (
    AgentParseError,
    AgentUnavailableError,
    CacheCorruptionError,
    predictive_text_agent,
    supervisory_qc_agent,
    agent_unavailable,
    load_cached_agent_evidence,
    parse_api_payload,
    require_api_key,
)
from .adapter import GaiaDAGTAdapter, DryRunResult, render_sci_svg
from .contracts import (
    DEFAULT_PROMPT_VERSION,
    AgentEvidence,
    ModelBatch,
    ModelOutput,
    TrackSpec,
)
from .controls import apply_control, counterfactual_control, random_control, real_control, shuffle_control
from .source_lock import (
    DEFAULT_SOURCE_MANIFEST,
    SourceFileRecord,
    SourceLockError,
    SourceManifest,
    SourceManifestStatus,
    verify_default_source_manifest,
)

__all__ = [
    "AgentEvidence",
    "AgentParseError",
    "AgentUnavailableError",
    "CacheCorruptionError",
    "DEFAULT_PROMPT_VERSION",
    "DEFAULT_SOURCE_MANIFEST",
    "DryRunResult",
    "GaiaDAGTAdapter",
    "ModelBatch",
    "ModelOutput",
    "SourceFileRecord",
    "SourceLockError",
    "SourceManifest",
    "SourceManifestStatus",
    "TrackSpec",
    "agent_unavailable",
    "apply_control",
    "counterfactual_control",
    "load_cached_agent_evidence",
    "parse_api_payload",
    "predictive_text_agent",
    "random_control",
    "real_control",
    "render_sci_svg",
    "require_api_key",
    "shuffle_control",
    "supervisory_qc_agent",
    "verify_default_source_manifest",
]
