from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_PROMPT_VERSION = "p6.gaia.dagt.v1"
DEFAULT_DENY_LIST = (
    "label",
    "labels",
    "target",
    "targets",
    "y",
    "truth",
    "ground_truth",
    "groundtruth",
    "test",
    "test_label",
    "test_labels",
    "test_metric",
    "test_metrics",
    "metric",
    "metrics",
    "stat",
    "stats",
    "statistics",
    "test_stat",
    "test_statistics",
    "score",
    "path",
    "paths",
    "file",
    "file_path",
    "filepath",
    "uri",
    "url",
    "checkpoint",
    "holdout",
    "frozen",
    "split",
    "prediction_error",
    "error",
    "residual",
    "distance",
    "offset",
    "inferred_ac_offset",
)
ALLOWED_AGENT_MODES = (
    "predictive_text_agent",
    "supervisory_qc_agent",
    "agent_unavailable",
)
ALLOWED_TASK_TYPES = (
    "classification",
    "regression",
    "segmentation_2d",
    "multitask",
    "volume_3d",
)


def _normalize_value(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_value(item) for item in value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_normalize_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def infer_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or value is None:
        return ()
    if isinstance(value, Mapping):
        return (len(value),)
    if isinstance(value, Sequence):
        if not value:
            return (0,)
        return (len(value),) + infer_shape(value[0])
    return ()


def contains_nan(value: Any) -> bool:
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, Mapping):
        return any(contains_nan(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(contains_nan(item) for item in value)
    return False


def _normalize_key(key: Any) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _is_prohibited_key(key: Any, deny_list: set[str]) -> bool:
    normalized = _normalize_key(key)
    return any(token and (token in normalized or normalized in token) for token in deny_list)


def _contains_prohibited_key(value: Any, deny_list: set[str]) -> tuple[bool, str | None]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _is_prohibited_key(key, deny_list):
                return True, str(key)
            hit, field = _contains_prohibited_key(nested, deny_list)
            if hit:
                return True, field
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            hit, field = _contains_prohibited_key(nested, deny_list)
            if hit:
                return True, field
    return False, None


_WINDOWS_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


def _contains_prohibited_string(value: Any) -> tuple[bool, str | None]:
    if isinstance(value, str):
        if value.startswith("file://") or value.startswith("/") or _WINDOWS_PATH_RE.match(value):
            return True, value
        return False, None
    if isinstance(value, Mapping):
        for nested in value.values():
            hit, field = _contains_prohibited_string(nested)
            if hit:
                return True, field
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            hit, field = _contains_prohibited_string(nested)
            if hit:
                return True, field
    return False, None


def _freeze_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _normalize_value(value)
        for key, value in sorted(mapping.items(), key=lambda item: str(item[0]))
    }


@dataclass(frozen=True, slots=True)
class TrackSpec:
    track_id: str
    task_type: str
    modality: str
    input_fields: tuple[str, ...]
    target_fields: tuple[str, ...]
    allowed_paths: tuple[str, ...] = ()
    forbidden_paths: tuple[str, ...] = ()
    metric_names: tuple[str, ...] = ()
    base_seed: int = 2693
    prompt_version: str = DEFAULT_PROMPT_VERSION
    source_manifest_digest: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task_type not in ALLOWED_TASK_TYPES:
            raise ValueError(f"Unsupported task_type: {self.task_type}")
        if not self.source_manifest_digest:
            raise ValueError("TrackSpec.source_manifest_digest must be set")
        if self.prompt_version != DEFAULT_PROMPT_VERSION:
            raise ValueError("TrackSpec.prompt_version must be pinned to the default prompt version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "task_type": self.task_type,
            "modality": self.modality,
            "input_fields": list(self.input_fields),
            "target_fields": list(self.target_fields),
            "allowed_paths": list(self.allowed_paths),
            "forbidden_paths": list(self.forbidden_paths),
            "metric_names": list(self.metric_names),
            "base_seed": self.base_seed,
            "prompt_version": self.prompt_version,
            "source_manifest_digest": self.source_manifest_digest,
            "provenance": _freeze_mapping(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrackSpec":
        return cls(
            track_id=str(payload["track_id"]),
            task_type=str(payload["task_type"]),
            modality=str(payload["modality"]),
            input_fields=tuple(payload.get("input_fields", ())),
            target_fields=tuple(payload.get("target_fields", ())),
            allowed_paths=tuple(payload.get("allowed_paths", ())),
            forbidden_paths=tuple(payload.get("forbidden_paths", ())),
            metric_names=tuple(payload.get("metric_names", ())),
            base_seed=int(payload.get("base_seed", 2693)),
            prompt_version=str(payload.get("prompt_version", DEFAULT_PROMPT_VERSION)),
            source_manifest_digest=str(payload.get("source_manifest_digest", "")),
            provenance=dict(payload.get("provenance", {})),
        )

    def cache_key(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class AgentEvidence:
    prompt_version: str
    agent_mode: str
    source_text_hash: str
    structured_priors: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    source_manifest_digest: str = ""
    control_mode: str = "real"
    seed: int = 0
    cache_key: str = ""
    deny_list: tuple[str, ...] = DEFAULT_DENY_LIST

    def __post_init__(self) -> None:
        if self.prompt_version != DEFAULT_PROMPT_VERSION:
            raise ValueError("AgentEvidence.prompt_version must be pinned to the default prompt version")
        if self.agent_mode not in ALLOWED_AGENT_MODES:
            raise ValueError(f"Unsupported agent_mode: {self.agent_mode}")
        normalized_deny = {_normalize_key(item) for item in self.deny_list}
        hit, field = _contains_prohibited_key(self.structured_priors, normalized_deny)
        if hit:
            raise ValueError(f"structured_priors contains prohibited field: {field}")
        hit, field = _contains_prohibited_string(self.structured_priors)
        if hit:
            raise ValueError(f"structured_priors contains prohibited path-like value: {field}")
        hit, field = _contains_prohibited_key(self.provenance, normalized_deny)
        if hit:
            raise ValueError(f"provenance contains prohibited field: {field}")
        hit, field = _contains_prohibited_string(self.provenance)
        if hit:
            raise ValueError(f"provenance contains prohibited path-like value: {field}")
        hit, field = _contains_prohibited_key({"evidence": self.evidence, "warnings": self.warnings}, normalized_deny)
        if hit:
            raise ValueError(f"evidence contains prohibited field: {field}")
        hit, field = _contains_prohibited_string({"evidence": self.evidence, "warnings": self.warnings})
        if hit:
            raise ValueError(f"evidence contains prohibited path-like value: {field}")
        expected = self._expected_cache_key()
        if self.cache_key and self.cache_key != expected:
            raise ValueError("AgentEvidence.cache_key is not deterministic")
        object.__setattr__(self, "cache_key", expected)

    def _expected_cache_key(self) -> str:
        payload = {
            "agent_mode": self.agent_mode,
            "prompt_version": self.prompt_version,
            "source_text_hash": self.source_text_hash,
            "source_manifest_digest": self.source_manifest_digest,
            "control_mode": self.control_mode,
            "seed": self.seed,
            "structured_priors": _freeze_mapping(self.structured_priors),
            "provenance": _freeze_mapping(self.provenance),
        }
        return sha256_json(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_version": self.prompt_version,
            "agent_mode": self.agent_mode,
            "source_text_hash": self.source_text_hash,
            "structured_priors": _freeze_mapping(self.structured_priors),
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
            "provenance": _freeze_mapping(self.provenance),
            "source_manifest_digest": self.source_manifest_digest,
            "control_mode": self.control_mode,
            "seed": self.seed,
            "cache_key": self.cache_key,
            "deny_list": list(self.deny_list),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AgentEvidence":
        return cls(
            prompt_version=str(payload["prompt_version"]),
            agent_mode=str(payload["agent_mode"]),
            source_text_hash=str(payload["source_text_hash"]),
            structured_priors=dict(payload.get("structured_priors", {})),
            confidence=float(payload.get("confidence", 0.0)),
            evidence=tuple(payload.get("evidence", ())),
            warnings=tuple(payload.get("warnings", ())),
            provenance=dict(payload.get("provenance", {})),
            source_manifest_digest=str(payload.get("source_manifest_digest", "")),
            control_mode=str(payload.get("control_mode", "real")),
            seed=int(payload.get("seed", 0)),
            cache_key=str(payload.get("cache_key", "")),
            deny_list=tuple(payload.get("deny_list", DEFAULT_DENY_LIST)),
        )


@dataclass(frozen=True, slots=True)
class ModelBatch:
    track_spec: TrackSpec
    features: Any
    target: Any = None
    mask: Any = None
    task_targets: Mapping[str, Any] = field(default_factory=dict)
    task_metrics: Mapping[str, str] = field(default_factory=dict)
    task_masks: Mapping[str, Any] = field(default_factory=dict)
    feasibility: Mapping[str, bool] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    agent_evidence: AgentEvidence | None = None
    voxel_index_map: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_spec": self.track_spec.to_dict(),
            "features": _normalize_value(self.features),
            "target": _normalize_value(self.target),
            "mask": _normalize_value(self.mask),
            "task_targets": _normalize_value(self.task_targets),
            "task_metrics": _freeze_mapping(self.task_metrics),
            "task_masks": _normalize_value(self.task_masks),
            "feasibility": _freeze_mapping(self.feasibility),
            "metadata": _freeze_mapping(self.metadata),
            "agent_evidence": None if self.agent_evidence is None else self.agent_evidence.to_dict(),
            "voxel_index_map": _normalize_value(self.voxel_index_map),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelBatch":
        agent_payload = payload.get("agent_evidence")
        return cls(
            track_spec=TrackSpec.from_dict(payload["track_spec"]),
            features=payload.get("features"),
            target=payload.get("target"),
            mask=payload.get("mask"),
            task_targets=dict(payload.get("task_targets", {})),
            task_metrics=dict(payload.get("task_metrics", {})),
            task_masks=dict(payload.get("task_masks", {})),
            feasibility=dict(payload.get("feasibility", {})),
            metadata=dict(payload.get("metadata", {})),
            agent_evidence=None if agent_payload is None else AgentEvidence.from_dict(agent_payload),
            voxel_index_map=payload.get("voxel_index_map"),
        )

    def kind(self) -> str:
        return self.track_spec.task_type

    def shape(self) -> tuple[int, ...]:
        return infer_shape(self.features)


@dataclass(frozen=True, slots=True)
class ModelOutput:
    track_id: str
    prediction: Any
    logits: Any = None
    metric: Mapping[str, float] = field(default_factory=dict)
    uncertainty: Any = None
    agent_mode: str = "agent_unavailable"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "prediction": _normalize_value(self.prediction),
            "logits": _normalize_value(self.logits),
            "metric": _freeze_mapping(self.metric),
            "uncertainty": _normalize_value(self.uncertainty),
            "agent_mode": self.agent_mode,
            "provenance": _freeze_mapping(self.provenance),
            "diagnostics": _freeze_mapping(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelOutput":
        return cls(
            track_id=str(payload["track_id"]),
            prediction=payload.get("prediction"),
            logits=payload.get("logits"),
            metric=dict(payload.get("metric", {})),
            uncertainty=payload.get("uncertainty"),
            agent_mode=str(payload.get("agent_mode", "agent_unavailable")),
            provenance=dict(payload.get("provenance", {})),
            diagnostics=dict(payload.get("diagnostics", {})),
        )
