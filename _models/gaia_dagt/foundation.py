"""Typed, fail-closed contract for modality-specific foundation models.

Natural-language prompts are only one conditioning type.  Time windows,
tabular support sets, measured-depth windows, spatial prompts and masked
volumes are first-class contracts with their own leakage rules.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import canonical_json


FOUNDATION_SCHEMA_VERSION = "gaia-foundation-task/v1"
FOUNDATION_PROMPT_VERSION = "p8.foundation.conditioning.v1"
FOUNDATION_ROUTE_SCHEMA_VERSION = "gaia-foundation-routes/v1"

FOUNDATION_STATES = (
    "CONNECTED_UNVERIFIED",
    "VERIFIED_NO_GAIN",
    "PROMOTED_DEV",
    "CONFIRMED_HOLDOUT",
)
_STATE_TRANSITIONS = {
    "CONNECTED_UNVERIFIED": {"VERIFIED_NO_GAIN", "PROMOTED_DEV"},
    "VERIFIED_NO_GAIN": {"PROMOTED_DEV"},
    "PROMOTED_DEV": {"VERIFIED_NO_GAIN", "CONFIRMED_HOLDOUT"},
    "CONFIRMED_HOLDOUT": set(),
}
AXIS_KINDS = ("none", "time", "measured_depth", "image_xy", "volume_kji")
CONDITIONING_KINDS = (
    "language_prompt",
    "time_window",
    "support_set",
    "depth_window",
    "spatial_prompt",
    "masked_volume",
)
TASK_TYPES = (
    "time_forecasting",
    "tabular_regression",
    "depth_classification",
    "segmentation_2d",
    "segmentation_3d",
    "volume_regression_3d",
)
ARTIFACT_STATES = ("not_cached", "cached_unverified", "cached_verified")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hex(value: str, length: int, name: str) -> None:
    if len(value) != length or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"{name} must be a {length}-character hexadecimal digest")


def _require_keys(payload: Mapping[str, Any], required: set[str], kind: str) -> None:
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{kind} conditioning is missing: {missing}")


def _is_monotonic(values: list[Any]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


@dataclass(frozen=True, slots=True)
class FoundationModelRef:
    model_id: str
    family: str
    source_url: str
    source_revision: str
    code_license: str
    weights_uri: str
    weights_revision: str
    weight_license: str
    artifact_state: str = "not_cached"
    weights_sha256: str = ""
    weights_size_bytes: int = 0
    parameter_count: int = 0
    license_status: str = "approved"

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "family",
            "source_url",
            "source_revision",
            "code_license",
            "weights_uri",
            "weights_revision",
            "weight_license",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"FoundationModelRef.{name} must be set")
        _require_hex(self.source_revision, 40, "source_revision")
        _require_hex(self.weights_revision, 40, "weights_revision")
        if self.artifact_state not in ARTIFACT_STATES:
            raise ValueError(f"unsupported artifact_state: {self.artifact_state}")
        if self.artifact_state == "cached_verified":
            _require_hex(self.weights_sha256, 64, "weights_sha256")
            if self.weights_size_bytes <= 0:
                raise ValueError("cached_verified weights require weights_size_bytes")
        elif self.weights_sha256:
            _require_hex(self.weights_sha256, 64, "weights_sha256")
        if self.license_status not in {"approved", "review_required", "rejected"}:
            raise ValueError(f"unsupported license_status: {self.license_status}")
        if self.license_status == "rejected":
            raise ValueError("rejected foundation artifacts cannot be routed")

    def verify_local_artifact(self, path: Path) -> None:
        if not self.weights_sha256 or self.weights_size_bytes <= 0:
            raise RuntimeError(f"{self.model_id} has no auditable checkpoint digest")
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"missing local foundation artifact: {resolved}")
        if resolved.stat().st_size != self.weights_size_bytes:
            raise ValueError(f"{self.model_id} checkpoint size mismatch")
        if _sha256_file(resolved) != self.weights_sha256:
            raise ValueError(f"{self.model_id} checkpoint SHA256 mismatch")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FoundationModelRef":
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class TensorFieldSpec:
    name: str
    unit: str
    dtype: str
    shape: tuple[int | str, ...]
    mask_semantics: str
    coordinate_frame: str

    def __post_init__(self) -> None:
        for name in ("name", "unit", "dtype", "mask_semantics", "coordinate_frame"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"TensorFieldSpec.{name} must be set")
        if not self.shape:
            raise ValueError("TensorFieldSpec.shape must be set")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TensorFieldSpec":
        values = dict(payload)
        values["shape"] = tuple(values["shape"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class VisibilityPolicy:
    allowed_fields: tuple[str, ...]
    forbidden_fields: tuple[str, ...]
    cutoff: str = ""
    query_target_visible: bool = False
    frozen_test_accessed: bool = False

    def __post_init__(self) -> None:
        overlap = sorted(set(self.allowed_fields) & set(self.forbidden_fields))
        if overlap:
            raise ValueError(f"visibility whitelist contains forbidden fields: {overlap}")
        if self.query_target_visible:
            raise ValueError("query target must never be visible to a foundation request")
        if self.frozen_test_accessed:
            raise ValueError("foundation request cannot be built after frozen-test access")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_fields": list(self.allowed_fields),
            "forbidden_fields": list(self.forbidden_fields),
            "cutoff": self.cutoff,
            "query_target_visible": self.query_target_visible,
            "frozen_test_accessed": self.frozen_test_accessed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VisibilityPolicy":
        return cls(
            allowed_fields=tuple(payload.get("allowed_fields", ())),
            forbidden_fields=tuple(payload.get("forbidden_fields", ())),
            cutoff=str(payload.get("cutoff", "")),
            query_target_visible=bool(payload.get("query_target_visible", False)),
            frozen_test_accessed=bool(payload.get("frozen_test_accessed", False)),
        )


@dataclass(frozen=True, slots=True)
class ConditioningSpec:
    kind: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.kind not in CONDITIONING_KINDS:
            raise ValueError(f"unsupported conditioning kind: {self.kind}")
        payload = dict(self.payload)
        if self.kind == "language_prompt":
            _require_keys(payload, {"template_id", "output_schema", "deny_list"}, self.kind)
        elif self.kind == "time_window":
            _require_keys(
                payload,
                {"timestamps", "frequency", "history_length", "prediction_length"},
                self.kind,
            )
            timestamps = list(payload["timestamps"])
            if not timestamps or not _is_monotonic(timestamps):
                raise ValueError("time_window timestamps must be strictly increasing")
            if int(payload["history_length"]) != len(timestamps):
                raise ValueError("time_window history_length must match timestamps")
            if int(payload["prediction_length"]) <= 0:
                raise ValueError("time_window prediction_length must be positive")
            lowered = {str(key).lower() for key in payload}
            if {"future_target", "future_targets", "future_label"} & lowered:
                raise ValueError("time_window cannot contain a future target")
        elif self.kind == "support_set":
            _require_keys(
                payload,
                {"context_group_hash", "query_group_hash", "feature_names", "label_names"},
                self.kind,
            )
            if payload["context_group_hash"] == payload["query_group_hash"]:
                raise ValueError("support-set context and query groups must be isolated")
        elif self.kind == "depth_window":
            _require_keys(
                payload,
                {"coordinates", "unit", "center_index", "window_length", "feature_names"},
                self.kind,
            )
            coordinates = list(payload["coordinates"])
            if len(coordinates) != int(payload["window_length"]):
                raise ValueError("depth_window coordinates must match window_length")
            if not _is_monotonic(coordinates):
                raise ValueError("depth_window coordinates must be strictly increasing")
            center = int(payload["center_index"])
            if center < 0 or center >= len(coordinates):
                raise ValueError("depth_window center_index is out of range")
        elif self.kind == "spatial_prompt":
            _require_keys(
                payload,
                {"prompt_kind", "prompt_source", "coordinate_frame", "split_role"},
                self.kind,
            )
            prompt_kind = str(payload["prompt_kind"])
            if prompt_kind not in {
                "none",
                "points_2d",
                "mask_2d",
                "points_3d",
                "mask_3d",
            }:
                raise ValueError(f"unsupported spatial prompt kind: {prompt_kind}")
            source = str(payload["prompt_source"]).lower()
            split_role = str(payload["split_role"]).lower()
            leaked_sources = ("ground_truth", "groundtruth", "target", "label", "fault_stick")
            if split_role != "train" and any(token in source for token in leaked_sources):
                raise ValueError("validation/inference spatial prompts cannot be target-derived")
            if "box" in prompt_kind:
                raise ValueError("box prompts are not approved by the 3-D prompt contract")
        elif self.kind == "masked_volume":
            _require_keys(
                payload,
                {"axis_order", "spacing", "active_mask_semantics", "observation_visibility"},
                self.kind,
            )
            if tuple(payload["axis_order"]) != ("K", "J", "I"):
                raise ValueError("masked_volume axis_order must be K,J,I")
            if len(tuple(payload["spacing"])) != 3:
                raise ValueError("masked_volume spacing must contain three values")
            visibility = str(payload["observation_visibility"]).lower()
            if payload.get("mode") == "strict" and any(
                token in visibility for token in ("target", "truth", "label", "eclipse_poro")
            ):
                raise ValueError("strict volume requests cannot use target-derived observations")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "payload": dict(self.payload)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConditioningSpec":
        return cls(kind=str(payload["kind"]), payload=dict(payload["payload"]))


@dataclass(frozen=True, slots=True)
class PromotionGate:
    metric: str
    direction: str
    baseline_id: str
    required_relative_improvement: float
    minimum_winning_folds: int
    minimum_completed_fraction: float
    controls: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("promotion direction must be minimize or maximize")
        if self.required_relative_improvement < 0:
            raise ValueError("required_relative_improvement cannot be negative")
        if self.minimum_winning_folds <= 0:
            raise ValueError("minimum_winning_folds must be positive")
        if not 0 < self.minimum_completed_fraction <= 1:
            raise ValueError("minimum_completed_fraction must be in (0, 1]")
        if "random_init_same_architecture" not in self.controls:
            raise ValueError("foundation promotion requires a random-init control")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["controls"] = list(self.controls)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotionGate":
        values = dict(payload)
        values["controls"] = tuple(values["controls"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class FoundationTaskEnvelope:
    track_id: str
    task_type: str
    axis_kind: str
    model: FoundationModelRef
    split_hash: str
    sample_ids_hash: str
    input_schema: tuple[TensorFieldSpec, ...]
    target_schema: tuple[TensorFieldSpec, ...]
    visibility: VisibilityPolicy
    conditioning: ConditioningSpec
    output_schema: Mapping[str, Any]
    physical_constraints: Mapping[str, Any]
    uncertainty: Mapping[str, Any]
    fallback: Mapping[str, Any]
    promotion_gate: PromotionGate
    state: str = "CONNECTED_UNVERIFIED"
    schema_version: str = FOUNDATION_SCHEMA_VERSION
    prompt_version: str = FOUNDATION_PROMPT_VERSION
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.schema_version != FOUNDATION_SCHEMA_VERSION:
            raise ValueError("unsupported foundation task schema_version")
        if self.prompt_version != FOUNDATION_PROMPT_VERSION:
            raise ValueError("unsupported foundation prompt_version")
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"unsupported foundation task_type: {self.task_type}")
        if self.axis_kind not in AXIS_KINDS:
            raise ValueError(f"unsupported axis_kind: {self.axis_kind}")
        if self.state not in FOUNDATION_STATES:
            raise ValueError(f"unsupported foundation state: {self.state}")
        _require_hex(self.split_hash, 64, "split_hash")
        _require_hex(self.sample_ids_hash, 64, "sample_ids_hash")
        if not self.input_schema or not self.target_schema:
            raise ValueError("foundation task requires input and target schemas")
        expected_axis = {
            "time_forecasting": "time",
            "tabular_regression": "none",
            "depth_classification": "measured_depth",
            "segmentation_2d": "image_xy",
            "segmentation_3d": "volume_kji",
            "volume_regression_3d": "volume_kji",
        }[self.task_type]
        if self.axis_kind != expected_axis:
            raise ValueError(f"{self.task_type} requires axis_kind={expected_axis}")
        expected_conditioning = {
            "time_forecasting": "time_window",
            "tabular_regression": "support_set",
            "depth_classification": "depth_window",
            "segmentation_2d": "spatial_prompt",
            "segmentation_3d": "spatial_prompt",
            "volume_regression_3d": "masked_volume",
        }[self.task_type]
        if self.conditioning.kind != expected_conditioning:
            raise ValueError(
                f"{self.task_type} requires {expected_conditioning} conditioning"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "prompt_version": self.prompt_version,
            "track_id": self.track_id,
            "task_type": self.task_type,
            "axis_kind": self.axis_kind,
            "model": self.model.to_dict(),
            "split_hash": self.split_hash,
            "sample_ids_hash": self.sample_ids_hash,
            "input_schema": [item.to_dict() for item in self.input_schema],
            "target_schema": [item.to_dict() for item in self.target_schema],
            "visibility": self.visibility.to_dict(),
            "conditioning": self.conditioning.to_dict(),
            "output_schema": dict(self.output_schema),
            "physical_constraints": dict(self.physical_constraints),
            "uncertainty": dict(self.uncertainty),
            "fallback": dict(self.fallback),
            "promotion_gate": self.promotion_gate.to_dict(),
            "state": self.state,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FoundationTaskEnvelope":
        return cls(
            schema_version=str(payload["schema_version"]),
            prompt_version=str(payload["prompt_version"]),
            track_id=str(payload["track_id"]),
            task_type=str(payload["task_type"]),
            axis_kind=str(payload["axis_kind"]),
            model=FoundationModelRef.from_dict(payload["model"]),
            split_hash=str(payload["split_hash"]),
            sample_ids_hash=str(payload["sample_ids_hash"]),
            input_schema=tuple(
                TensorFieldSpec.from_dict(item) for item in payload["input_schema"]
            ),
            target_schema=tuple(
                TensorFieldSpec.from_dict(item) for item in payload["target_schema"]
            ),
            visibility=VisibilityPolicy.from_dict(payload["visibility"]),
            conditioning=ConditioningSpec.from_dict(payload["conditioning"]),
            output_schema=dict(payload["output_schema"]),
            physical_constraints=dict(payload["physical_constraints"]),
            uncertainty=dict(payload["uncertainty"]),
            fallback=dict(payload["fallback"]),
            promotion_gate=PromotionGate.from_dict(payload["promotion_gate"]),
            state=str(payload.get("state", "CONNECTED_UNVERIFIED")),
            notes=tuple(payload.get("notes", ())),
        )

    def request_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def transition(self, target_state: str) -> "FoundationTaskEnvelope":
        if target_state not in _STATE_TRANSITIONS[self.state]:
            raise ValueError(f"invalid foundation state transition: {self.state} -> {target_state}")
        return replace(self, state=target_state)


def load_foundation_routes(
    path: Path | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    route_path = path or Path(__file__).with_name("foundation_routes.v1.json")
    payload = json.loads(Path(route_path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != FOUNDATION_ROUTE_SCHEMA_VERSION:
        raise ValueError("unsupported foundation route manifest")
    if payload.get("prompt_version") != FOUNDATION_PROMPT_VERSION:
        raise ValueError("foundation route prompt version mismatch")
    routes = payload.get("routes")
    if not isinstance(routes, list):
        raise ValueError("foundation routes must be a list")
    by_track: dict[str, Mapping[str, Any]] = {}
    for route in routes:
        track_id = str(route["track_id"])
        if track_id in by_track:
            raise ValueError(f"duplicate foundation route for {track_id}")
        FoundationModelRef.from_dict(route["model"])
        if route.get("state") not in FOUNDATION_STATES:
            raise ValueError(f"unsupported route state for {track_id}")
        by_track[track_id] = dict(route)
    return by_track
