"""P4 task/model envelopes shared by every track.

The shared layer validates outer semantics only.  Tensor shapes and model heads
remain track-specific and are described by :class:`TaskSpec`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


_TASK_TYPES = {"binary", "multiclass", "regression", "ranking", "survival", "reconstruction"}


def _nonempty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class TaskSpec:
    track_id: str
    task_id: str
    task_type: str
    input_modalities: tuple[str, ...]
    targets: tuple[str, ...]
    units: Mapping[str, str]
    label_version: str
    target_masks: Mapping[str, str]
    group_keys: tuple[str, ...]
    target_transform: Mapping[str, Any]
    inverse_transform: Mapping[str, Any]
    train_loss: Mapping[str, Any]
    inference_transform: Mapping[str, Any]
    threshold_policy: Mapping[str, Any]
    calibration_policy: Mapping[str, Any]
    primary_metrics: tuple[str, ...]
    metric_directions: Mapping[str, str]
    secondary_metrics: tuple[str, ...] = ()
    guardrail_metrics: tuple[str, ...] = ()
    spatial_buffer: Mapping[str, Any] | None = None
    time_cutoff: Mapping[str, Any] | None = None
    hpo: Mapping[str, Any] = field(default_factory=dict)
    visualizer_id: str = ""
    required_figures: tuple[str, ...] = ()
    input_whitelist: tuple[str, ...] = ()
    forbidden_inputs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("track_id", "task_id", "task_type", "label_version"):
            _nonempty_string(getattr(self, name), name)
        if self.task_type not in _TASK_TYPES:
            raise ValueError(f"unsupported task_type={self.task_type!r}; expected one of {sorted(_TASK_TYPES)}")
        if not self.input_modalities:
            raise ValueError("input_modalities must not be empty")
        if not self.targets:
            raise ValueError("targets must not be empty")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("targets must be unique")
        missing_masks = sorted(set(self.targets) - set(self.target_masks))
        if missing_masks:
            raise ValueError(f"target_masks missing targets: {missing_masks}")
        missing_units = sorted(set(self.targets) - set(self.units))
        if missing_units:
            raise ValueError(f"units missing targets: {missing_units}")
        if not self.group_keys:
            raise ValueError("at least one leakage-safe group key is required")
        if not self.primary_metrics:
            raise ValueError("at least one primary metric is required")
        declared_metrics = set(self.primary_metrics) | set(self.secondary_metrics) | set(self.guardrail_metrics)
        missing_directions = sorted(declared_metrics - set(self.metric_directions))
        if missing_directions:
            raise ValueError(f"metric_directions missing metrics: {missing_directions}")
        invalid_directions = {
            metric: direction
            for metric, direction in self.metric_directions.items()
            if direction not in {"maximize", "minimize"}
        }
        if invalid_directions:
            raise ValueError(f"invalid metric directions: {invalid_directions}")
        for field_name in ("target_transform", "inverse_transform", "train_loss", "inference_transform"):
            missing = sorted(set(self.targets) - set(getattr(self, field_name)))
            if missing:
                raise ValueError(f"{field_name} missing targets: {missing}")
        if not self.visualizer_id:
            raise ValueError("visualizer_id must be explicit")
        if not self.required_figures:
            raise ValueError("required_figures must not be empty")
        leaked = sorted(set(self.input_whitelist) & set(self.forbidden_inputs))
        if leaked:
            raise ValueError(f"input whitelist contains forbidden label/future fields: {leaked}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskSpec":
        tuple_fields = {
            "input_modalities", "targets", "group_keys", "primary_metrics", "secondary_metrics",
            "guardrail_metrics", "required_figures", "input_whitelist", "forbidden_inputs",
        }
        values = dict(payload)
        for name in tuple_fields:
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)


@dataclass
class ModelBatch:
    inputs: Mapping[str, Any]
    targets: Mapping[str, Any] | None
    input_masks: Mapping[str, Any]
    target_masks: Mapping[str, Any]
    sample_ids: Sequence[str]
    groups: Mapping[str, Sequence[str]]
    coordinates: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("ModelBatch.inputs must not be empty")
        if not self.sample_ids:
            raise ValueError("ModelBatch.sample_ids must not be empty")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("ModelBatch.sample_ids must be unique")
        sample_count = len(self.sample_ids)
        for key, values in self.groups.items():
            if len(values) != sample_count:
                raise ValueError(f"group {key!r} has {len(values)} entries, expected {sample_count}")
        if self.targets is None and self.target_masks:
            raise ValueError("target_masks must be empty for target-free inference batches")
        if self.targets is not None and set(self.target_masks) != set(self.targets):
            raise ValueError("target_masks keys must exactly match targets keys")


@dataclass
class ModelOutput:
    raw: Mapping[str, Any]
    transformed: Mapping[str, Any] | None = None
    uncertainty: Mapping[str, Any] | None = None
    aux: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.raw:
            raise ValueError("ModelOutput.raw must not be empty")
        for optional_name in ("transformed", "uncertainty"):
            value = getattr(self, optional_name)
            if value is not None and not set(value).issubset(self.raw):
                raise ValueError(f"{optional_name} keys must be a subset of raw output keys")
