"""Canonical one-file model discovery under ``_models/<track>/<model_id>.py``."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Mapping

from .contracts import TaskSpec


@dataclass(frozen=True)
class DiscoveredModel:
    track_id: str
    model_id: str
    module: ModuleType
    capabilities: Mapping[str, Any]

    def build(self, task_spec: TaskSpec, **config: Any) -> Any:
        return self.module.build_model(task_spec, **config)

    def suggest_hparams(self, trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
        suggest = getattr(self.module, "suggest_hparams", None)
        return {} if suggest is None else dict(suggest(trial, task_spec))


def discover_model(track_id: str, model_id: str) -> DiscoveredModel:
    for value, name in ((track_id, "track_id"), (model_id, "model_id")):
        if not value or not value.replace("_", "").isalnum():
            raise ValueError(f"invalid {name}={value!r}")
    module_name = f"_models.{track_id}.{model_id}"
    module = importlib.import_module(module_name)
    missing = [name for name in ("model_id", "build_model", "capabilities") if not hasattr(module, name)]
    if missing:
        raise ValueError(f"{module_name} missing required exports: {missing}")
    if module.model_id != model_id:
        raise ValueError(f"{module_name}.model_id={module.model_id!r} does not match file name {model_id!r}")
    capabilities = module.capabilities()
    if not isinstance(capabilities, Mapping):
        raise TypeError(f"{module_name}.capabilities() must return a mapping")
    required = {"task_types", "input_modalities", "supports_missing_mask", "supports_uncertainty"}
    absent = sorted(required - set(capabilities))
    if absent:
        raise ValueError(f"{module_name} capabilities missing {absent}")
    return DiscoveredModel(track_id, model_id, module, dict(capabilities))
