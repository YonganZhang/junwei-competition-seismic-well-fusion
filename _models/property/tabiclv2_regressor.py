"""Fail-closed TabICLv2 adapter with explicit local-checkpoint gate."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _code.ml_framework.contracts import TaskSpec
from _models.property._p5_common import (
    IndependentTargetEstimatorAdapter,
    require_approved_weight,
    require_model_dependencies,
)


model_id = "tabiclv2_regressor"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["regression"],
        "input_modalities": ["tabular"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
        "supported_losses": ["not_applicable"],
        "supported_output_activations": ["not_applicable"],
        "requires_pretrained_weight": True,
        "auto_download": False,
        "target_strategy": "three independent in-context regressors",
    }


class TabICLPropertyAdapter(IndependentTargetEstimatorAdapter):
    def __init__(self, *, tabicl_module: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tabicl_module = tabicl_module

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for target, estimator in self.estimators.items():
            estimator.save(
                path / f"{target}.pkl",
                save_model_weights=False,
                save_training_data=True,
                save_kv_cache=False,
            )
        (path / "metadata.json").write_text(
            json.dumps({"model_id": self.model_id, "targets": list(self.task_spec.targets)}, indent=2),
            encoding="utf-8",
        )

    def load_checkpoint(self, path: Path) -> None:
        metadata_payload = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if metadata_payload != {"model_id": self.model_id, "targets": list(self.task_spec.targets)}:
            raise ValueError("TabICL checkpoint identity mismatch")
        self.estimators = {
            target: self.tabicl_module.TabICLRegressor.load(path / f"{target}.pkl")
            for target in self.task_spec.targets
        }


def build_model(task_spec: TaskSpec, **config: Any) -> TabICLPropertyAdapter:
    seed = int(config.get("seed", 2693))
    n_estimators = int(config.get("n_estimators", 2))
    batch_size = int(config.get("batch_size", 1))
    offload_mode = config.get("offload_mode", "auto")
    if n_estimators <= 0 or batch_size <= 0:
        raise ValueError("TabICL n_estimators and batch_size must be positive")
    if offload_mode not in {True, False, "auto", "gpu", "cpu", "disk"}:
        raise ValueError(
            "TabICL offload_mode must be boolean or one of auto/gpu/cpu/disk"
        )
    checkpoint = require_approved_weight(model_id, config)
    tabicl = require_model_dependencies(model_id)["tabicl"]

    def factory(target: str) -> Any:
        target_offset = task_spec.targets.index(target)
        return tabicl.TabICLRegressor(
            n_estimators=n_estimators,
            batch_size=batch_size,
            model_path=str(checkpoint),
            allow_auto_download=False,
            checkpoint_version="tabicl-regressor-v2-20260212.ckpt",
            device=str(config.get("device", "cpu")),
            offload_mode=offload_mode,
            random_state=seed + target_offset,
            verbose=False,
        )

    return TabICLPropertyAdapter(
        model_id=model_id,
        task_spec=task_spec,
        estimator_factory=factory,
        tabicl_module=tabicl,
    )
