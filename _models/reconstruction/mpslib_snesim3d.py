"""Science/dependency gate for MPSlib SNESIM 3-D reconstruction.

The Volve porosity benchmark has no independently licensed categorical
training image.  Stage 1 therefore stops before importing MPSlib unless an
explicit, external training-image path and provenance approval are supplied.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from _code.ml_framework.contracts import TaskSpec
from _models.reconstruction._p5_adapter import AdapterSkip, reconstruction_mode, require_dependency


model_id = "mpslib_snesim3d"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["reconstruction"],
        "input_modalities": ["training_image", "hard_constraints"],
        "supports_missing_mask": True,
        "supports_uncertainty": True,
        "batch_representation": "categorical_volume",
        "trainable": False,
        "dependency_group": "geostat-cpu",
        "requires_legal_training_image": True,
    }


def build_model(task_spec: TaskSpec, **config: Any) -> Any:
    reconstruction_mode(task_spec)
    training_image = config.get("training_image_path")
    approved = bool(config.get("training_image_provenance_approved", False))
    if not training_image or not approved:
        raise AdapterSkip(
            "missing_legal_training_image",
            "MPSlib is skipped: no independently licensed and approved training image was provided",
            model_id=model_id,
            training_image_path=None if not training_image else str(training_image),
            provenance_approved=approved,
            forbidden_source="frozen Eclipse reference/test volume",
        )
    path = Path(training_image)
    if not path.is_file():
        raise AdapterSkip(
            "training_image_missing", "approved MPS training image path does not exist",
            model_id=model_id, training_image_path=str(path),
        )
    module = require_dependency("mpslib", model_id=model_id, distribution="scikit-mps")
    raise AdapterSkip(
        "mps_continuous_target_adapter_not_approved",
        "MPSlib import is available, but categorical discretization of continuous porosity is not approved",
        model_id=model_id,
        module=getattr(module, "__name__", "mpslib"),
    )
