"""Compatibility package for canonical facies models."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _code.ml_framework.contracts import TaskSpec  # noqa: E402


def compatibility_task_spec(num_classes: int) -> TaskSpec:
    return TaskSpec(
        track_id="facies", task_id="legacy_facies_compatibility", task_type="multiclass",
        input_modalities=("seismic_section",), targets=("facies",), units={"facies": "class_id"},
        label_version="legacy-compat-v1", target_masks={"facies": "all_pixels_valid"},
        group_keys=("legacy_group",), target_transform={"facies": "identity"},
        inverse_transform={"facies": "identity"}, train_loss={"facies": "cross_entropy"},
        inference_transform={"facies": "softmax"}, threshold_policy={}, calibration_policy={},
        primary_metrics=("miou",), metric_directions={"miou": "maximize"},
        visualizer_id="legacy_compatibility", required_figures=("prediction.png",),
        metadata={"num_classes": int(num_classes)},
    )
