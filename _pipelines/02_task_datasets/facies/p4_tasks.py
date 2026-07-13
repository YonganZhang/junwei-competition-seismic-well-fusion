"""Strict P4 task declarations for the two independent facies label spaces."""
from __future__ import annotations

from typing import Any, Mapping

from _code.ml_framework.contracts import TaskSpec
from _code.ml_framework.hpo import HPOPlan

from pipeline_contract import get_task_schema


TASK_IDS = ("facies_f3", "facies_penobscot")
DEFAULT_MODEL_ID = "facies_linear_pixel"
DEFAULT_LOSS_ID = "cross_entropy"
LABEL_VERSIONS = {
    "facies_f3": "f3-zenodo-1471548-ids-0-9-v1",
    "facies_penobscot": "penobscot-dataset-log-v3-ids-0-7-v1",
}
INTERNAL_BUFFER_GROUPS = {
    "facies_f3": 25,
    "facies_penobscot": 23,
}
EXPECTED_OUTER_SPLITS = {
    "facies_f3": {
        "development_inline_range": (100, 586),
        "external_guard_inline_range": (587, 619),
        "test_inline_range": (620, 750),
    },
    "facies_penobscot": {
        "development_inline_range": (1000, 1448),
        "external_guard_inline_range": (1449, 1479),
        "test_inline_range": (1480, 1600),
    },
}


def get_task_spec(task_id: str) -> TaskSpec:
    """Return a fully explicit multiclass segmentation contract."""
    schema = get_task_schema(task_id)
    if task_id not in LABEL_VERSIONS:
        raise ValueError(f"unsupported P4 facies task {task_id!r}")
    classes = list(schema.valid_label_ids)
    return TaskSpec(
        track_id="facies",
        task_id=task_id,
        task_type="multiclass",
        input_modalities=("seismic_amplitude_2d",),
        targets=("facies",),
        units={"facies": "class_id"},
        label_version=LABEL_VERSIONS[task_id],
        target_masks={"facies": "all_pixels_valid"},
        group_keys=("inline",),
        spatial_buffer={
            "group_key": "inline",
            "unit": "inline_section",
            "strategy": "permanent_contiguous_buffers_between_cv_core_blocks",
            "buffer_groups": INTERNAL_BUFFER_GROUPS[task_id],
            **EXPECTED_OUTER_SPLITS[task_id],
        },
        target_transform={
            "facies": {"name": "identity_integer_ids", "valid_ids": classes}
        },
        inverse_transform={"facies": {"name": "identity_integer_ids"}},
        train_loss={
            "facies": {
                "default": DEFAULT_LOSS_ID,
                "candidates": (
                    "cross_entropy",
                    "focal",
                    "cross_entropy_plus_dice",
                    "cross_entropy_plus_lovasz",
                ),
                "class_weights": "inverse_sqrt_fold_train_only_clipped_0.2_5.0",
                "input": "raw_logits",
            }
        },
        inference_transform={
            "facies": {"name": "softmax", "dimension": 1, "prediction": "argmax"}
        },
        threshold_policy={
            "facies": {"name": "none", "reason": "closed_set_multiclass_argmax"}
        },
        calibration_policy={
            "facies": {
                "fit_scope": "pooled_oof_only",
                "candidates": ("identity", "temperature_scaling"),
                "test_refit_forbidden": True,
            }
        },
        primary_metrics=("miou",),
        secondary_metrics=("macro_f1", "accuracy", "nll", "brier", "ece"),
        guardrail_metrics=("all_classes_supported", "finite_logits"),
        metric_directions={
            "miou": "maximize",
            "macro_f1": "maximize",
            "accuracy": "maximize",
            "nll": "minimize",
            "brier": "minimize",
            "ece": "minimize",
            "all_classes_supported": "maximize",
            "finite_logits": "maximize",
        },
        hpo=hpo_contract(),
        visualizer_id="facies_archived_dense_diagnostics_v1",
        required_figures=(
            "seismic_gt_prediction_confidence_error",
            "confusion_matrix",
            "per_class_support_f1",
            "reliability_calibration",
        ),
        input_whitelist=("seismic_patch", "inline", "crossline", "time_ms"),
        forbidden_inputs=(
            "facies_label",
            "test_label_distribution",
            "test_metrics",
            "future_prediction",
        ),
        metadata={
            "num_classes": schema.num_classes,
            "valid_label_ids": classes,
            "ignore_index": schema.ignore_index,
            "schema_evidence": schema.source,
            "label_spaces_must_not_be_combined": True,
            "requested_n_splits": 5,
        },
    )


def hpo_contract() -> dict[str, Any]:
    """Return the frozen development-only search direction and budget."""
    plan = HPOPlan()
    return {
        "backend": "optional_optuna_sequential",
        "fixed_baseline_without_optuna": True,
        "primary_metric": "miou",
        "direction": plan.direction,
        "sanity_trials": plan.sanity_trials,
        "pilot_trials": plan.pilot_trials,
        "top_configs": plan.top_configs,
        "confirm_seeds": plan.confirm_seeds,
        "sampler": plan.sampler,
        "pruner": plan.pruner,
        "test_access": "forbidden",
        "search_space": {
            "loss_id": [
                "cross_entropy",
                "focal",
                "cross_entropy_plus_dice",
                "cross_entropy_plus_lovasz",
            ],
            "learning_rate": {"type": "log_float", "low": 1e-4, "high": 3e-3},
            "weight_decay": {"type": "log_float_or_zero", "low": 1e-6, "high": 1e-3},
            "batch_size": [8, 16, 32],
            "focal_gamma": {"type": "float", "low": 1.0, "high": 3.0},
            "compound_weight": {"type": "float", "low": 0.1, "high": 0.7},
        },
        "selection_evidence": ("fold_mean", "fold_std", "worst_fold", "guardrails", "cost"),
    }


def fixed_baseline_config(task_id: str) -> dict[str, Any]:
    """A small deterministic baseline used to prove the complete P4 SOP."""
    spec = get_task_spec(task_id)
    return {
        "task_id": task_id,
        "label_version": spec.label_version,
        "model_id": DEFAULT_MODEL_ID,
        "model_config": {},
        "loss_id": DEFAULT_LOSS_ID,
        "loss_config": {},
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 16,
        "normalization": "zscore",
        "root_seed": 2693,
        "requested_n_splits": 5,
        "metric_direction": "maximize",
        "primary_metric": "miou",
    }


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    """Declare only facies development parameters; never inspect data or test."""
    if task_spec.task_id not in TASK_IDS:
        raise ValueError(f"not a facies TaskSpec: {task_spec.task_id}")
    loss_id = trial.suggest_categorical(
        "loss_id", list(task_spec.train_loss["facies"]["candidates"])
    )
    params: dict[str, Any] = {
        "loss_id": loss_id,
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
    }
    if loss_id == "focal":
        params["focal_gamma"] = trial.suggest_float("focal_gamma", 1.0, 3.0)
    if loss_id in {"cross_entropy_plus_dice", "cross_entropy_plus_lovasz"}:
        params["compound_weight"] = trial.suggest_float("compound_weight", 0.1, 0.7)
    return params
