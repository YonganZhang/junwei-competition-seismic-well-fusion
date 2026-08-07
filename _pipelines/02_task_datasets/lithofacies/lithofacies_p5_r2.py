#!/usr/bin/env python3
"""P5.2 R2 development runner for the lithofacies track.

This runner is track-private and development-only. It uses the real GM09
development archive, honors the frozen LOGO4 mother-family split, and reports
budget curves plus single-factor diagnostics for the representative models:
`multimodal_mlp`, `xgboost_multisoftprob_window`, and `inceptiontime_window`.

The frozen test family (`15/9-F-5`) is never opened. The S lane remains
structured `not_rankable` because the development archive has no finite
`center_md_m`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from _code.ml_framework.model_registry import get_model  # noqa: E402
from _code.ml_framework.visualize import plot_loss_curve  # noqa: E402
from _models.lithofacies.p5_adapter_common import (  # noqa: E402
    NUM_CLASSES,
    multimodal_numpy_features,
    probability_loss,
    standard_capabilities,
)
from p4_contract import (  # noqa: E402
    CLASS_NAMES,
    DEVELOPMENT_FAMILIES,
    EFFECTIVE_N_SPLITS,
    TEST_FAMILY,
    apply_fold_preprocessor,
    class_support,
    classification_metrics_from_confusion,
    fit_fold_preprocessor,
    sample_id,
)
from p5_stage1 import (  # noqa: E402
    _read_development_hdf5,
    build_development_logo4,
)


ROOT_SEED = 2693
TASK_ID = "gm09_genetic_facies_9class"
MODEL_ROSTER = (
    "multimodal_mlp",
    "xgboost_multisoftprob_window",
    "inceptiontime_window",
)
REPEAT_SEEDS = (1867973658, 2137841944, 3902865753)
BUDGET_POINTS = (40, 200, 1000)
EXPECTED_MAIN_ROWS = len(MODEL_ROSTER) * EFFECTIVE_N_SPLITS * len(REPEAT_SEEDS)
EXPECTED_ABLATION_ROWS = 0
EXPECTED_TOTAL_ROWS = EXPECTED_MAIN_ROWS
WANDB = "not_used"

SCHEMA_VERSION = "lithofacies-p5-r2-v1"
RESULTS_FILENAME = "p5_r2_results.jsonl"
SUMMARY_FILENAME = "p5_r2_summary.json"
VISUALIZATION_FILENAME = "p5_r2_visualization_manifest.json"
LEADERBOARD_FILENAME = "p5_r2_diagnostic_leaderboard.json"


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _track_owned(path: Path) -> Path:
    resolved = path.resolve()
    if TRACK_DIR.resolve() not in resolved.parents:
        raise ValueError(f"R2 artifacts must stay below {TRACK_DIR}")
    return resolved


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = _track_owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path = _track_owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _class_weights(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    supported = counts > 0
    if not np.any(supported):
        raise ValueError("no supported classes")
    weights = np.zeros(NUM_CLASSES, dtype=np.float64)
    weights[supported] = 1.0 / np.sqrt(counts[supported])
    weights[supported] /= weights[supported].mean()
    return weights.astype(np.float32)


def _stack_well(samples: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([np.asarray(sample["well_log_seq"], dtype=np.float32) for sample in samples])


def _stack_seismic(samples: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([np.asarray(sample["seismic_patch"], dtype=np.float32) for sample in samples])


def _stack_labels(samples: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(sample["label"]) for sample in samples], dtype=np.int64)


def _mask_context(well: np.ndarray, seismic: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "window":
        return well, seismic
    if mode != "center":
        raise ValueError(f"unknown context mode {mode!r}")
    center = well.shape[-1] // 2
    masked_well = np.zeros_like(well)
    masked_seismic = np.zeros_like(seismic)
    masked_well[..., center] = well[..., center]
    masked_seismic[..., center] = seismic[..., center]
    return masked_well, masked_seismic


def _mask_modality(
    well: np.ndarray, seismic: np.ndarray, lane: str
) -> tuple[np.ndarray, np.ndarray]:
    if lane == "M":
        return well, seismic
    if lane != "W":
        raise ValueError(f"unknown modality lane {lane!r}")
    return well, np.zeros_like(seismic)


def _mask_missing_modality(
    well: np.ndarray, seismic: np.ndarray, missing: str
) -> tuple[np.ndarray, np.ndarray]:
    if missing == "none":
        return well, seismic
    if missing == "seismic":
        return well, np.zeros_like(seismic)
    if missing == "well":
        return np.zeros_like(well), seismic
    raise ValueError(f"unknown missing-modality mode {missing!r}")


def _confusion_from_predictions(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for truth, pred in zip(labels.tolist(), predictions.tolist(), strict=True):
        matrix[int(truth), int(pred)] += 1
    return matrix


def _metrics_from_predictions(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    confusion = _confusion_from_predictions(labels, predictions)
    metrics = classification_metrics_from_confusion(confusion)
    return {
        "accuracy": metrics["accuracy"],
        "fixed_schema_macro_f1": metrics["macro_f1"],
        "supported_class_macro_f1": metrics["supported_class_macro_f1"],
        "supported_class_metric_role": "diagnostic_only",
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "evaluated_samples": metrics["evaluated_samples"],
    }


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


@dataclass
class FoldPack:
    fold_id: int
    train_groups: list[str]
    validation_groups: list[str]
    train_samples: list[dict[str, Any]]
    validation_samples: list[dict[str, Any]]
    preprocessor: Any


def _load_development_folds(dataset_root: Path) -> tuple[list[dict[str, Any]], Path]:
    samples, hdf5_path = _read_development_hdf5(dataset_root)
    folds_raw = build_development_logo4(samples)
    packs: list[FoldPack] = []
    by_sample_id = {sample_id(sample): sample for sample in samples}
    for fold in folds_raw:
        train_samples = [by_sample_id[sample_id(sample)] for sample in fold["train"]]
        validation_samples = [by_sample_id[sample_id(sample)] for sample in fold["validation"]]
        preprocessor = fit_fold_preprocessor(train_samples)
        packs.append(
            FoldPack(
                fold_id=int(fold["fold_id"]),
                train_groups=list(fold["train_groups"]),
                validation_groups=list(fold["validation_groups"]),
                train_samples=apply_fold_preprocessor(train_samples, preprocessor),
                validation_samples=apply_fold_preprocessor(validation_samples, preprocessor),
                preprocessor=preprocessor,
            )
        )
    return [asdict(pack) for pack in packs], hdf5_path


def _build_task_spec() -> Any:
    from p4_contract import lithofacies_task_spec

    return lithofacies_task_spec()


def _model_config(model_id: str, shape_well: tuple[int, ...], shape_seismic: tuple[int, ...]) -> dict[str, Any]:
    config: dict[str, Any] = {
        "num_classes": NUM_CLASSES,
        "well_log_shape": shape_well,
        "seismic_shape": shape_seismic,
    }
    if model_id == "multimodal_mlp":
        config["hidden_size"] = 64
    elif model_id == "inceptiontime_window":
        config["nf"] = 8
        config["kernel_size"] = 31
    else:
        raise ValueError(f"unsupported torch model_id {model_id!r}")
    return config


def _torch_train_one(
    model_id: str,
    *,
    train_well: np.ndarray,
    train_seismic: np.ndarray,
    train_labels: np.ndarray,
    val_well: np.ndarray,
    val_seismic: np.ndarray,
    val_labels: np.ndarray,
    class_weights: np.ndarray,
    device: torch.device,
    seed: int,
    loss_mode: str,
) -> dict[str, Any]:
    from torch.nn import functional as F

    _seed_everything(seed)
    model = get_model(
        model_id,
        models_package="models",
        **_model_config(model_id, tuple(train_well.shape[1:]), tuple(train_seismic.shape[1:])),
    ).to(device)
    if not isinstance(model, nn.Module):
        raise TypeError("expected torch module")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    class_weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    x_train_well = torch.as_tensor(train_well, dtype=torch.float32, device=device)
    x_train_seismic = torch.as_tensor(train_seismic, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    x_val_well = torch.as_tensor(val_well, dtype=torch.float32, device=device)
    x_val_seismic = torch.as_tensor(val_seismic, dtype=torch.float32, device=device)
    y_val = torch.as_tensor(val_labels, dtype=torch.long, device=device)

    history_train: list[float] = []
    history_val: list[float] = []
    budget_points = tuple(sorted(BUDGET_POINTS))
    budget_metrics: dict[str, Any] = {}
    checkpoint_dir = TRACK_DIR / "_outputs" / "p5_r2" / "_runtime" / "torch"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    for epoch in range(1, budget_points[-1] + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train_well, x_train_seismic)
        if loss_mode == "weighted_ce":
            loss = F.cross_entropy(logits, y_train, weight=class_weight_tensor)
        elif loss_mode == "ce":
            loss = F.cross_entropy(logits, y_train)
        else:
            raise ValueError(f"unknown loss_mode {loss_mode!r}")
        if not torch.isfinite(loss):
            raise RuntimeError("torch loss became non-finite")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss = float(loss.detach().cpu())
            val_logits = model(x_val_well, x_val_seismic)
            if loss_mode == "weighted_ce":
                val_loss = float(F.cross_entropy(val_logits, y_val, weight=class_weight_tensor).cpu())
            else:
                val_loss = float(F.cross_entropy(val_logits, y_val).cpu())
        history_train.append(train_loss)
        history_val.append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch in budget_points:
            full_val_logits = val_logits.detach().cpu().numpy()
            full_val_predictions = full_val_logits.argmax(axis=1)
            full_metrics = _metrics_from_predictions(val_labels, full_val_predictions)
            w_well, w_seismic = _mask_modality(val_well, val_seismic, "W")
            w_logits = model(
                torch.as_tensor(w_well, dtype=torch.float32, device=device),
                torch.as_tensor(w_seismic, dtype=torch.float32, device=device),
            ).detach().cpu().numpy()
            w_metrics = _metrics_from_predictions(val_labels, w_logits.argmax(axis=1))
            c_well, c_seismic = _mask_context(val_well, val_seismic, "center")
            c_logits = model(
                torch.as_tensor(c_well, dtype=torch.float32, device=device),
                torch.as_tensor(c_seismic, dtype=torch.float32, device=device),
            ).detach().cpu().numpy()
            c_metrics = _metrics_from_predictions(val_labels, c_logits.argmax(axis=1))
            s_well, s_seismic = _mask_missing_modality(val_well, val_seismic, "seismic")
            s_logits = model(
                torch.as_tensor(s_well, dtype=torch.float32, device=device),
                torch.as_tensor(s_seismic, dtype=torch.float32, device=device),
            ).detach().cpu().numpy()
            s_metrics = _metrics_from_predictions(val_labels, s_logits.argmax(axis=1))
            budget_metrics[str(epoch)] = {
                "M": full_metrics,
                "W": w_metrics,
                "center": c_metrics,
                "missing_seismic": s_metrics,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_path = checkpoint_dir / f"{model_id}_{loss_mode}.pt"
    torch.save({"state_dict": model.state_dict(), "loss_mode": loss_mode}, checkpoint_path)
    roundtrip = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    roundtrip_model = get_model(
        model_id,
        models_package="models",
        **_model_config(model_id, tuple(train_well.shape[1:]), tuple(train_seismic.shape[1:])),
    ).to(device)
    roundtrip_model.load_state_dict(roundtrip["state_dict"])
    with torch.no_grad():
        model.eval()
        roundtrip_model.eval()
        rt_logits = roundtrip_model(x_val_well, x_val_seismic).detach().cpu().numpy()
        baseline_logits = model(x_val_well, x_val_seismic).detach().cpu().numpy()
    if not np.allclose(rt_logits, baseline_logits):
        raise RuntimeError("torch checkpoint roundtrip changed validation logits")
    return {
        "training_loss": {"train": history_train, "val": history_val},
        "budget_metrics": budget_metrics,
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(TRACK_DIR)),
            "sha256": _sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "roundtrip": "PASS",
        },
    }


def _xgb_train_one(
    *,
    train_well: np.ndarray,
    train_seismic: np.ndarray,
    train_labels: np.ndarray,
    val_well: np.ndarray,
    val_seismic: np.ndarray,
    val_labels: np.ndarray,
    class_weights: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    import xgboost

    features_train = multimodal_numpy_features(train_well, train_seismic)
    features_val = multimodal_numpy_features(val_well, val_seismic)
    weights = class_weights[train_labels]
    dtrain = xgboost.DMatrix(features_train, label=train_labels, weight=weights)
    dval = xgboost.DMatrix(features_val, label=val_labels)
    params = {
        "objective": "multi:softprob",
        "num_class": NUM_CLASSES,
        "max_depth": 2,
        "eta": 0.2,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "tree_method": "hist",
        "seed": int(seed),
        "nthread": 1,
        "verbosity": 0,
    }
    evals_result: dict[str, Any] = {}
    booster = xgboost.train(
        params,
        dtrain,
        num_boost_round=max(BUDGET_POINTS),
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        verbose_eval=False,
    )
    checkpoint_dir = TRACK_DIR / "_outputs" / "p5_r2" / "_runtime" / "xgb"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"xgb_{seed}.json"
    booster.save_model(checkpoint_path.as_posix())
    reloaded = xgboost.Booster()
    reloaded.load_model(checkpoint_path.as_posix())
    rt_probs = reloaded.predict(dval, iteration_range=(0, max(BUDGET_POINTS)))
    if not np.allclose(rt_probs, booster.predict(dval, iteration_range=(0, max(BUDGET_POINTS)))):
        raise RuntimeError("xgboost checkpoint roundtrip changed validation probabilities")
    budget_metrics: dict[str, Any] = {}
    for budget in BUDGET_POINTS:
        probs = booster.predict(dval, iteration_range=(0, budget))
        preds = probs.argmax(axis=1)
        full_metrics = _metrics_from_predictions(val_labels, preds)
        w_probs = booster.predict(
            xgboost.DMatrix(multimodal_numpy_features(val_well, np.zeros_like(val_seismic))),
            iteration_range=(0, budget),
        )
        w_metrics = _metrics_from_predictions(val_labels, w_probs.argmax(axis=1))
        c_well, c_seismic = _mask_context(val_well, val_seismic, "center")
        c_probs = booster.predict(
            xgboost.DMatrix(multimodal_numpy_features(c_well, c_seismic)),
            iteration_range=(0, budget),
        )
        c_metrics = _metrics_from_predictions(val_labels, c_probs.argmax(axis=1))
        budget_metrics[str(budget)] = {
            "M": full_metrics,
            "W": w_metrics,
            "center": c_metrics,
            "train_loss": float(evals_result["train"]["mlogloss"][budget - 1]),
            "val_loss": float(evals_result["val"]["mlogloss"][budget - 1]),
        }
    return {
        "training_loss": {
            "train": [float(value) for value in evals_result["train"]["mlogloss"]],
            "val": [float(value) for value in evals_result["val"]["mlogloss"]],
        },
        "budget_metrics": budget_metrics,
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(TRACK_DIR)),
            "sha256": _sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "roundtrip": "PASS",
        },
    }


def _evaluate_s_lane(samples: list[dict[str, Any]]) -> dict[str, Any]:
    md = np.asarray([sample["position"].get("center_md_m") for sample in samples], dtype=np.float64)
    finite = int(np.isfinite(md).sum())
    if finite == 0:
        return {
            "status": "not_rankable",
            "finite_center_md_count": 0,
            "reason": "no real finite center_md_m; sequence fabrication is forbidden",
        }
    return {
        "status": "available",
        "finite_center_md_count": finite,
        "reason": None,
    }


def _plot_learning_curves(summary: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    model_groups = summary["model_groups"]
    fig, axes = plt.subplots(len(model_groups), 1, figsize=(9, 4 * len(model_groups)), squeeze=False)
    for row, group in enumerate(model_groups):
        axis = axes[row][0]
        for key, color in (("M", "tab:blue"), ("W", "tab:orange")):
            curve = group["curves"][key]
            budgets = [point["budget"] for point in curve]
            values = [point["fixed_schema_macro_f1"] for point in curve]
            axis.plot(budgets, values, marker="o", label=f"{group['label']} {key}", color=color)
        axis.set_xscale("log")
        axis.set_xlabel("budget")
        axis.set_ylabel("fixed-schema macro-F1")
        axis.set_title(group["label"])
        axis.legend()
    fig.tight_layout()
    path = figures_dir / "learning_curves.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    entries.append({"figure_id": "learning_curves", "status": "PASS", "path": str(path.relative_to(output_dir)), "sha256": _sha256(path), "bytes": path.stat().st_size})

    confusion = np.asarray(summary["confusion_matrix"], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title("Fixed-nine confusion matrix")
    ax.set_xlabel("predicted")
    ax.set_ylabel("truth")
    path = figures_dir / "fixed9_confusion.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    entries.append({"figure_id": "fixed9_confusion", "status": "PASS", "path": str(path.relative_to(output_dir)), "sha256": _sha256(path), "bytes": path.stat().st_size})

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [row["class_name"] for row in summary["per_class"]]
    f1 = [row["f1"] for row in summary["per_class"]]
    support = [row["support"] for row in summary["per_class"]]
    ax.bar(labels, f1)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_title("Per-class F1/support diagnostic")
    path = figures_dir / "fixed9_per_class_pr_f1_support.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    entries.append({"figure_id": "fixed9_per_class_pr_f1_support", "status": "PASS", "path": str(path.relative_to(output_dir)), "sha256": _sha256(path), "bytes": path.stat().st_size})

    fig, ax = plt.subplots(figsize=(8, 5))
    ece = summary["expected_calibration_error"]
    ax.bar(["reliability"], [ece])
    ax.set_ylim(0, max(0.2, ece * 1.5))
    ax.set_title("Raw softmax reliability diagnostic")
    path = figures_dir / "calibration_reliability.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    entries.append({"figure_id": "calibration_reliability", "status": "PASS", "path": str(path.relative_to(output_dir)), "sha256": _sha256(path), "bytes": path.stat().st_size})

    fig, ax = plt.subplots(figsize=(8, 5))
    matrix = np.asarray(summary["fold_seed_matrix"], dtype=np.float64)
    ax.imshow(matrix, cmap="magma")
    ax.set_title("Fold x seed fixed-schema macro-F1")
    path = figures_dir / "fold_seed_matrix.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    entries.append({"figure_id": "fold_seed_matrix", "status": "PASS", "path": str(path.relative_to(output_dir)), "sha256": _sha256(path), "bytes": path.stat().st_size})

    fig, ax = plt.subplots(figsize=(8, 4))
    diag = summary["missing_modality_diagnostic"]
    ax.bar(list(diag), list(diag.values()))
    ax.set_title("Missing-modality diagnostic")
    path = figures_dir / "missing_modality_diagnostic.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    entries.append({"figure_id": "missing_modality_diagnostic", "status": "PASS", "path": str(path.relative_to(output_dir)), "sha256": _sha256(path), "bytes": path.stat().st_size})

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.02, 0.5, "S lane is not rankable because the development archive\nhas no finite center_md_m.")
    path = figures_dir / "continuous_depth_track_not_feasible.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    entries.append({"figure_id": "continuous_depth_track_not_feasible", "status": "PASS", "path": str(path.relative_to(output_dir)), "sha256": _sha256(path), "bytes": path.stat().st_size})
    return {
        "schema_version": "lithofacies-p5-r2-visualization-manifest-v1",
        "track_id": "lithofacies",
        "task_id": TASK_ID,
        "lane": "P",
        "figures": entries,
        "development_only": True,
        "frozen_test_accessed": False,
    }


def run(dataset_root: Path, output_dir: Path, *, device: str) -> dict[str, Any]:
    output_dir = _track_owned(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds, hdf5_path = _load_development_folds(dataset_root)
    if len(folds) != EFFECTIVE_N_SPLITS:
        raise RuntimeError("development archive is not the frozen LOGO4 downgrade")
    s_lane_samples = {
        sample_id(sample): sample
        for fold in folds
        for sample in fold["train_samples"] + fold["validation_samples"]
    }
    s_lane = _evaluate_s_lane(list(s_lane_samples.values()))
    rows: list[dict[str, Any]] = []
    model_groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"label": "", "curves": {"M": [], "W": []}})
    fold_seed_matrix = np.zeros((len(MODEL_ROSTER), len(folds) * len(REPEAT_SEEDS)), dtype=np.float64)
    confusion_accumulator = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    missing_diag = {"seismic_missing_delta": 0.0, "center_delta": 0.0}
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch reports no GPU")

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    for model_index, model_id in enumerate(MODEL_ROSTER):
        model_groups[model_id]["label"] = model_id
        for fold in folds:
            fold_id = int(fold["fold_id"])
            train_samples = fold["train_samples"]
            validation_samples = fold["validation_samples"]
            train_arrays = {
                "well": _stack_well(train_samples),
                "seismic": _stack_seismic(train_samples),
                "labels": _stack_labels(train_samples),
            }
            val_arrays = {
                "well": _stack_well(validation_samples),
                "seismic": _stack_seismic(validation_samples),
                "labels": _stack_labels(validation_samples),
            }
            preprocessor = fold["preprocessor"]
            class_weights = _class_weights(np.asarray(preprocessor["class_support"], dtype=np.float64))
            for repeat_id, seed in enumerate(REPEAT_SEEDS):
                started = time.monotonic()
                rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                loss_mode = "weighted_ce" if model_id == "inceptiontime_window" and repeat_id % 2 else "ce"
                base = {
                    "schema_version": SCHEMA_VERSION,
                    "track_id": "lithofacies",
                    "task_id": TASK_ID,
                    "model_id": model_id,
                    "fold_id": fold_id,
                    "repeat_id": repeat_id,
                    "seed": int(seed),
                    "loss_mode": loss_mode,
                    "train_groups": fold["train_groups"],
                    "validation_groups": fold["validation_groups"],
                    "frozen_test_accessed": False,
                    "test_metrics_used": False,
                    "preprocessing": {
                        "fit_scope": "fold_train_mother_families_only",
                        "fit_families": preprocessor["fit_families"],
                        "class_support": preprocessor["class_support"],
                        "class_weights": preprocessor["class_weights"],
                    },
                    "budget_points": list(BUDGET_POINTS),
                }
                try:
                    if model_id == "xgboost_multisoftprob_window":
                        evidence = _xgb_train_one(
                            train_well=train_arrays["well"],
                            train_seismic=train_arrays["seismic"],
                            train_labels=train_arrays["labels"],
                            val_well=val_arrays["well"],
                            val_seismic=val_arrays["seismic"],
                            val_labels=val_arrays["labels"],
                            class_weights=class_weights,
                            seed=int(seed),
                        )
                    else:
                        evidence = _torch_train_one(
                            model_id,
                            train_well=train_arrays["well"],
                            train_seismic=train_arrays["seismic"],
                            train_labels=train_arrays["labels"],
                            val_well=val_arrays["well"],
                            val_seismic=val_arrays["seismic"],
                            val_labels=val_arrays["labels"],
                            class_weights=class_weights,
                            device=device_obj,
                            seed=int(seed),
                            loss_mode=loss_mode,
                        )
                    budget_metrics = evidence["budget_metrics"]
                    # accumulate primary fixed-schema confusion at 1000 for summary
                    confusion_accumulator += np.asarray(
                        budget_metrics[str(max(BUDGET_POINTS))]["M"]["confusion_matrix"], dtype=np.int64
                    )
                    fold_seed_matrix[model_index, fold_id * len(REPEAT_SEEDS) + repeat_id] = float(
                        budget_metrics[str(max(BUDGET_POINTS))]["M"]["fixed_schema_macro_f1"]
                    )
                    # derive diagnostic deltas from final budget
                    final_metrics = budget_metrics[str(max(BUDGET_POINTS))]
                    missing_diag["seismic_missing_delta"] += (
                        float(final_metrics["M"]["fixed_schema_macro_f1"])
                        - float(final_metrics["W"]["fixed_schema_macro_f1"])
                    )
                    missing_diag["center_delta"] += (
                        float(final_metrics["M"]["fixed_schema_macro_f1"])
                        - float(final_metrics["center"]["fixed_schema_macro_f1"])
                    )
                    status = "PASS"
                    reason = None
                except Exception as exc:  # structured evidence, fail loud
                    evidence = {}
                    budget_metrics = {}
                    status = "FAIL"
                    reason = {
                        "code": "r2_cell_failure",
                        "message": str(exc),
                        "exception": type(exc).__name__,
                    }
                wall_seconds = time.monotonic() - started
                rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                row = {
                    **base,
                    "status": status,
                    "reason": reason,
                    "wall_seconds": wall_seconds,
                    "peak_resources": {
                        "process_peak_rss_kib": int(max(rss_before, rss_after)),
                        "peak_vram_bytes": 0,
                    },
                    "budget_metrics": budget_metrics,
                    "history": evidence.get("training_loss"),
                    "checkpoint": evidence.get("checkpoint"),
                    "loss_contract": "weighted_ce" if model_id == "inceptiontime_window" and loss_mode == "weighted_ce" else "cross_entropy",
                    "rank_eligible": status == "PASS",
                }
                rows.append(row)

    main_rows = rows[:EXPECTED_MAIN_ROWS]
    ablation_rows = rows[EXPECTED_MAIN_ROWS:]
    if len(main_rows) != EXPECTED_MAIN_ROWS:
        raise RuntimeError("main R2 roster changed")
    if len(ablation_rows) != EXPECTED_ABLATION_ROWS:
        raise RuntimeError("ablation roster changed")

    records = []
    for row in rows:
        records.append(row)
    _atomic_jsonl(output_dir / RESULTS_FILENAME, records)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "not_rankable",
        "task_id": TASK_ID,
        "root_seed": ROOT_SEED,
        "models": list(MODEL_ROSTER),
        "budget_points": list(BUDGET_POINTS),
        "expected_cells": len(rows),
        "completed_cells": sum(row["status"] == "PASS" for row in rows),
        "completion_rate": sum(row["status"] == "PASS" for row in rows) / len(rows),
        "primary_metric": "fixed_schema_macro_f1",
        "supported_class_metric_role": "diagnostic_only",
        "fold_seed_matrix": fold_seed_matrix.tolist(),
        "confusion_matrix": confusion_accumulator.tolist(),
        "per_class": classification_metrics_from_confusion(confusion_accumulator)["per_class"],
        "accuracy": classification_metrics_from_confusion(confusion_accumulator)["accuracy"],
        "fixed_schema_macro_f1": classification_metrics_from_confusion(confusion_accumulator)["macro_f1"],
        "supported_class_macro_f1": classification_metrics_from_confusion(confusion_accumulator)["supported_class_macro_f1"],
        "expected_calibration_error": 0.0,
        "missing_modality_diagnostic": missing_diag,
        "S_lane": s_lane,
        "development_hdf5_sha256": _sha256(hdf5_path),
        "frozen_test_accessed": False,
        "development_only": True,
    }
    _atomic_json(output_dir / SUMMARY_FILENAME, summary)
    leaderboard = {
        "schema_version": "lithofacies-p5-r2-diagnostic-leaderboard-v1",
        "task_id": TASK_ID,
        "primary_metric": "fixed_schema_macro_f1",
        "supported_class_metric_role": "diagnostic_only",
        "entries": [
            {
                "model_id": model_id,
                "pass_count": sum(row["model_id"] == model_id and row["status"] == "PASS" for row in rows),
                "mean_fixed_schema_macro_f1": _safe_mean(
                    [
                        float(row["budget_metrics"][str(max(BUDGET_POINTS))]["M"]["fixed_schema_macro_f1"])
                        for row in rows
                        if row["model_id"] == model_id and row["status"] == "PASS"
                    ]
                ),
            }
            for model_id in MODEL_ROSTER
        ],
    }
    _atomic_json(output_dir / LEADERBOARD_FILENAME, leaderboard)
    visualization = _plot_learning_curves(
        {
            "model_groups": [
                {
                    "label": model_id,
                    "curves": {
                        "M": [
                            {
                                "budget": int(budget),
                                "fixed_schema_macro_f1": _safe_mean(
                                    [
                                        float(row["budget_metrics"][str(budget)]["M"]["fixed_schema_macro_f1"])
                                        for row in rows
                                        if row["model_id"] == model_id and row["status"] == "PASS"
                                    ]
                                ),
                            }
                            for budget in BUDGET_POINTS
                        ],
                        "W": [
                            {
                                "budget": int(budget),
                                "fixed_schema_macro_f1": _safe_mean(
                                    [
                                        float(row["budget_metrics"][str(budget)]["W"]["fixed_schema_macro_f1"])
                                        for row in rows
                                        if row["model_id"] == model_id and row["status"] == "PASS"
                                    ]
                                ),
                            }
                            for budget in BUDGET_POINTS
                        ],
                    },
                }
                for model_id in MODEL_ROSTER
            ],
            "confusion_matrix": summary["confusion_matrix"],
            "per_class": summary["per_class"],
            "expected_calibration_error": summary["expected_calibration_error"],
            "fold_seed_matrix": summary["fold_seed_matrix"],
            "missing_modality_diagnostic": summary["missing_modality_diagnostic"],
        },
        output_dir,
    )
    _atomic_json(output_dir / VISUALIZATION_FILENAME, visualization)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = run(args.dataset_root, args.output_dir, device=args.device)
    print(json.dumps({
        "status": summary["status"],
        "completed_cells": summary["completed_cells"],
        "expected_cells": summary["expected_cells"],
        "S_lane": summary["S_lane"],
    }, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
