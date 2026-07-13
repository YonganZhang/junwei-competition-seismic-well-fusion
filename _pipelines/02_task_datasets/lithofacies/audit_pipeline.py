#!/usr/bin/env python3
"""Anti-fake completion audit over real lithofacies data and run artifacts."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from _code.dataset_io import load_dataset  # noqa: E402
from pipeline_contract import (  # noqa: E402
    CLASS_NAMES,
    PIPELINE_VERSION,
    assert_family_isolation,
)


def project_relative(path: Path) -> str:
    """Serialize project-owned evidence paths without a host/worktree prefix."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"缺失或空JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_finite_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"非有限数值: {path}={value}")


def run_audit() -> dict[str, Any]:
    manifest_path = TRACK_DIR / "_outputs/split_manifest.json"
    metrics_path = TRACK_DIR / "_outputs/multimodal_mlp/metrics.json"
    history_path = TRACK_DIR / "_outputs/multimodal_mlp/checkpoints/history.json"
    manifest = _load_json(manifest_path)
    run = _load_json(metrics_path)
    history = _load_json(history_path)
    saved_train = list(load_dataset("lithofacies", "train"))
    saved_test = list(load_dataset("lithofacies", "test"))
    checks: list[dict[str, Any]] = []

    def check(name: str, assertion: Callable[[], Any]) -> None:
        try:
            evidence = assertion()
        except Exception as exc:
            checks.append({"name": name, "status": "FAIL", "evidence": f"{type(exc).__name__}: {exc}"})
            return
        checks.append({"name": name, "status": "PASS", "evidence": evidence})

    def label_contract() -> dict[str, Any]:
        contract = manifest["label_contract"]
        observed = (contract["workbooks"], contract["wells"], contract["intervals"])
        if observed != (11, 11, 139):
            raise ValueError(f"标签计数变化: {observed}")
        if tuple(contract["class_names"]) != CLASS_NAMES:
            raise ValueError("固定9类schema变化")
        if contract["source"] != "GM09" or contract["curve_type"] != "GENETIC FACIES":
            raise ValueError("标签来源变化")
        return {"workbooks": 11, "wells": 11, "intervals": 139, "classes": 9}

    def sample_contract() -> dict[str, Any]:
        if len(saved_train) != 447 or len(saved_test) != 120:
            raise ValueError(f"保存样本计数变化: train={len(saved_train)}, test={len(saved_test)}")
        roles = {"train": 0, "guard": 0, "test": 0}
        records = []
        for sample in saved_train + saved_test:
            role = sample["meta"]["partition"]
            roles[role] += 1
            records.append({"partition": role, "family_id": sample["meta"]["family_id"]})
            trace = sample["meta"]["label_trace"]
            if trace["source"] != "GM09" or trace["curve_type"] != "GENETIC FACIES":
                raise ValueError("保存样本出现非法标签来源")
            if trace["class_name"] in ("UNKNOWN", "UNDEFINED"):
                raise ValueError("UNKNOWN/UNDEFINED混入保存样本")
            if not np.isfinite(sample["well_log_seq"]).all():
                raise ValueError("well_log_seq含NaN/Inf")
            if not np.isfinite(sample["seismic_patch"]).all():
                raise ValueError("seismic_patch含NaN/Inf")
        if roles != {"train": 360, "guard": 87, "test": 120}:
            raise ValueError(f"partition计数变化: {roles}")
        return {"roles": roles, "families": assert_family_isolation(records)}

    def normalization_contract() -> dict[str, Any]:
        normalization = manifest["normalization"]
        if normalization["fit_scope"] != "train_mother_well_families_only":
            raise ValueError("归一化fit scope非法")
        error = float(normalization["max_round_trip_error"])
        if not math.isfinite(error) or error > 1e-10:
            raise ValueError(f"归一化round-trip误差过大: {error}")
        return {"fit_scope": normalization["fit_scope"], "max_round_trip_error": error}

    def training_contract() -> dict[str, Any]:
        if run["pipeline_version"] != PIPELINE_VERSION:
            raise ValueError("训练pipeline版本变化")
        if run["model_name"] != "multimodal_mlp" or run["architecture"] != "MultimodalMLP":
            raise ValueError("模型不是冻结的小型动态注册baseline")
        train_loss = history["train_loss"]
        val_loss = history["val_loss"]
        if len(train_loss) != 80 or len(val_loss) != 80:
            raise ValueError("没有完整80轮train/guard曲线")
        expected_best = int(np.argmin(val_loss))
        if history["best_epoch"] != expected_best or run["best_epoch"] != expected_best + 1:
            raise ValueError("best checkpoint不是最小guard loss轮次")
        _assert_finite_tree(run["metrics"], "metrics")
        return {
            "epochs": 80,
            "best_epoch": run["best_epoch"],
            "best_guard_loss": run["best_guard_loss"],
            "metrics_finite": True,
        }

    def artifact_contract() -> dict[str, int]:
        paths = {
            "best_checkpoint": TRACK_DIR / "_outputs/multimodal_mlp/checkpoints/best.ckpt",
            "loss_curve": TRACK_DIR / "_outputs/multimodal_mlp/loss_curve.png",
            "confusion_matrix": TRACK_DIR / "_outputs/multimodal_mlp/confusion_matrix.png",
            "best_checkpoint_predictions": TRACK_DIR / "_outputs/multimodal_mlp/best_checkpoint_predictions.png",
        }
        sizes = {name: path.stat().st_size for name, path in paths.items() if path.exists()}
        if set(sizes) != set(paths) or any(size <= 0 for size in sizes.values()):
            raise ValueError(f"运行产物缺失/为空: {sizes}")
        return sizes

    check("explicit_gm09_label_contract", label_contract)
    check("real_multimodal_samples_and_family_isolation", sample_contract)
    check("train_family_only_reversible_normalization", normalization_contract)
    check("shared_loop_best_checkpoint_and_finite_metrics", training_contract)
    check("nonempty_checkpoint_and_visual_artifacts", artifact_contract)
    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    return {
        "audit": "lithofacies_completion_v1",
        "pipeline_version": PIPELINE_VERSION,
        "status": status,
        "checks": checks,
        "source_files": {
            "split_manifest": project_relative(manifest_path),
            "metrics": project_relative(metrics_path),
            "history": project_relative(history_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=TRACK_DIR / "_outputs/completion_audit.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
