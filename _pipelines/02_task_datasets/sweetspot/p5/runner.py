"""P5 Stage-1 contract runner for the first ten sweetspot model families.

The runner cannot construct labels and exposes no test loader/path argument.
It first applies the static model-target matrix, then the target-specific label
approval gate, then the runtime/source gate.  Only a content-addressed
``split=development, contains_test=false`` manifest can reach an adapter.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from _code.ml_framework.model_discovery import discover_model
from _models.sweetspot.p5_common import AdapterSkip

from .development_data import DevelopmentBatch, load_development_batch
from .label_gate import DEFAULT_INVENTORY, LabelGateResult, evaluate_label_spec
from .matrix import MODEL_ORDER, TARGET_ORDER, matrix_gate, matrix_payload
from .source_lock import inspect_runtime, load_source_lock
from .target_specs import build_task_spec


ROOT_SEED = 2693


def _parse_assignments(values: Sequence[str], *, allowed: Sequence[str], flag: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{flag} entries must use TARGET=PATH")
        target, raw_path = value.split("=", 1)
        if target not in allowed:
            raise ValueError(f"{flag}: unknown target {target!r}")
        if target in result:
            raise ValueError(f"{flag}: duplicate assignment for {target}")
        result[target] = Path(raw_path)
    return result


def _synthetic_from(batch: DevelopmentBatch, task_type: str, class_count: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    import pandas as pd

    synthetic_inputs: dict[str, Any] = {}
    sample_count = min(8, len(batch.sample_ids))
    rng = np.random.default_rng(ROOT_SEED)
    for key, value in batch.inputs.items():
        if isinstance(value, pd.DataFrame):
            frame = value.copy()
            frame["target"] = np.linspace(0.0, 1.0, len(frame))
            synthetic_inputs[key] = frame
            sample_count = len(frame)
        else:
            array = np.asarray(value)[:sample_count]
            if key == "edge_index":
                synthetic_inputs[key] = array
            else:
                synthetic_inputs[key] = rng.normal(size=array.shape).astype(np.float32)
    if task_type == "binary":
        target = np.arange(sample_count, dtype=np.float32) % 2
    elif task_type == "multiclass":
        target = np.arange(sample_count, dtype=np.int64) % max(2, class_count)
    else:
        target = rng.normal(size=sample_count).astype(np.float32)
    first = next((np.asarray(value) for value in synthetic_inputs.values() if not isinstance(value, pd.DataFrame)), None)
    if first is not None and first.ndim >= 4:
        target = rng.normal(size=(sample_count, *first.shape[2:])).astype(np.float32)
    return synthetic_inputs, target, np.ones_like(target, dtype=bool)


def _skip_result(model_id: str, target_id: str, rating: str | None, reason: str, detail: Any) -> dict[str, Any]:
    return {
        "model_id": model_id, "target_id": target_id, "rating": rating,
        "status": "SKIP", "reason_code": reason, "detail": detail,
        "evidence_state": "scouted", "label_generated": False,
        "test_accessed": False, "scientific_metrics": None,
    }


def _run_pair(
    model_id: str,
    target_id: str,
    *,
    gate: LabelGateResult,
    development_manifest: Path | None,
    source_entry: Mapping[str, Any],
    batch_limit: int,
) -> dict[str, Any]:
    cell = matrix_gate(model_id, target_id)
    if not cell.eligible:
        return _skip_result(model_id, target_id, None, "matrix_not_applicable", cell.to_dict())
    if not gate.approved:
        reason = gate.reason_codes[0] if gate.reason_codes else "label_spec_invalid"
        return _skip_result(model_id, target_id, cell.rating, reason, gate.to_dict())
    runtime = inspect_runtime(source_entry)
    if not runtime["available"]:
        return _skip_result(
            model_id, target_id, cell.rating, str(runtime["reason_code"]), runtime,
        )
    if not runtime["version_allowed"]:
        return _skip_result(model_id, target_id, cell.rating, "runtime_version_not_locked", runtime)
    if development_manifest is None:
        return _skip_result(
            model_id, target_id, cell.rating, "development_batch_missing",
            "an approved, content-addressed development manifest is required",
        )
    try:
        task_spec = build_task_spec(target_id, gate)
        batch = load_development_batch(
            development_manifest, target_id=target_id,
            label_spec_sha256=str(gate.spec_sha256), limit=batch_limit,
        )
        discovered = discover_model("sweetspot", model_id)
        class_count = int(task_spec.metadata.get("class_count", 0))
        synthetic_inputs, synthetic_target, synthetic_mask = _synthetic_from(
            batch, task_spec.task_type, class_count,
        )
        synthetic_adapter = discovered.build(task_spec)
        synthetic = synthetic_adapter.stage1_smoke(
            synthetic_inputs, synthetic_target, synthetic_mask, seed=ROOT_SEED,
        )
        replay_adapter = discovered.build(task_spec)
        replay = replay_adapter.stage1_smoke(
            synthetic_inputs, synthetic_target, synthetic_mask, seed=ROOT_SEED,
        )
        deterministic = synthetic.get("output_sha256") == replay.get("output_sha256")
        if not deterministic:
            raise RuntimeError("same-seed synthetic replay changed the archived output hash")
        synthetic["same_seed_replay"] = True
        real_adapter = discovered.build(task_spec)
        real = real_adapter.stage1_smoke(
            batch.inputs, batch.target, batch.target_mask, seed=ROOT_SEED,
        )
    except AdapterSkip as exc:
        return _skip_result(model_id, target_id, cell.rating, exc.reason_code, exc.detail)
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        return _skip_result(
            model_id, target_id, cell.rating, "contract_or_data_gate_failed",
            f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:
        return {
            "model_id": model_id, "target_id": target_id, "rating": cell.rating,
            "status": "FAILED", "reason_code": "adapter_runtime_failure",
            "detail": f"{type(exc).__name__}: {exc}", "evidence_state": "scouted",
            "label_generated": False, "test_accessed": False, "scientific_metrics": None,
        }
    return {
        "model_id": model_id, "target_id": target_id, "rating": cell.rating,
        "status": "PASS", "reason_code": None, "evidence_state": "contract_smoked",
        "task_spec": task_spec.to_dict(),
        "source_lock": {"revision": source_entry["revision"], "license": source_entry["license"], "runtime": runtime},
        "development_manifest": {"path": batch.manifest_path, "sha256": batch.manifest_sha256, "sample_count": len(batch.sample_ids)},
        "synthetic_smoke": synthetic, "real_development_smoke": real,
        "label_generated": False, "test_accessed": False, "scientific_metrics": None,
    }


def run_stage1(
    *,
    model_ids: Sequence[str] = MODEL_ORDER,
    target_ids: Sequence[str] = TARGET_ORDER,
    label_specs: Mapping[str, Path] | None = None,
    development_manifests: Mapping[str, Path] | None = None,
    inventory_path: Path = DEFAULT_INVENTORY,
    batch_limit: int = 64,
) -> dict[str, Any]:
    labels = dict(label_specs or {})
    manifests = dict(development_manifests or {})
    source_lock = load_source_lock()
    gates = {
        target: evaluate_label_spec(target, labels.get(target), inventory_path=inventory_path)
        for target in target_ids
    }
    approved_hashes: dict[str, list[str]] = {}
    for target, gate in gates.items():
        if gate.approved and gate.spec_sha256:
            approved_hashes.setdefault(gate.spec_sha256, []).append(target)
    for digest, targets in approved_hashes.items():
        if len(targets) > 1:
            for target in targets:
                current = gates[target]
                gates[target] = LabelGateResult(
                    target, False, "SKIP", ("shared_label_spec_forbidden",),
                    (f"one label_spec hash cannot define multiple independent targets: {targets}",),
                    current.spec_path, digest, current.spec,
                )
    results = [
        _run_pair(
            model, target, gate=gates[target], development_manifest=manifests.get(target),
            source_entry=source_lock[model], batch_limit=batch_limit,
        )
        for model in model_ids for target in target_ids
    ]
    counts = {status: sum(item["status"] == status for item in results) for status in ("PASS", "SKIP", "FAILED")}
    source_summary = {
        model: {
            "revision": source_lock[model]["revision"],
            "license": source_lock[model]["license"],
            "environment_group": source_lock[model]["environment_group"],
            "implementation_mode": source_lock[model]["implementation_mode"],
            "runtime": inspect_runtime(source_lock[model]),
        }
        for model in model_ids
    }
    return {
        "schema_version": "sweetspot-p5-stage1/v1",
        "stage": "contract_smoke",
        "root_seed": ROOT_SEED,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "model_order": list(model_ids), "target_order": list(target_ids),
        "source_lock": source_summary,
        "matrix": matrix_payload(), "label_gates": {key: value.to_dict() for key, value in gates.items()},
        "results": results, "counts": counts,
        "test_loader_api_present": False, "test_accessed": False,
        "labels_generated": False, "scientific_metrics_reported": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=MODEL_ORDER, dest="models")
    parser.add_argument("--target", action="append", choices=TARGET_ORDER, dest="targets")
    parser.add_argument("--label-spec", action="append", default=[], metavar="TARGET=PATH")
    parser.add_argument("--development-manifest", action="append", default=[], metavar="TARGET=PATH")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--batch-limit", type=int, default=64)
    parser.add_argument("--output", type=Path, help="optional JSON path; stdout-only when omitted")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    models = tuple(args.models or MODEL_ORDER); targets = tuple(args.targets or TARGET_ORDER)
    if args.batch_limit <= 0:
        raise ValueError("batch-limit must be positive")
    labels = _parse_assignments(args.label_spec, allowed=TARGET_ORDER, flag="--label-spec")
    manifests = _parse_assignments(args.development_manifest, allowed=TARGET_ORDER, flag="--development-manifest")
    report = run_stage1(
        model_ids=models, target_ids=targets, label_specs=labels,
        development_manifests=manifests, inventory_path=args.inventory, batch_limit=args.batch_limit,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 1 if report["counts"]["FAILED"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
