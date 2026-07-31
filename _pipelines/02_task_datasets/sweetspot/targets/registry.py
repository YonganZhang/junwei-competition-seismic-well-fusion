"""Build the portable registry for all seven independent sweetspot targets."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Mapping

from _code.ml_framework.artifacts import atomic_write_json, hash_file


HERE = Path(__file__).resolve().parent
BASE = "_pipelines.02_task_datasets.sweetspot.targets"


def _relative(path: Path) -> str:
    return path.relative_to(HERE).as_posix()


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path, *, required: bool) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return None
    return {"path": _relative(path), "sha256": hash_file(path), "bytes": path.stat().st_size}


def _trained_case(number: int, target_id: str) -> dict[str, Any]:
    root = HERE / target_id / "_outputs" / "baseline_v1"
    status_path = root / "status.json"
    if not status_path.is_file():
        return {
            "target_number": number,
            "target_id": target_id,
            "status": "implementation_ready_not_run",
            "independent_case": True,
            "run_root": _relative(root),
        }
    status = _read_json(status_path)
    return {
        "target_number": number,
        "target_id": target_id,
        "task_id": status["task_id"],
        "label_version": status["label_version"],
        "status": status["status"],
        "independent_case": True,
        "run_root": _relative(root),
        "model_id": status["model_id"],
        "requested_folds": status["requested_folds"],
        "effective_folds": status["effective_folds"],
        "development_samples": status["development_samples"],
        "test_samples": status["test_samples"],
        "test_consumed_once": status["test_consumed_once"],
        "artifacts": {
            "task_spec": _artifact(root / "task_spec.json", required=True),
            "split_manifest": _artifact(root / "split_manifest.json", required=True),
            "config": _artifact(root / "config.json", required=True),
            "checkpoint": _artifact(root / "refit" / "checkpoint_best.pkl", required=True),
            "metrics": _artifact(root / "frozen_test" / "metrics.json", required=True),
            "predictions": _artifact(root / "frozen_test" / "predictions.csv", required=True),
            "visualization_manifest": _artifact(root / "visualizations" / "visualization_manifest.json", required=True),
            "artifact_manifest": _artifact(root / "manifest.json", required=True),
        },
    }


def _remaining_oil_case() -> dict[str, Any]:
    contract = importlib.import_module(f"{BASE}.remaining_oil_infill.contract")
    root = HERE / "remaining_oil_infill" / "_outputs"
    evidence_path = root / "not_feasible.json"
    evidence = contract.not_feasible_evidence()
    atomic_write_json(evidence_path, evidence)
    return {
        "target_number": 5,
        "target_id": "remaining_oil_infill",
        "task_id": contract.TASK_ID,
        "status": "not_feasible",
        "independent_case": True,
        "reason": "dynamic state/cutoff/candidate/economic labels are not frozen",
        "artifacts": {"not_feasible": _artifact(evidence_path, required=True)},
    }


def _property_case(number: int, target_id: str, case_id: str, *, extra_case: str | None = None) -> dict[str, Any]:
    root = HERE / target_id / "_outputs" / case_id
    status = _read_json(root / "status.json")
    result = {
        "target_number": number,
        "target_id": target_id,
        "task_id": _read_json(root / "task_spec.json")["task_id"],
        "label_version": status["label_version"],
        "status": status["status"],
        "independent_case": True,
        "run_root": _relative(root),
        "model_id": status["model"],
        "requested_folds": status["requested_folds"],
        "effective_folds": status["effective_folds"],
        "development_samples": status["development_samples"],
        "test_samples": status["test_samples"],
        "test_consumed_once": status["test_consumed_once"],
        "artifacts": {
            "task_spec": _artifact(root / "task_spec.json", required=True),
            "split_manifest": _artifact(root / "split_manifest.json", required=True),
            "config": _artifact(root / "config.json", required=True),
            "checkpoint": _artifact(root / "refit" / "checkpoint_best.pkl", required=True),
            "metrics": _artifact(root / "frozen_test" / "metrics.json", required=True),
            "predictions": _artifact(root / "frozen_test" / "predictions.csv", required=True),
            "artifact_manifest": _artifact(root / "manifest.json", required=True),
        },
    }
    if extra_case is not None:
        extra_root = HERE / target_id / "_outputs" / extra_case
        extra_status = _read_json(extra_root / "status.json")
        result["additional_independent_case"] = {
            "case_id": extra_case,
            "status": extra_status["status"],
            "reason": extra_status["reason"],
            "task_spec": _artifact(extra_root / "task_spec.json", required=True),
            "label_availability": _artifact(extra_root / "label_availability.json", required=True),
            "status_artifact": _artifact(extra_root / "status.json", required=True),
        }
    return result


def build_registry(output_path: Path | None = None) -> dict[str, Any]:
    targets = [
        _trained_case(1, "reservoir_quality"),
        _trained_case(2, "hydrocarbon_pay"),
        _trained_case(3, "productivity"),
        _trained_case(4, "water_breakthrough"),
        _remaining_oil_case(),
        _property_case(6, "porosity", "phif", extra_case="phie"),
        _property_case(7, "permeability", "klogh"),
    ]
    if [item["target_number"] for item in targets] != list(range(1, 8)):
        raise RuntimeError("seven-target registry is not ordered 1..7")
    payload = {
        "registry_version": "p4-sweetspot-seven-targets-v1",
        "root_seed": 2693,
        "target_count": 7,
        "all_targets_independent": all(item["independent_case"] for item in targets),
        "targets": targets,
    }
    destination = output_path or HERE / "_outputs" / "registry_targets_1_to_7.json"
    atomic_write_json(destination, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_registry(args.output), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
