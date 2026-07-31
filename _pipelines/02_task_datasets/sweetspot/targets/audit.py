"""Generate auditable real-data contracts for sweetspot targets one through five."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from _code.ml_framework.artifacts import atomic_write_json


FEASIBLE = (
    "reservoir_quality", "hydrocarbon_pay", "productivity", "water_breakthrough",
)
BASE = "_pipelines.02_task_datasets.sweetspot.targets"


def _hash_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8")); digest.update(b"\0")
    return digest.hexdigest()


def _target_values(dataset: Mapping[str, Any]) -> np.ndarray:
    for name in ("target", "target_future_30d_mean_oil_sm3_day", "event_within_30d"):
        if name in dataset:
            return np.asarray(dataset[name], dtype=np.float64)
    raise KeyError("dataset has no recognized target array")


def _dataset_summary(dataset: Mapping[str, Any]) -> dict[str, Any]:
    target = _target_values(dataset)
    normalized = np.nan_to_num(target, nan=np.inf, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    groups = [str(value) for value in dataset["groups"]]
    return {
        "sample_count": len(dataset["sample_ids"]),
        "group_counts": {group: groups.count(group) for group in sorted(set(groups))},
        "sample_ids_sha256": _hash_strings(list(dataset["sample_ids"])),
        "target_bytes_sha256": hashlib.sha256(normalized.tobytes()).hexdigest(),
        "finite_target_count": int(np.isfinite(target).sum()),
        "positive_target_count": int((target > 0).sum()),
    }


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    registry: dict[str, Any] = {"root_seed": 2693, "targets": {}}
    for name in FEASIBLE:
        module = importlib.import_module(f"{BASE}.{name}.contract")
        dataset, manifest, evidence = module.build_dataset_and_manifest()
        target_dir = output_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target_dir / "task_spec.json", module.task_spec().to_dict())
        atomic_write_json(target_dir / "split_manifest.json", manifest.to_dict())
        atomic_write_json(target_dir / "data_evidence.json", evidence)
        summary = _dataset_summary(dataset)
        atomic_write_json(target_dir / "dataset_summary.json", summary)
        registry["targets"][name] = {
            "task_id": module.TASK_ID,
            "status": module.STATUS,
            "split_hash": manifest.stable_hash(),
            "effective_n_splits": manifest.effective_n_splits,
            **summary,
        }
    remaining = importlib.import_module(f"{BASE}.remaining_oil_infill.contract")
    blocked = remaining.not_feasible_evidence()
    blocked_dir = output_dir / "remaining_oil_infill"
    blocked_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(blocked_dir / "not_feasible.json", blocked)
    registry["targets"]["remaining_oil_infill"] = {
        "task_id": remaining.TASK_ID,
        "status": remaining.STATUS,
        "blockers": blocked["blockers"],
    }
    atomic_write_json(output_dir / "registry_targets_1_to_5.json", registry)
    return registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
