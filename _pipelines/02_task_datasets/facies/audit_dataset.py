#!/usr/bin/env python3
"""Fail-loud audit of real processed facies datasets and leakage contracts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from _code.dataset_io import load_dataset  # noqa: E402
from _code.ml_framework.preprocess import NormStats, denormalize  # noqa: E402
from pipeline_contract import (  # noqa: E402
    DEFAULT_VALIDATION_FRACTION,
    DEFAULT_VALIDATION_GUARD_FRACTION,
    PIPELINE_VERSION,
    TASK_SCHEMAS,
    is_near_constant_patch,
    ordered_spatial_split,
    validate_label_array,
)

DEFAULT_MANIFEST = TRACK_DIR / "_outputs" / PIPELINE_VERSION / "data_build_manifest.json"
DEFAULT_OUTPUT = TRACK_DIR / "_outputs" / PIPELINE_VERSION / "dataset_audit.json"


def line_range(lines: set[int]) -> list[int]:
    return [min(lines), max(lines)]


def audit_split(task: str, split: str) -> tuple[dict[str, Any], set[int], dict[str, Any]]:
    schema = TASK_SCHEMAS[task]
    lines: set[int] = set()
    coordinates: set[tuple[int, int, int]] = set()
    histogram = np.zeros(schema.num_classes, dtype=np.int64)
    reference_stats: dict[str, Any] | None = None
    max_roundtrip_error = 0.0
    count = 0
    for sample in load_dataset(task, split):
        count += 1
        line = int(sample["position"]["inline"])
        row, col = (int(value) for value in sample["meta"]["patch_origin"])
        coordinate = (line, row, col)
        if coordinate in coordinates:
            raise ValueError(f"duplicate patch coordinate in {task}/{split}: {coordinate}")
        coordinates.add(coordinate)
        lines.add(line)

        label = np.asarray(sample["label"])
        validate_label_array(label, schema)
        histogram += np.bincount(
            label.reshape(-1), minlength=schema.num_classes
        )[: schema.num_classes]

        meta = sample["meta"]
        if meta.get("pipeline_version") != PIPELINE_VERSION:
            raise ValueError(f"{task}/{split} contains a pre-{PIPELINE_VERSION} sample")
        if meta.get("normalization_fit_scope") != "model_train_inline_only":
            raise ValueError(f"{task}/{split} normalization fit scope is not model-train only")
        stats_dict = meta.get("normalization_stats")
        if reference_stats is None:
            reference_stats = stats_dict
        elif stats_dict != reference_stats:
            raise ValueError(f"{task}/{split} normalization stats vary by sample")
        stats = NormStats.from_dict(stats_dict)
        raw = np.asarray(denormalize(sample["seismic_patch"], stats), dtype=np.float32)
        if is_near_constant_patch(raw):
            raise ValueError(f"{task}/{split} retained near-constant patch {coordinate}")
        max_roundtrip_error = max(
            max_roundtrip_error,
            float(meta["normalization_roundtrip_max_abs_error"]),
        )

    if count == 0 or not lines or reference_stats is None:
        raise ValueError(f"{task}/{split} is empty")
    if np.any(histogram <= 0):
        raise ValueError(f"{task}/{split} lacks configured classes: {histogram.tolist()}")
    return (
        {
            "samples": count,
            "unique_coordinates": len(coordinates),
            "inline_count": len(lines),
            "inline_range": line_range(lines),
            "label_histogram": histogram.tolist(),
            "normalization_stats": reference_stats,
            "normalization_roundtrip_max_abs_error": max_roundtrip_error,
            "near_constant_retained": 0,
        },
        lines,
        reference_stats,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("pipeline_version") != PIPELINE_VERSION:
        raise ValueError("build manifest pipeline version mismatch")
    manifest_tasks = {record["task"]: record for record in manifest["tasks"]}

    report: dict[str, Any] = {"pipeline_version": PIPELINE_VERSION, "tasks": {}}
    for task in TASK_SCHEMAS:
        train_record, train_lines, train_stats = audit_split(task, "train")
        test_record, test_lines, test_stats = audit_split(task, "test")
        if train_stats != test_stats:
            raise ValueError(f"{task} test uses different normalization statistics")
        if train_lines & test_lines:
            raise ValueError(f"{task} train/test inline leakage detected")

        task_manifest = manifest_tasks[task]
        if train_record["inline_range"] != task_manifest["train_line_range"]:
            raise ValueError(f"{task} train inline range differs from build manifest")
        if test_record["inline_range"] != task_manifest["test_line_range"]:
            raise ValueError(f"{task} test inline range differs from build manifest")

        model_train, validation_guard, validation = ordered_spatial_split(
            train_lines,
            DEFAULT_VALIDATION_FRACTION,
            DEFAULT_VALIDATION_GUARD_FRACTION,
        )
        expected_fit_range = [min(model_train), max(model_train)]
        if expected_fit_range != task_manifest["model_train_line_range"]:
            raise ValueError(f"{task} normalization fit range differs from internal split")
        report["tasks"][task] = {
            "train": train_record,
            "test": test_record,
            "external_train_test_overlap": [],
            "external_guard_inline_range": task_manifest["guard_line_range"],
            "internal_model_train_inline_range": line_range(model_train),
            "internal_validation_guard_inline_range": line_range(validation_guard),
            "internal_validation_inline_range": line_range(validation),
            "internal_overlap": [],
            "normalization_fit_inline_range": expected_fit_range,
            "near_constant_filtered": task_manifest["near_constant_filter"],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
