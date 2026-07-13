"""Buffered spatial CV manifests for dense seismic facies segmentation."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from _code.ml_framework.splits import Fold, SplitManifest, validate_manifest

from p4_data import ArchiveRecord, label_histogram
from p4_tasks import EXPECTED_OUTER_SPLITS, INTERNAL_BUFFER_GROUPS, LABEL_VERSIONS, TASK_IDS
from pipeline_contract import get_task_schema


def _partition_with_permanent_buffers(
    groups: Sequence[int], n_splits: int, buffer_groups: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Return contiguous core blocks separated by globally excluded buffers."""
    if n_splits < 2 or buffer_groups < 1:
        raise ValueError("spatial CV requires >=2 folds and a positive buffer")
    core_total = len(groups) - (n_splits - 1) * buffer_groups
    if core_total < n_splits:
        raise ValueError(
            f"{len(groups)} groups cannot support {n_splits} core blocks plus "
            f"{n_splits - 1} buffers of width {buffer_groups}"
        )
    base, remainder = divmod(core_total, n_splits)
    core_sizes = [base + (1 if index < remainder else 0) for index in range(n_splits)]
    cores: list[list[int]] = []
    buffers: list[list[int]] = []
    cursor = 0
    for index, core_size in enumerate(core_sizes):
        cores.append(list(groups[cursor : cursor + core_size]))
        cursor += core_size
        if index < n_splits - 1:
            buffers.append(list(groups[cursor : cursor + buffer_groups]))
            cursor += buffer_groups
    if cursor != len(groups) or any(not core for core in cores):
        raise AssertionError("internal spatial partition did not consume every input group")
    return cores, buffers


def _records_for_groups(
    records: Sequence[ArchiveRecord], groups: set[int]
) -> tuple[ArchiveRecord, ...]:
    return tuple(record for record in records if record.inline in groups)


def _all_classes_supported(records: Sequence[ArchiveRecord], num_classes: int) -> bool:
    return bool(np.all(label_histogram(records, num_classes) > 0))


def build_facies_spatial_manifest(
    task_id: str,
    development_records: Sequence[ArchiveRecord],
    frozen_test_records: Sequence[ArchiveRecord],
    *,
    requested_n_splits: int = 5,
    buffer_groups: int | None = None,
) -> SplitManifest:
    """Freeze test first, then form buffered contiguous development folds.

    The shared manifest requires every declared development sample to receive
    exactly one OOF prediction.  Therefore the inter-block buffers are
    permanently excluded from declared development, listed explicitly in
    metadata, and never silently assigned to training.
    """
    if task_id not in TASK_IDS:
        raise ValueError(f"unknown facies task {task_id!r}")
    if requested_n_splits < 2:
        raise ValueError("requested_n_splits must be >=2")
    schema = get_task_schema(task_id)
    dev = tuple(development_records)
    test = tuple(frozen_test_records)
    if not dev or not test:
        raise ValueError("both development and frozen-test metadata must be present")
    if any(record.task_id != task_id or record.split != "train" for record in dev):
        raise ValueError("development records have wrong task/split")
    if any(record.task_id != task_id or record.split != "test" for record in test):
        raise ValueError("frozen-test records have wrong task/split")
    if any(record.label_support is None for record in dev):
        raise ValueError("development support must be indexed from labels")
    if any(record.label_support is not None for record in test):
        raise ValueError("test labels were consumed before the frozen-test lifecycle stage")

    dev_lines = sorted({record.inline for record in dev})
    test_lines = sorted({record.inline for record in test})
    if set(dev_lines) & set(test_lines):
        raise ValueError("development and frozen-test inline groups overlap")
    expected = EXPECTED_OUTER_SPLITS[task_id]
    if (dev_lines[0], dev_lines[-1]) != tuple(expected["development_inline_range"]):
        raise ValueError(
            f"{task_id} development range {(dev_lines[0], dev_lines[-1])} differs from frozen contract "
            f"{expected['development_inline_range']}"
        )
    if (test_lines[0], test_lines[-1]) != tuple(expected["test_inline_range"]):
        raise ValueError(
            f"{task_id} test range {(test_lines[0], test_lines[-1])} differs from frozen contract "
            f"{expected['test_inline_range']}"
        )
    observed_external_gap = test_lines[0] - dev_lines[-1] - 1
    expected_external_gap = (
        expected["external_guard_inline_range"][1]
        - expected["external_guard_inline_range"][0]
        + 1
    )
    if observed_external_gap < expected_external_gap:
        raise ValueError(
            f"{task_id} external test guard is only {observed_external_gap}, expected >= {expected_external_gap}"
        )

    width = INTERNAL_BUFFER_GROUPS[task_id] if buffer_groups is None else int(buffer_groups)
    attempts: list[str] = []
    chosen: tuple[int, list[list[int]], list[list[int]]] | None = None
    for candidate in range(min(requested_n_splits, len(dev_lines)), 1, -1):
        try:
            cores, buffers = _partition_with_permanent_buffers(dev_lines, candidate, width)
        except ValueError as exc:
            attempts.append(f"{candidate}-fold: {exc}")
            continue
        supported = True
        for fold_id, validation_core in enumerate(cores):
            validation_set = set(validation_core)
            train_set = set(value for index, core in enumerate(cores) if index != fold_id for value in core)
            validation_records = _records_for_groups(dev, validation_set)
            train_records = _records_for_groups(dev, train_set)
            if not _all_classes_supported(validation_records, schema.num_classes):
                attempts.append(f"{candidate}-fold: validation fold {fold_id} lacks configured classes")
                supported = False
                break
            if not _all_classes_supported(train_records, schema.num_classes):
                attempts.append(f"{candidate}-fold: train fold {fold_id} lacks configured classes")
                supported = False
                break
        if supported:
            chosen = candidate, cores, buffers
            break
    if chosen is None:
        raise ValueError("support/buffer constraints leave fewer than two folds: " + "; ".join(attempts))

    effective, cores, buffers = chosen
    core_groups = set(value for core in cores for value in core)
    buffer_group_set = set(value for band in buffers for value in band)
    core_records = _records_for_groups(dev, core_groups)
    buffer_records = _records_for_groups(dev, buffer_group_set)
    folds: list[Fold] = []
    for fold_id, validation_core in enumerate(cores):
        validation_groups = set(validation_core)
        train_groups = core_groups - validation_groups
        validation_records = _records_for_groups(dev, validation_groups)
        train_records = _records_for_groups(dev, train_groups)
        minimum_distance = min(abs(train - validation) for train in train_groups for validation in validation_groups)
        if minimum_distance <= width:
            raise AssertionError(
                f"fold {fold_id} nearest train/validation distance {minimum_distance} does not exceed buffer {width}"
            )
        adjacent_buffers: list[int] = []
        if fold_id > 0:
            adjacent_buffers.extend(buffers[fold_id - 1])
        if fold_id < len(buffers):
            adjacent_buffers.extend(buffers[fold_id])
        folds.append(
            Fold(
                fold_id=fold_id,
                train_groups=tuple(str(value) for value in sorted(train_groups)),
                validation_groups=tuple(str(value) for value in sorted(validation_groups)),
                train_sample_ids=tuple(record.sample_id for record in train_records),
                validation_sample_ids=tuple(record.sample_id for record in validation_records),
                purge={
                    "strategy": "permanent_contiguous_inline_buffer",
                    "buffer_groups": width,
                    "adjacent_excluded_inline_groups": sorted(adjacent_buffers),
                    "nearest_train_validation_inline_distance": minimum_distance,
                },
                support={
                    "train_per_class_pixels": label_histogram(
                        train_records, schema.num_classes
                    ).tolist(),
                    "validation_per_class_pixels": label_histogram(
                        validation_records, schema.num_classes
                    ).tolist(),
                },
            )
        )

    downgrade_reason = None
    if effective < requested_n_splits:
        downgrade_reason = (
            f"requested {requested_n_splits} folds but support/buffer constraints allow {effective}; "
            + "; ".join(attempts)
        )
    manifest = SplitManifest(
        manifest_version="facies-p4-v1",
        group_key="inline",
        requested_n_splits=requested_n_splits,
        effective_n_splits=effective,
        downgrade_reason=downgrade_reason,
        test_groups=tuple(str(value) for value in test_lines),
        test_sample_ids=tuple(record.sample_id for record in test),
        development_groups=tuple(str(value) for value in sorted(core_groups)),
        development_sample_ids=tuple(record.sample_id for record in core_records),
        folds=tuple(folds),
        metadata={
            "track_id": "facies",
            "task_id": task_id,
            "label_version": LABEL_VERSIONS[task_id],
            "num_classes": schema.num_classes,
            "valid_label_ids": list(schema.valid_label_ids),
            "ignore_index": schema.ignore_index,
            "test_labels_read_during_split": False,
            "outer_split": expected,
            "observed_external_guard_groups": observed_external_gap,
            "all_saved_train_groups": dev_lines,
            "cv_core_groups": sorted(core_groups),
            "cv_excluded_buffer_groups": sorted(buffer_group_set),
            "cv_excluded_buffer_sample_ids": [record.sample_id for record in buffer_records],
            "cv_excluded_buffer_sample_count": len(buffer_records),
            "oof_coverage_scope": "all declared cv_core development samples exactly once",
        },
    )
    validate_manifest(manifest)
    return manifest
