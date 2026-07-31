"""Leakage-safe group fold manifests with an already-frozen test set."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    train_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    purge: Mapping[str, Any]
    support: Mapping[str, Any]


@dataclass(frozen=True)
class SplitManifest:
    manifest_version: str
    group_key: str
    requested_n_splits: int
    effective_n_splits: int
    downgrade_reason: str | None
    test_groups: tuple[str, ...]
    test_sample_ids: tuple[str, ...]
    development_groups: tuple[str, ...]
    development_sample_ids: tuple[str, ...]
    folds: tuple[Fold, ...]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def stable_hash(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_sample_groups(sample_ids: Sequence[str], groups: Sequence[str]) -> None:
    if len(sample_ids) != len(groups):
        raise ValueError("sample_ids and groups must have the same length")
    if not sample_ids:
        raise ValueError("cannot build a split manifest from zero samples")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be globally unique")
    if any(not group for group in groups):
        raise ValueError("group IDs must be non-empty")


def build_group_folds(
    sample_ids: Sequence[str],
    groups: Sequence[str],
    *,
    group_key: str,
    test_groups: Sequence[str],
    requested_n_splits: int = 5,
    seed: int = 2693,
    max_splits_by_support: int | None = None,
    support_reason: str | None = None,
    purge: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SplitManifest:
    _validate_sample_groups(sample_ids, groups)
    if requested_n_splits < 2:
        raise ValueError("requested_n_splits must be >=2")
    declared_test_groups = tuple(dict.fromkeys(test_groups))
    if not declared_test_groups:
        raise ValueError("test_groups must be frozen before building development folds")
    available_groups = set(groups)
    unknown_test = sorted(set(declared_test_groups) - available_groups)
    if unknown_test:
        raise ValueError(f"test groups are absent from data: {unknown_test}")

    test_pairs = [(sid, group) for sid, group in zip(sample_ids, groups) if group in declared_test_groups]
    dev_pairs = [(sid, group) for sid, group in zip(sample_ids, groups) if group not in declared_test_groups]
    dev_groups = sorted({group for _, group in dev_pairs})
    if len(dev_groups) < 2:
        raise ValueError("at least two independent development groups are required")
    effective = min(requested_n_splits, len(dev_groups))
    reasons: list[str] = []
    if effective < requested_n_splits:
        reasons.append(f"only {len(dev_groups)} independent development groups are available")
    if max_splits_by_support is not None:
        if max_splits_by_support < 2:
            raise ValueError("support constraints leave fewer than two valid folds")
        if max_splits_by_support < effective:
            effective = max_splits_by_support
            reasons.append(support_reason or f"label/class/positive support limits folds to {effective}")

    shuffled = list(dev_groups)
    random.Random(seed).shuffle(shuffled)
    validation_buckets: list[list[str]] = [[] for _ in range(effective)]
    for index, group in enumerate(shuffled):
        validation_buckets[index % effective].append(group)

    folds: list[Fold] = []
    all_dev = set(dev_groups)
    for fold_id, validation_groups in enumerate(validation_buckets):
        validation_set = set(validation_groups)
        train_set = all_dev - validation_set
        train_ids = tuple(sid for sid, group in dev_pairs if group in train_set)
        validation_ids = tuple(sid for sid, group in dev_pairs if group in validation_set)
        folds.append(
            Fold(
                fold_id=fold_id,
                train_groups=tuple(sorted(train_set)),
                validation_groups=tuple(sorted(validation_set)),
                train_sample_ids=train_ids,
                validation_sample_ids=validation_ids,
                purge=dict(purge or {}),
                support={},
            )
        )

    manifest = SplitManifest(
        manifest_version="p4-v1",
        group_key=group_key,
        requested_n_splits=requested_n_splits,
        effective_n_splits=effective,
        downgrade_reason="; ".join(reasons) or None,
        test_groups=tuple(sorted(declared_test_groups)),
        test_sample_ids=tuple(sid for sid, _ in test_pairs),
        development_groups=tuple(sorted(dev_groups)),
        development_sample_ids=tuple(sid for sid, _ in dev_pairs),
        folds=tuple(folds),
        metadata=dict(metadata or {}),
    )
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: SplitManifest) -> None:
    test_ids = set(manifest.test_sample_ids)
    dev_ids = set(manifest.development_sample_ids)
    if test_ids & dev_ids:
        raise ValueError("test and development sample IDs overlap")
    if set(manifest.test_groups) & set(manifest.development_groups):
        raise ValueError("test and development groups overlap")
    oof_ids: list[str] = []
    for fold in manifest.folds:
        train_ids = set(fold.train_sample_ids)
        val_ids = set(fold.validation_sample_ids)
        if train_ids & val_ids:
            raise ValueError(f"fold {fold.fold_id} train/validation samples overlap")
        if set(fold.train_groups) & set(fold.validation_groups):
            raise ValueError(f"fold {fold.fold_id} train/validation groups overlap")
        if (train_ids | val_ids) != dev_ids:
            raise ValueError(f"fold {fold.fold_id} does not cover all development samples")
        oof_ids.extend(fold.validation_sample_ids)
    if sorted(oof_ids) != sorted(manifest.development_sample_ids):
        raise ValueError("OOF contract violated: every development sample must appear exactly once in validation")
