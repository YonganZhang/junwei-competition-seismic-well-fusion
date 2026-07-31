"""Blind-test audit and buffered spatial CV for sparse fault-stick labels."""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRACK_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _code.ml_framework.artifacts import atomic_write_json, hash_file, hash_payload  # noqa: E402


@dataclass(frozen=True)
class SpatialSample:
    sample_id: str
    inline: int
    positive_count: int
    verified_negative_count: int
    proxy_count: int = 0

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if min(self.positive_count, self.verified_negative_count, self.proxy_count) < 0:
            raise ValueError("label support counts must be non-negative")


@dataclass(frozen=True)
class BufferedFold:
    fold_id: int
    validation_block_id: str
    train_block_ids: tuple[str, ...]
    train_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    train_inline_ranges: tuple[tuple[int, int], ...]
    validation_inline_range: tuple[int, int]
    buffer_inlines: int
    support: Mapping[str, int]


@dataclass(frozen=True)
class BufferedCVPlan:
    plan_version: str
    requested_n_splits: int
    effective_n_splits: int
    status: str
    downgrade_reason: str | None
    buffer_inlines: int
    development_sample_ids: tuple[str, ...]
    excluded_buffer_sample_ids: tuple[str, ...]
    folds: tuple[BufferedFold, ...]
    block_support: Mapping[str, Mapping[str, int]]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def stable_hash(self) -> str:
        return hash_payload(self.to_dict())


def _partition_for_k(
    samples: Sequence[SpatialSample],
    *,
    n_splits: int,
    buffer_inlines: int,
) -> tuple[list[list[SpatialSample]], list[SpatialSample], list[float]]:
    minimum = min(sample.inline for sample in samples)
    maximum = max(sample.inline for sample in samples)
    span = maximum - minimum + 1
    boundaries = [minimum + span * index / n_splits for index in range(1, n_splits)]
    blocks: list[list[SpatialSample]] = [[] for _ in range(n_splits)]
    excluded: list[SpatialSample] = []
    for sample in samples:
        if any(abs(sample.inline - boundary) <= buffer_inlines for boundary in boundaries):
            excluded.append(sample)
            continue
        block_index = min(int((sample.inline - minimum) * n_splits / span), n_splits - 1)
        blocks[block_index].append(sample)
    return blocks, excluded, boundaries


def _support(samples: Sequence[SpatialSample]) -> dict[str, int]:
    return {
        "samples": len(samples),
        "positive_labels": sum(sample.positive_count for sample in samples),
        "verified_negative_labels": sum(sample.verified_negative_count for sample in samples),
        "proxy_labels": sum(sample.proxy_count for sample in samples),
    }


def build_buffered_spatial_cv(
    samples: Sequence[SpatialSample],
    *,
    requested_n_splits: int = 5,
    buffer_inlines: int = 8,
) -> BufferedCVPlan:
    """Build globally separated contiguous blocks and honest supported folds.

    Samples close to every block boundary are excluded from the development
    universe once, so no fold can silently train across the declared buffer.
    A formal binary fold requires both stick positives and audited negatives in
    every validation block and in its complementary training blocks.  Proxy
    negatives never satisfy this support check.
    """

    if requested_n_splits < 2:
        raise ValueError("requested_n_splits must be >=2")
    if buffer_inlines < 1:
        raise ValueError("buffer_inlines must be >=1")
    if not samples:
        raise ValueError("cannot build fault CV from zero samples")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("fault CV sample IDs must be unique")

    unique_inlines = sorted({sample.inline for sample in samples})
    search_start = min(requested_n_splits, len(unique_inlines))
    failure_reasons: list[str] = []
    for effective in range(search_start, 1, -1):
        blocks, excluded, boundaries = _partition_for_k(
            samples,
            n_splits=effective,
            buffer_inlines=buffer_inlines,
        )
        supports = [_support(block) for block in blocks]
        unsupported = [
            index
            for index, support in enumerate(supports)
            if support["samples"] == 0
            or support["positive_labels"] == 0
            or support["verified_negative_labels"] == 0
        ]
        if unsupported:
            failure_reasons.append(
                f"{effective}-fold blocks without both positive and verified-negative support: {unsupported}"
            )
            continue

        folds: list[BufferedFold] = []
        block_support: dict[str, Mapping[str, int]] = {}
        for block_id, support in enumerate(supports):
            block_support[f"block-{block_id}"] = support
        for validation_index, validation_samples in enumerate(blocks):
            train_samples = [
                sample
                for block_index, block in enumerate(blocks)
                if block_index != validation_index
                for sample in block
            ]
            train_support = _support(train_samples)
            if train_support["positive_labels"] == 0 or train_support["verified_negative_labels"] == 0:
                failure_reasons.append(f"{effective}-fold train complement lacks binary support")
                folds = []
                break
            train_ranges = tuple(
                (min(sample.inline for sample in block), max(sample.inline for sample in block))
                for block_index, block in enumerate(blocks)
                if block_index != validation_index
            )
            folds.append(
                BufferedFold(
                    fold_id=validation_index,
                    validation_block_id=f"block-{validation_index}",
                    train_block_ids=tuple(
                        f"block-{index}" for index in range(effective) if index != validation_index
                    ),
                    train_sample_ids=tuple(sample.sample_id for sample in train_samples),
                    validation_sample_ids=tuple(sample.sample_id for sample in validation_samples),
                    train_inline_ranges=train_ranges,
                    validation_inline_range=(
                        min(sample.inline for sample in validation_samples),
                        max(sample.inline for sample in validation_samples),
                    ),
                    buffer_inlines=buffer_inlines,
                    support={
                        "train_positive_labels": train_support["positive_labels"],
                        "train_verified_negative_labels": train_support["verified_negative_labels"],
                        "validation_positive_labels": supports[validation_index]["positive_labels"],
                        "validation_verified_negative_labels": supports[validation_index][
                            "verified_negative_labels"
                        ],
                    },
                )
            )
        if not folds:
            continue

        development_ids = tuple(sample.sample_id for block in blocks for sample in block)
        seen_oof = [sample_id for fold in folds for sample_id in fold.validation_sample_ids]
        if sorted(seen_oof) != sorted(development_ids):
            raise AssertionError("buffered CV OOF construction is incomplete")
        reason_parts: list[str] = []
        if effective < requested_n_splits:
            reason_parts.append(
                f"requested {requested_n_splits}, reduced to {effective} by independent block/label support"
            )
        if excluded:
            reason_parts.append(
                f"excluded {len(excluded)} samples inside global {buffer_inlines}-inline boundary buffers"
            )
        return BufferedCVPlan(
            plan_version="fault-buffered-cv-v1",
            requested_n_splits=requested_n_splits,
            effective_n_splits=effective,
            status="ready",
            downgrade_reason="; ".join(reason_parts) or None,
            buffer_inlines=buffer_inlines,
            development_sample_ids=development_ids,
            excluded_buffer_sample_ids=tuple(sample.sample_id for sample in excluded),
            folds=tuple(folds),
            block_support=block_support,
            metadata={
                "boundaries": boundaries,
                "negative_support_policy": "verified_negative_only",
                "proxy_counts_never_enable_a_fold": True,
            },
        )

    reason = "; ".join(failure_reasons[-3:])
    if not any(sample.verified_negative_count for sample in samples):
        reason = "no audited verified-negative labels; proxy/unknown voxels cannot enable binary CV"
    return BufferedCVPlan(
        plan_version="fault-buffered-cv-v1",
        requested_n_splits=requested_n_splits,
        effective_n_splits=0,
        status="not_feasible",
        downgrade_reason=reason or "fewer than two independently supported buffered blocks",
        buffer_inlines=buffer_inlines,
        development_sample_ids=(),
        excluded_buffer_sample_ids=(),
        folds=(),
        block_support={},
        metadata={
            "observed_unique_inlines": len(unique_inlines),
            "negative_support_policy": "verified_negative_only",
            "proxy_counts_never_enable_a_fold": True,
        },
    )


FoldRunner = Callable[[BufferedFold], Mapping[str, Any]]


def run_buffered_development_cv(
    plan: BufferedCVPlan,
    fold_runner: FoldRunner,
    *,
    output_dir: Path,
    primary_metric: str = "average_precision",
    metric_direction: str = "maximize",
) -> dict[str, Any]:
    """Run development folds and archive exact pooled OOF records.

    This development API deliberately has no test argument or test loader.
    """

    if plan.status != "ready" or plan.effective_n_splits < 2:
        raise RuntimeError(f"fault development CV is not feasible: {plan.downgrade_reason}")
    if metric_direction != "maximize":
        raise ValueError("fault average_precision direction is frozen to 'maximize'")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "split_manifest.json", plan.to_dict())

    fold_records: list[dict[str, Any]] = []
    oof_records: list[dict[str, Any]] = []
    metric_pairs: list[tuple[float, int]] = []
    for fold in plan.folds:
        record = dict(fold_runner(fold))
        returned_ids = tuple(record.get("validation_sample_ids", ()))
        if len(returned_ids) != len(fold.validation_sample_ids) or set(returned_ids) != set(
            fold.validation_sample_ids
        ):
            raise RuntimeError(f"fold {fold.fold_id} did not return its locked validation samples")
        predictions = [dict(item) for item in record.pop("oof_predictions", ())]
        prediction_ids = [str(item.get("sample_id", "")) for item in predictions]
        if len(prediction_ids) != len(fold.validation_sample_ids) or set(prediction_ids) != set(
            fold.validation_sample_ids
        ):
            raise RuntimeError(f"fold {fold.fold_id} did not archive one OOF prediction per sample")
        metrics = dict(record.get("metrics", {}))
        if primary_metric not in metrics:
            raise RuntimeError(f"fold {fold.fold_id} is missing primary metric {primary_metric!r}")
        score = float(metrics[primary_metric])
        valid_count = int(record.get("valid_label_count", 0))
        if not math.isfinite(score) or valid_count <= 0:
            raise RuntimeError(f"fold {fold.fold_id} returned invalid metric evidence")
        metric_pairs.append((score, valid_count))
        oof_records.extend(predictions)
        record.update(
            {
                "fold_id": fold.fold_id,
                "validation_sample_ids": returned_ids,
                "valid_label_count": valid_count,
            }
        )
        fold_records.append(record)
        atomic_write_json(output_dir / "folds" / f"fold_{fold.fold_id}" / "fold_result.json", record)

    oof_ids = [str(record["sample_id"]) for record in oof_records]
    if len(oof_ids) != len(set(oof_ids)) or set(oof_ids) != set(plan.development_sample_ids):
        raise RuntimeError("pooled OOF records do not cover development exactly once")
    total_weight = sum(weight for _, weight in metric_pairs)
    weighted = sum(score * weight for score, weight in metric_pairs) / total_weight
    mean = sum(score for score, _ in metric_pairs) / len(metric_pairs)
    scores = [score for score, _ in metric_pairs]
    summary = {
        "primary_metric": primary_metric,
        "metric_direction": metric_direction,
        "primary_score_valid_label_weighted": weighted,
        "fold_mean_unweighted": mean,
        "fold_std_unweighted": math.sqrt(sum((score - mean) ** 2 for score in scores) / len(scores)),
        "worst_fold": min(scores) if metric_direction == "maximize" else max(scores),
        "requested_n_splits": plan.requested_n_splits,
        "effective_n_splits": plan.effective_n_splits,
        "valid_label_count": total_weight,
        "oof_sample_count": len(oof_records),
        "split_hash": plan.stable_hash(),
        "folds": fold_records,
    }
    atomic_write_json(output_dir / "oof" / "predictions.json", oof_records)
    atomic_write_json(output_dir / "oof" / "summary.json", summary)
    return summary


def spatial_samples_from_audited_manifest(manifest: Mapping[str, Any]) -> list[SpatialSample]:
    """Convert historical development centres without upgrading weak negatives."""

    records: list[SpatialSample] = []
    for index, centre in enumerate(manifest["sample_centres"]["train"]):
        kind = str(centre["sample_kind"])
        records.append(
            SpatialSample(
                sample_id=(
                    f"audited-v2:il{int(centre['inline'])}:xl{int(centre['crossline'])}:"
                    f"t{int(centre['time_index'])}:n{index}"
                ),
                inline=int(centre["inline"]),
                positive_count=1 if kind == "fault" else 0,
                verified_negative_count=0,
                proxy_count=1 if kind == "non_fault" else 0,
            )
        )
    return records


def _candidate_complete_blocks(coverage_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for raw in coverage_payload.get("blocks", ()):  # absent coverage is intentionally not inferred
        block = dict(raw)
        bounds = block.get("inline_range")
        if (
            block.get("annotation_status") == "complete"
            and block.get("contiguous") is True
            and isinstance(bounds, list)
            and len(bounds) == 2
            and int(bounds[0]) <= int(bounds[1])
            and int(block.get("positive_labels", 0)) > 0
            and int(block.get("verified_negative_labels", 0)) > 0
        ):
            candidates.append(block)
    return candidates


def audit_blind_test(
    *,
    fault_points_path: Path,
    seismic_index_path: Path,
    audited_build_summary_path: Path,
    audited_split_manifest_path: Path,
    annotation_coverage_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze an unconsumed complete block, or emit deterministic not-feasible evidence."""

    required = (
        fault_points_path,
        seismic_index_path,
        audited_build_summary_path,
        audited_split_manifest_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"blind-test audit inputs are missing: {missing}")
    with np.load(fault_points_path, allow_pickle=True) as faults:
        point_inlines = np.asarray(faults["inline"], dtype=np.int64)
        point_count = int(len(point_inlines))
    with np.load(seismic_index_path, allow_pickle=False) as index:
        inline_extent = [int(index["il_min"]), int(index["il_max"])]
    if point_count == 0 or point_inlines.min() < inline_extent[0] or point_inlines.max() > inline_extent[1]:
        raise ValueError("fault points do not lie inside the seismic coordinate extent")

    summary = json.loads(audited_build_summary_path.read_text(encoding="utf-8"))
    historical_split = json.loads(audited_split_manifest_path.read_text(encoding="utf-8"))
    coverage_payload: Mapping[str, Any] = {}
    if annotation_coverage_path is not None and annotation_coverage_path.is_file():
        coverage_payload = json.loads(annotation_coverage_path.read_text(encoding="utf-8"))
    candidates = _candidate_complete_blocks(coverage_payload)
    historical_train_range = tuple(
        int(value) for value in historical_split["split_plan"]["train"]
    )
    consumed_range = tuple(int(value) for value in historical_split["split_plan"]["test"])
    exposed_ranges = (historical_train_range, consumed_range)
    unconsumed = [
        block
        for block in candidates
        if all(
            int(block["inline_range"][1]) < exposed[0]
            or int(block["inline_range"][0]) > exposed[1]
            for exposed in exposed_ranges
        )
    ]
    selected = sorted(unconsumed, key=lambda block: tuple(int(v) for v in block["inline_range"]))[0] if unconsumed else None

    def source_name(path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path.resolve())

    sources = {source_name(path): hash_file(path) for path in required}
    if annotation_coverage_path is not None and annotation_coverage_path.is_file():
        sources[source_name(annotation_coverage_path)] = hash_file(annotation_coverage_path)
    common = {
        "audit_version": "fault-blind-test-v1",
        "track_id": "fault",
        "task_id": "fault_stick_segmentation",
        "searched_seismic_inline_extent": inline_extent,
        "fault_point_count": point_count,
        "source_sha256": sources,
        "annotation_coverage_evidence": (
            source_name(annotation_coverage_path)
            if annotation_coverage_path is not None and annotation_coverage_path.is_file()
            else None
        ),
        "complete_candidate_blocks": candidates,
        "consumed_regression_blocks": [
            {
                "run_id": "audited_v2",
                "inline_range": list(historical_train_range),
                "role": "historical_development_exposure",
                "model_training_already_observed": True,
            },
            {
                "run_id": "audited_v2",
                "inline_range": list(consumed_range),
                "role": "regression_evidence_only",
                "test_metrics_already_observed": True,
                "build_dataset_sha256": summary["dataset_sha256"],
            }
        ],
        "label_semantics": {
            "fault_stick": "positive",
            "unlabelled": "unknown; valid_label_mask=false",
            "weak_negative": "proxy only",
            "formal_negative": "requires complete annotation coverage audit",
        },
    }
    if selected is None:
        result = {
            **common,
            "status": "not_feasible",
            "reason_code": "NO_UNCONSUMED_COMPLETE_ANNOTATION_BLOCK",
            "reason": (
                "No continuous block has evidence of complete positive and verified-negative annotation "
                "coverage outside the already observed audited_v2 test range."
            ),
            "required_to_unblock": [
                "a contiguous block with annotation_status=complete",
                "audited positive and verified-negative label counts",
                "coordinates disjoint from consumed regression blocks",
            ],
            "prohibited_fallbacks": [
                "treat unlabelled voxels as background",
                "rename audited_v2 as blind test",
                "use proxy negatives for formal test metrics",
            ],
        }
    else:
        result = {
            **common,
            "status": "frozen",
            "blind_test_block": selected,
            "split_hash": hash_payload(selected),
            "selection_rule": "first coordinate-sorted unconsumed completely annotated contiguous block",
        }
    result["audit_hash"] = hash_payload(result)
    if output_path is not None:
        atomic_write_json(output_path, result)
    return result
