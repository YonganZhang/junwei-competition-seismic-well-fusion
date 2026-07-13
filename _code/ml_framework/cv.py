"""Development-only cross-validation orchestration and OOF coverage checks."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping

from .artifacts import atomic_write_json
from .splits import Fold, SplitManifest, validate_manifest


FoldRunner = Callable[[Fold], Mapping[str, Any]]


def run_development_cv(
    manifest: SplitManifest,
    fold_runner: FoldRunner,
    *,
    output_dir: Path,
    primary_metric: str,
) -> dict[str, Any]:
    """Run locked development folds.

    There is intentionally no test data/loader argument in this API.  The
    runner must return exactly the current fold's validation sample IDs.
    """
    validate_manifest(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_records: list[dict[str, Any]] = []
    oof_sample_ids: list[str] = []
    metric_pairs: list[tuple[float, int]] = []

    for fold in manifest.folds:
        record = dict(fold_runner(fold))
        returned_ids = tuple(record.get("validation_sample_ids", ()))
        if set(returned_ids) != set(fold.validation_sample_ids) or len(returned_ids) != len(fold.validation_sample_ids):
            raise RuntimeError(f"fold {fold.fold_id} did not return exactly its locked validation samples")
        metrics = dict(record.get("metrics", {}))
        if primary_metric not in metrics:
            raise RuntimeError(f"fold {fold.fold_id} is missing primary metric {primary_metric!r}")
        score = float(metrics[primary_metric])
        if not math.isfinite(score):
            raise RuntimeError(f"fold {fold.fold_id} returned non-finite primary metric")
        valid_count = int(record.get("valid_label_count", len(returned_ids)))
        if valid_count <= 0:
            raise RuntimeError(f"fold {fold.fold_id} has zero valid labels")
        metric_pairs.append((score, valid_count))
        oof_sample_ids.extend(returned_ids)
        record.update(
            {
                "fold_id": fold.fold_id,
                "train_groups": fold.train_groups,
                "validation_groups": fold.validation_groups,
                "validation_sample_ids": returned_ids,
                "valid_label_count": valid_count,
            }
        )
        fold_records.append(record)
        atomic_write_json(output_dir / "folds" / f"fold_{fold.fold_id}" / "fold_result.json", record)

    if sorted(oof_sample_ids) != sorted(manifest.development_sample_ids):
        raise RuntimeError("OOF output does not contain every development sample exactly once")
    total_weight = sum(weight for _, weight in metric_pairs)
    weighted_score = sum(score * weight for score, weight in metric_pairs) / total_weight
    unweighted_mean = sum(score for score, _ in metric_pairs) / len(metric_pairs)
    summary = {
        "primary_metric": primary_metric,
        "primary_score_valid_label_weighted": weighted_score,
        "fold_mean_unweighted": unweighted_mean,
        "fold_std_unweighted": math.sqrt(
            sum((score - unweighted_mean) ** 2 for score, _ in metric_pairs) / len(metric_pairs)
        ),
        "worst_fold": min(score for score, _ in metric_pairs),
        "effective_n_splits": manifest.effective_n_splits,
        "oof_sample_count": len(oof_sample_ids),
        "valid_label_count": total_weight,
        "folds": fold_records,
    }
    atomic_write_json(output_dir / "oof" / "summary.json", summary)
    atomic_write_json(output_dir / "oof" / "sample_ids.json", oof_sample_ids)
    return summary
