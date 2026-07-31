#!/usr/bin/env python3
"""Audit and remove held-fold labels from P18 meta-selection fits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


TRACK = Path(__file__).resolve().parent
ROOT = TRACK.parents[2]
sys.path[:0] = [str(TRACK), str(ROOT)]

import p11_residual_fusion as base  # noqa: E402
import p17_foundation_geostatistics as p17  # noqa: E402
import p18_anisotropic_foundation_geostatistics as p18  # noqa: E402


def _rows(values: np.ndarray) -> set[tuple[int, int, int]]:
    return set(map(tuple, np.asarray(values, dtype=np.int64).tolist()))


def _without_coordinates(
    fold: p17.FoldSamples,
    forbidden: set[tuple[int, int, int]],
) -> tuple[p17.FoldSamples, int]:
    keep = np.asarray(
        [tuple(row) not in forbidden for row in fold.train_indices_kji],
        dtype=bool,
    )
    removed = int(np.sum(~keep))
    return (
        p17.FoldSamples(
            fold_id=fold.fold_id,
            train_target=fold.train_target[keep],
            train_raw_features=fold.train_raw_features[keep],
            validation_raw_features=fold.validation_raw_features,
            train_indices_kji=fold.train_indices_kji[keep],
            validation_indices_kji=fold.validation_indices_kji,
            source_hashes=fold.source_hashes,
        ),
        removed,
    )


def _candidate_bank(
    *,
    oof: base.OOFDevelopment,
    folds: tuple[p17.FoldSamples, ...],
    requested_indices: np.ndarray,
    foundation_features: np.ndarray,
    vertical_weights: tuple[float, ...] = p18.VERTICAL_WEIGHTS,
    foundation_weights: tuple[float, ...] = p18.FOUNDATION_WEIGHTS,
    seismic_weights: tuple[float, ...] = p18.SEISMIC_WEIGHTS,
    neighbour_counts: tuple[int, ...] = p18.NEIGHBOUR_COUNTS,
    distance_powers: tuple[float, ...] = p18.DISTANCE_POWERS,
    blend_weights: tuple[float, ...] = p18.BLEND_WEIGHTS,
) -> dict[str, np.ndarray]:
    prepared, _ = p18._prepare_fold_metrics(  # noqa: SLF001
        folds=folds,
        requested_indices=requested_indices,
        foundation_features=foundation_features,
    )
    triples = tuple(
        (vertical, foundation, seismic)
        for vertical in vertical_weights
        for foundation in foundation_weights
        for seismic in seismic_weights
    )
    return p18._build_candidate_bank(  # noqa: SLF001
        oof=oof,
        prepared=prepared,
        metric_triples=triples,
        neighbour_counts=neighbour_counts,
        distance_powers=distance_powers,
        blend_weights=blend_weights,
    )


def run(*, data_dir: Path, stage3_root: Path, output_dir: Path) -> dict:
    base.ensure_no_holdout_paths((data_dir, stage3_root, output_dir))
    inputs = base.resolve_dev_inputs(data_dir)
    oof = base.load_oof_development(stage3_root)
    folds, sample_audit = p17.load_fold_samples(
        stage3_root=stage3_root,
        train_h5=inputs.train_h5,
        oof=oof,
    )
    requested = p17._unique_indices(folds)  # noqa: SLF001
    cache_indices, foundation, feature_audit = p18._load_feature_cache(  # noqa: SLF001
        p18.DEFAULT_FEATURE_CACHE,
        train_h5=inputs.train_h5,
        expected_indices=requested,
    )
    original_bank = _candidate_bank(
        oof=oof,
        folds=folds,
        requested_indices=cache_indices,
        foundation_features=foundation,
    )
    names = list(original_bank)
    original_matrix = np.stack([original_bank[name] for name in names])
    corrected = np.full(len(oof.target), np.nan, dtype=np.float64)
    selections: list[dict] = []
    overlap_audit: list[dict] = []

    for held_fold in base.FOLD_IDS:
        held = next(fold for fold in folds if fold.fold_id == held_fold)
        forbidden = _rows(held.validation_indices_kji)
        purged_folds: list[p17.FoldSamples] = []
        removed_by_fold: dict[str, int] = {}
        for fold in folds:
            if fold.fold_id == held_fold:
                purged_folds.append(fold)
                removed_by_fold[str(fold.fold_id)] = 0
                continue
            purged, removed = _without_coordinates(fold, forbidden)
            purged_folds.append(purged)
            removed_by_fold[str(fold.fold_id)] = removed
        purged_bank = _candidate_bank(
            oof=oof,
            folds=tuple(purged_folds),
            requested_indices=cache_indices,
            foundation_features=foundation,
        )
        selection_mask = oof.fold_ids != held_fold
        selection_rmse = np.asarray(
            [
                p17._metrics(  # noqa: SLF001
                    oof.target[selection_mask],
                    purged_bank[name][selection_mask],
                )["rmse"]
                for name in names
            ],
            dtype=np.float64,
        )
        chosen = np.argsort(selection_rmse, kind="stable")[
            : p18.NESTED_ENSEMBLE_SIZE
        ]
        held_mask = oof.fold_ids == held_fold
        corrected[held_mask] = np.mean(
            original_matrix[chosen][:, held_mask], axis=0
        )
        selections.append(
            {
                "held_out_fold": int(held_fold),
                "selection_folds": [
                    int(value) for value in base.FOLD_IDS if value != held_fold
                ],
                "held_validation_labels_removed_from_meta_fit": True,
                "removed_train_labels_by_selection_fold": removed_by_fold,
                "removed_train_labels_total": int(sum(removed_by_fold.values())),
                "selected_candidates": [names[position] for position in chosen],
                "selection_rmse": [
                    float(selection_rmse[position]) for position in chosen
                ],
            }
        )
        overlap_audit.append(
            {
                "held_out_fold": int(held_fold),
                "held_validation_rows": int(len(forbidden)),
                "overlap_with_other_fold_train_union": int(
                    len(
                        forbidden
                        & set().union(
                            *(
                                _rows(fold.train_indices_kji)
                                for fold in folds
                                if fold.fold_id != held_fold
                            )
                        )
                    )
                ),
                "removed_occurrences": int(sum(removed_by_fold.values())),
            }
        )

    if not np.all(np.isfinite(corrected)):
        raise RuntimeError("corrected nested prediction is incomplete")
    original_summary = json.loads(
        (
            TRACK
            / "_outputs"
            / "p18_anisotropic_foundation_geostatistics"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    original = np.load(
        TRACK
        / "_outputs"
        / "p18_anisotropic_foundation_geostatistics"
        / "prediction_errors.npz",
        allow_pickle=False,
    )["candidate_prediction"]
    corrected_metrics = p17._metrics(oof.target, corrected)  # noqa: SLF001
    baseline_metrics = p17._metrics(oof.target, oof.baseline)  # noqa: SLF001
    original_metrics = p17._metrics(oof.target, original)  # noqa: SLF001
    fold_rows, counts = p18._fold_results(  # noqa: SLF001
        target=oof.target,
        baseline=oof.baseline,
        candidate=corrected,
        fold_ids=oof.fold_ids,
    )
    bootstrap = p17._whole_fold_bootstrap(  # noqa: SLF001
        target=oof.target,
        baseline=oof.baseline,
        candidate=corrected,
        fold_ids=oof.fold_ids,
    )
    result = {
        "schema_version": "reconstruction-p19-meta-purged-geostatistics/v1",
        "objective": "P18 nested selection with held-fold labels removed from every meta-fit",
        "baseline": baseline_metrics,
        "p18_reported": original_metrics,
        "meta_purged": corrected_metrics,
        "relative_rmse_change_vs_pykrige": (
            corrected_metrics["rmse"] - baseline_metrics["rmse"]
        )
        / baseline_metrics["rmse"],
        "rmse_change_vs_reported_p18": (
            corrected_metrics["rmse"] - original_metrics["rmse"]
        ),
        "per_fold": fold_rows,
        "outcome_counts": counts,
        "whole_fold_bootstrap": bootstrap,
        "selections": selections,
        "overlap_audit": overlap_audit,
        "sample_audit": sample_audit,
        "feature_audit": feature_audit,
        "source_p18_script_sha256": original_summary["implementation"][
            "script_sha256"
        ],
        "firewall": {
            "train_labels_per_fold_before_meta_purge": 512,
            "held_prediction_uses_original_held_fold_512_train_labels": True,
            "test_h5_opened": False,
            "frozen_holdout_opened": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta_purge_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "meta_purge_predictions.npz",
        indices_kji=oof.indices_kji,
        fold_ids=oof.fold_ids,
        target=oof.target,
        baseline_prediction=oof.baseline,
        p18_prediction=original,
        meta_purged_prediction=corrected,
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT.parent / "track-reconstruction" / "_data" / "processed" / "reconstruction",
    )
    parser.add_argument(
        "--stage3-root",
        type=Path,
        default=ROOT.parent / "p5-stage3-reconstruction" / "_tmp" / "p5_stage3_reconstruction",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRACK / "_outputs" / "p19_meta_purged_geostatistics",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run(**vars(args))
    print(
        json.dumps(
            {
                "baseline_rmse": result["baseline"]["rmse"],
                "reported_p18_rmse": result["p18_reported"]["rmse"],
                "meta_purged_rmse": result["meta_purged"]["rmse"],
                "outcome_counts": result["outcome_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
