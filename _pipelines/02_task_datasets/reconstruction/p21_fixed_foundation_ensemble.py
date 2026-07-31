#!/usr/bin/env python3
"""P21 fixed foundation-kernel ensemble and PEFT residual audit.

P19 selected candidate kernels separately for each held spatial fold.  P21
removes that unstable meta-selection and averages the same three preregistered
foundation kernels in every fold.  The legal 512-label outer training budget,
PyKrige OOF base, metrics, and holdout firewall remain unchanged.

Target-free multi-view LoRA residual experiments were run in the sandbox and
are recorded as rejected diagnostics; they are not part of this promoted
predictor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(PROJECT_ROOT)]

import p11_residual_fusion as base  # noqa: E402
import p17_foundation_geostatistics as p17  # noqa: E402
import p18_anisotropic_foundation_geostatistics as p18  # noqa: E402


SCHEMA_VERSION = "reconstruction-p21-fixed-foundation-ensemble/v1"
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "p21_fixed_foundation_ensemble"
VERTICAL_WEIGHT = 4.0
FOUNDATION_WEIGHT = 0.1
SEISMIC_WEIGHTS = (0.0, 0.1, 0.2)
NEIGHBOURS = 64
DISTANCE_POWER = 1.5
BLEND_WEIGHT = 0.75
EXPECTED_P19_RMSE = 0.027751397627827728
EXPECTED_CANDIDATE_RMSE = 0.027734374378067677
REJECTED_RESIDUAL_ROUTES = (
    {
        "route": "contrastive_lora_neural_residual",
        "rmse": 0.028278976996868686,
        "rmse_delta_vs_p19": 0.0005275793690409578,
        "verdict": "reject_constant_bias_collapse",
    },
    {
        "route": "contrastive_lora_calibrated_ridge_residual",
        "rmse": 0.02961841822716806,
        "rmse_delta_vs_p19": 0.0018670205993403313,
        "verdict": "reject_single_calibration_instability",
    },
    {
        "route": "contrastive_lora_fivefold_ridge_residual",
        "rmse": 0.02808003976063105,
        "rmse_delta_vs_p19": 0.00032864213280332094,
        "verdict": "reject_spatially_nontransferable_residual",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "valid_voxels": int(len(error)),
    }


def _load_p19(path: Path, oof: base.OOFDevelopment) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["indices_kji"], oof.indices_kji)
        np.testing.assert_array_equal(payload["fold_ids"], oof.fold_ids)
        np.testing.assert_allclose(payload["target"], oof.target, rtol=0.0, atol=0.0)
        prediction = np.asarray(payload["meta_purged_prediction"], dtype=np.float64)
    metrics = _metrics(oof.target, prediction)
    if not np.isclose(metrics["rmse"], EXPECTED_P19_RMSE, rtol=0.0, atol=1e-12):
        raise RuntimeError("P21 P19 reference identity drift")
    return prediction


def _fixed_candidate(
    *,
    oof: base.OOFDevelopment,
    folds: Sequence[p17.FoldSamples],
    requested_indices: np.ndarray,
    foundation_features: np.ndarray,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    prepared, transform_audits = p18._prepare_fold_metrics(  # noqa: SLF001
        folds=folds,
        requested_indices=requested_indices,
        foundation_features=foundation_features,
    )
    triples = tuple(
        (VERTICAL_WEIGHT, FOUNDATION_WEIGHT, seismic)
        for seismic in SEISMIC_WEIGHTS
    )
    bank = p18._build_candidate_bank(  # noqa: SLF001
        oof=oof,
        prepared=prepared,
        metric_triples=triples,
        neighbour_counts=(NEIGHBOURS,),
        distance_powers=(DISTANCE_POWER,),
        blend_weights=(BLEND_WEIGHT,),
    )
    names = [
        p18._candidate_name(  # noqa: SLF001
            VERTICAL_WEIGHT,
            FOUNDATION_WEIGHT,
            seismic,
            NEIGHBOURS,
            DISTANCE_POWER,
            BLEND_WEIGHT,
        )
        for seismic in SEISMIC_WEIGHTS
    ]
    if set(names) != set(bank):
        raise RuntimeError("P21 fixed candidate bank identity drift")
    prediction = np.mean(np.stack([bank[name] for name in names]), axis=0)
    return prediction, names, transform_audits


def _fold_audit(
    *,
    target: np.ndarray,
    pykrige: np.ndarray,
    p19_prediction: np.ndarray,
    candidate: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    outcomes = {"win": 0, "loss": 0, "tie": 0}
    for fold_id in base.FOLD_IDS:
        mask = fold_ids == fold_id
        p19_metrics = _metrics(target[mask], p19_prediction[mask])
        candidate_metrics = _metrics(target[mask], candidate[mask])
        delta = float(candidate_metrics["rmse"]) - float(p19_metrics["rmse"])
        outcome = "win" if delta < -1e-12 else "loss" if delta > 1e-12 else "tie"
        outcomes[outcome] += 1
        rows.append(
            {
                "fold_id": int(fold_id),
                "pykrige": _metrics(target[mask], pykrige[mask]),
                "p19": p19_metrics,
                "candidate": candidate_metrics,
                "rmse_delta_candidate_minus_p19": delta,
                "maximum_absolute_prediction_change": float(
                    np.max(np.abs(candidate[mask] - p19_prediction[mask]))
                ),
                "outcome": outcome,
            }
        )
    return rows, outcomes


def _write_evidence(output_dir: Path, result: Mapping[str, Any]) -> None:
    comparison = result["comparison"]
    lines = [
        "# P21 fixed foundation ensemble",
        "",
        f"- P19 pooled OOF RMSE: `{comparison['p19']['rmse']:.12f}`.",
        f"- Fixed ensemble pooled OOF RMSE: `{comparison['candidate']['rmse']:.12f}`.",
        f"- RMSE delta: `{comparison['rmse_delta_candidate_minus_p19']:.12f}`.",
        f"- Fold outcomes vs P19: `{comparison['outcomes_vs_p19']}`.",
        "",
        "The promoted route removes per-fold meta-selection and always averages",
        "the z=4, foundation=0.1, seismic={0,0.1,0.2}, k=64, p=1.5,",
        "blend=0.75 kernels. Four folds are prediction-equivalent to P19; the",
        "remaining fold improves. The whole-fold bootstrap does not establish a",
        "broad statistical effect, so the decision is a deterministic simplicity",
        "win rather than a causal foundation-model claim.",
        "",
        "Target-free contrastive LoRA residual routes were rejected because their",
        "inner calibration did not transfer across outer spatial folds.",
    ]
    (output_dir / "evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    data_dir: Path,
    stage3_root: Path,
    p19_predictions: Path,
    output_dir: Path,
) -> dict[str, Any]:
    base.ensure_no_holdout_paths((data_dir, stage3_root, p19_predictions, output_dir))
    inputs = base.resolve_dev_inputs(data_dir)
    oof = base.load_oof_development(stage3_root)
    folds, fold_loading_audit = p17.load_fold_samples(
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
    p19_prediction = _load_p19(p19_predictions, oof)
    candidate, names, transform_audits = _fixed_candidate(
        oof=oof,
        folds=folds,
        requested_indices=cache_indices,
        foundation_features=foundation,
    )
    candidate_metrics = _metrics(oof.target, candidate)
    if not np.isclose(
        candidate_metrics["rmse"], EXPECTED_CANDIDATE_RMSE, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("P21 fixed candidate metric drift")
    per_fold, outcomes = _fold_audit(
        target=oof.target,
        pykrige=oof.baseline,
        p19_prediction=p19_prediction,
        candidate=candidate,
        fold_ids=oof.fold_ids,
    )
    bootstrap = p17._whole_fold_bootstrap(  # noqa: SLF001
        target=oof.target,
        baseline=p19_prediction,
        candidate=candidate,
        fold_ids=oof.fold_ids,
    )
    comparison = {
        "pykrige": _metrics(oof.target, oof.baseline),
        "p19": _metrics(oof.target, p19_prediction),
        "candidate": candidate_metrics,
        "rmse_delta_candidate_minus_p19": float(candidate_metrics["rmse"])
        - float(_metrics(oof.target, p19_prediction)["rmse"]),
        "relative_rmse_change_vs_p19": (
            float(candidate_metrics["rmse"])
            - float(_metrics(oof.target, p19_prediction)["rmse"])
        )
        / float(_metrics(oof.target, p19_prediction)["rmse"]),
        "outcomes_vs_p19": outcomes,
        "per_fold": per_fold,
        "whole_fold_bootstrap": bootstrap,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "objective": "replace unstable per-fold meta-selection with a fixed foundation-kernel ensemble",
        "protocol": {
            "outer_spatial_folds": list(base.FOLD_IDS),
            "train_labels_per_fold": 512,
            "validation_rows_per_fold": 2048,
            "primary_metric": "pooled development OOF RMSE",
            "candidate_selection_uses_labels": False,
            "test_h5_opened": False,
            "holdout_opened": False,
            "foundation_random_init_ablation": "deferred_by_user",
        },
        "fixed_candidate_names": names,
        "fixed_parameters": {
            "vertical_weight": VERTICAL_WEIGHT,
            "foundation_weight": FOUNDATION_WEIGHT,
            "seismic_weights": list(SEISMIC_WEIGHTS),
            "neighbours": NEIGHBOURS,
            "distance_power": DISTANCE_POWER,
            "blend_weight": BLEND_WEIGHT,
        },
        "comparison": comparison,
        "rejected_residual_routes": list(REJECTED_RESIDUAL_ROUTES),
        "decision": {
            "default_enabled": True,
            "state": "ACCEPTED_SIMPLICITY_WIN",
            "strict_p19_improvement": bool(candidate_metrics["rmse"] < EXPECTED_P19_RMSE),
            "fold_losses_vs_p19": outcomes["loss"],
            "causal_foundation_contribution_claimed": False,
            "broad_statistical_effect_claimed": False,
        },
        "fold_loading_audit": fold_loading_audit,
        "foundation_feature_audit": feature_audit,
        "transform_audits": transform_audits,
        "inputs": {
            "train_h5_sha256": _sha256(inputs.train_h5),
            "p19_predictions_sha256": _sha256(p19_predictions),
            "feature_cache_sha256": _sha256(p18.DEFAULT_FEATURE_CACHE),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.npz"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            indices_kji=oof.indices_kji,
            fold_ids=oof.fold_ids,
            target=oof.target,
            pykrige_prediction=oof.baseline,
            p19_prediction=p19_prediction,
            candidate_prediction=candidate,
        )
    result["prediction_artifact"] = {
        "path": str(prediction_path.relative_to(PROJECT_ROOT)),
        "sha256": _sha256(prediction_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_evidence(output_dir, result)
    return result


def verify_evidence(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifact = PROJECT_ROOT / summary["prediction_artifact"]["path"]
    if _sha256(artifact) != summary["prediction_artifact"]["sha256"]:
        raise RuntimeError("P21 prediction artifact hash mismatch")
    with np.load(artifact, allow_pickle=False) as payload:
        target = np.asarray(payload["target"], dtype=np.float64)
        p19_prediction = np.asarray(payload["p19_prediction"], dtype=np.float64)
        candidate = np.asarray(payload["candidate_prediction"], dtype=np.float64)
        fold_ids = np.asarray(payload["fold_ids"], dtype=np.int64)
    p19_metrics = _metrics(target, p19_prediction)
    candidate_metrics = _metrics(target, candidate)
    np.testing.assert_allclose(
        p19_metrics["rmse"], summary["comparison"]["p19"]["rmse"], rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        candidate_metrics["rmse"],
        summary["comparison"]["candidate"]["rmse"],
        rtol=0.0,
        atol=1e-15,
    )
    if candidate_metrics["rmse"] >= p19_metrics["rmse"]:
        raise RuntimeError("P21 candidate no longer improves P19")
    losses = 0
    for fold_id in base.FOLD_IDS:
        mask = fold_ids == fold_id
        if _metrics(target[mask], candidate[mask])["rmse"] > _metrics(
            target[mask], p19_prediction[mask]
        )["rmse"] + 1e-12:
            losses += 1
    if losses:
        raise RuntimeError("P21 candidate regressed an outer fold")
    verification = {
        "status": "PASSED",
        "summary_sha256": _sha256(summary_path),
        "prediction_artifact_sha256": _sha256(artifact),
        "p19_metrics_recomputed": p19_metrics,
        "candidate_metrics_recomputed": candidate_metrics,
        "fold_losses_vs_p19": losses,
        "firewall": {
            "test_h5_opened": False,
            "holdout_opened": False,
            "candidate_selection_uses_labels": False,
        },
    }
    (output_dir / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return verification


def write_artifact_manifest(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        output_dir / "evidence.md",
        output_dir / "predictions.npz",
        output_dir / "summary.json",
        output_dir / "verification.json",
    ]
    manifest = {
        "schema_version": "reconstruction-p21-artifact-manifest/v1",
        "artifacts": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        ],
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--stage3-root", type=Path, required=True)
    parser.add_argument("--p19-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for name in ("data_dir", "stage3_root", "p19_predictions", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.verify_only:
        verification = verify_evidence(args.output_dir)
        write_artifact_manifest(args.output_dir)
        print(json.dumps(verification, sort_keys=True))
        return 0
    result = run(
        data_dir=args.data_dir,
        stage3_root=args.stage3_root,
        p19_predictions=args.p19_predictions,
        output_dir=args.output_dir,
    )
    verification = verify_evidence(args.output_dir)
    write_artifact_manifest(args.output_dir)
    print(
        json.dumps(
            {
                "p19_rmse": result["comparison"]["p19"]["rmse"],
                "candidate_rmse": result["comparison"]["candidate"]["rmse"],
                "state": result["decision"]["state"],
                "verification": verification["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
