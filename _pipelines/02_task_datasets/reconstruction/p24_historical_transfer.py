#!/usr/bin/env python3
"""P24 preregistered same-field historical-version transfer test.

This phase evaluates the frozen P21 predictor on an unused historical PHIF
property stored on the same Volve grid.  It is deliberately labelled as a
same-field transfer stress test, not a fresh blind or cross-field test.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
from zipfile import ZipFile

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(PROJECT_ROOT)]

import p11_residual_fusion as base  # noqa: E402
import p17_foundation_geostatistics as p17  # noqa: E402
import p18_anisotropic_foundation_geostatistics as p18  # noqa: E402
import p21_fixed_foundation_ensemble as p21  # noqa: E402


SCHEMA_VERSION = "reconstruction-p24-historical-transfer/v1"
PREREGISTRATION = HERE / "p24_historical_transfer_preregistration.json"
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "p24_historical_transfer"
GRID_SHAPE_KJI = (63, 100, 108)
ECLIPSE_ASCII_PORO_MEMBER = (
    "Reservoir_Model-Eclipse_model/Volve_sim_model_PPA-Eclipse Res Model/PHIF_NW"
)
GEOMATIC_MARKER = b"End GEOMATIC file header"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def _validate_preregistration(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != (
        "reconstruction-p24-historical-transfer-preregistration/v1"
    ):
        raise RuntimeError("P24 preregistration schema drift")
    evidence = payload["evidence_class"]
    if evidence.get("fresh_blind") or evidence.get("cross_field"):
        raise RuntimeError("P24 evidence class may not claim blind/cross-field status")
    lock = payload["evaluation_lock"]
    if lock.get("hyperparameter_search_after_target_open") is not False:
        raise RuntimeError("P24 post-opening tuning prohibition is missing")
    if lock.get("single_opening") is not True:
        raise RuntimeError("P24 one-opening lock is missing")
    fixed = lock["candidate"]
    expected = {
        "vertical_weight": p21.VERTICAL_WEIGHT,
        "foundation_weight": p21.FOUNDATION_WEIGHT,
        "seismic_weights": list(p21.SEISMIC_WEIGHTS),
        "neighbours": p21.NEIGHBOURS,
        "distance_power": p21.DISTANCE_POWER,
        "blend_weight": p21.BLEND_WEIGHT,
    }
    for key, value in expected.items():
        if fixed.get(key) != value:
            raise RuntimeError(f"P24/P21 parameter lock drift: {key}")


def _member_record(zf: ZipFile, member: str) -> dict[str, Any]:
    info = zf.getinfo(member)
    return {
        "path": member,
        "zip_crc32": f"{info.CRC:08x}",
        "compressed_bytes": int(info.compress_size),
        "uncompressed_bytes": int(info.file_size),
    }


def _read_geomatic(
    zf: ZipFile,
    lock: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    member = str(lock["path"])
    record = _member_record(zf, member)
    if record["zip_crc32"] != lock["zip_crc32"]:
        raise RuntimeError(f"GEOMATIC CRC drift: {member}")
    blob = zf.read(member)
    if _bytes_sha256(blob) != lock["member_sha256"]:
        raise RuntimeError(f"GEOMATIC member hash drift: {member}")
    marker = blob.find(GEOMATIC_MARKER)
    if marker < 0:
        raise ValueError(f"GEOMATIC header terminator missing: {member}")
    payload_start = marker + len(GEOMATIC_MARKER)
    while payload_start < len(blob) and blob[payload_start] in b"\r\n":
        payload_start += 1
    raw = blob[payload_start:]
    if _bytes_sha256(raw) != lock["payload_sha256"]:
        raise RuntimeError(f"GEOMATIC payload hash drift: {member}")
    points = int(lock["points"])
    if len(raw) != points * 4:
        raise ValueError(f"GEOMATIC payload size mismatch: {member}")
    values = np.frombuffer(raw, dtype=">f4").astype(np.float32)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(f"GEOMATIC payload is non-finite: {member}")
    return values, {
        **record,
        "member_sha256": lock["member_sha256"],
        "payload_sha256": lock["payload_sha256"],
        "points": points,
        "payload_dtype": ">f4",
    }


def _read_eclipse_ascii_poro(eclipse_zip: Path) -> np.ndarray:
    with ZipFile(eclipse_zip) as zf:
        text = zf.read(ECLIPSE_ASCII_PORO_MEMBER).decode("ascii").strip()
    keyword, body = text.split(None, 1)
    if keyword.upper() != "PORO":
        raise ValueError("Eclipse ASCII property is not PORO")
    values = np.fromstring(body.rsplit("/", 1)[0], sep=" ").astype(np.float32)
    if values.size != int(np.prod(GRID_SHAPE_KJI)):
        raise ValueError("Eclipse ASCII PORO cell count drift")
    return values.reshape(GRID_SHAPE_KJI)


def _historical_kji_volume(
    current_kji: np.ndarray,
    current_rms: np.ndarray,
    historical_rms: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    current_kji = np.asarray(current_kji, dtype=np.float32)
    current_rms = np.asarray(current_rms, dtype=np.float32)
    historical_rms = np.asarray(historical_rms, dtype=np.float32)
    if current_kji.shape != GRID_SHAPE_KJI:
        raise ValueError("current KJI volume shape drift")
    current_ijk = np.transpose(current_kji, (2, 1, 0))
    flat = current_ijk.ravel()
    mask = flat > 0.0
    if int(mask.sum()) != len(current_rms) or len(historical_rms) != len(current_rms):
        raise ValueError("RMS/Eclipse point-count mismatch")
    if not np.array_equal(flat[mask], current_rms):
        raise RuntimeError("final RMS payload does not exactly match Eclipse IJK order")
    historical_ijk = np.zeros_like(flat)
    historical_ijk[mask] = historical_rms
    historical_kji = np.transpose(
        historical_ijk.reshape((GRID_SHAPE_KJI[2], GRID_SHAPE_KJI[1], GRID_SHAPE_KJI[0])),
        (2, 1, 0),
    )
    return historical_kji, {
        "mapping": "Eclipse KJI -> IJK/C-order -> positive mask -> RMS payload",
        "current_reference_exact_elementwise_match": True,
        "mapped_points": int(mask.sum()),
        "grid_shape_kji": list(GRID_SHAPE_KJI),
    }


def _sample_kji(volume: np.ndarray, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("KJI indices must have shape [N,3]")
    values = np.asarray(volume[tuple(indices.T)], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise RuntimeError("historical target is missing at an evaluation KJI cell")
    return values


def _pykrige_class(pykrige_site: Path) -> tuple[Any, str]:
    site = str(Path(pykrige_site).expanduser().resolve())
    if site not in sys.path:
        sys.path.append(site)
    from pykrige.ok3d import OrdinaryKriging3D  # noqa: PLC0415

    version = importlib.metadata.version("pykrige")
    if version != "1.7.3":
        raise RuntimeError(f"P24 requires PyKrige 1.7.3, got {version}")
    return OrdinaryKriging3D, version


def _fold_normalized_features(
    stage3_root: Path,
    fold_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = stage3_root / "cache" / base.EXPECTED_LANE / f"fold_{fold_id:02d}"
    with np.load(root / "point_train.npz", allow_pickle=False) as payload:
        train_features = np.asarray(payload["input_values"], dtype=np.float64)
        train_target = np.asarray(payload["target_values"], dtype=np.float64)
    with np.load(root / "point_validation.npz", allow_pickle=False) as payload:
        validation_features = np.asarray(payload["input_values"], dtype=np.float64)
        validation_target = np.asarray(payload["target_values"], dtype=np.float64)
    return train_features, train_target, validation_features, validation_target


def _fit_pykrige(
    ordinary_kriging_3d: Any,
    train_features: np.ndarray,
    train_target: np.ndarray,
    validation_features: np.ndarray,
) -> np.ndarray:
    xyz = np.asarray(train_features[:, -3:], dtype=np.float64)
    xyz_validation = np.asarray(validation_features[:, -3:], dtype=np.float64)
    model = ordinary_kriging_3d(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        np.asarray(train_target, dtype=np.float64),
        variogram_model="linear",
        nlags=4,
        verbose=False,
        enable_plotting=False,
    )
    prediction, _ = model.execute(
        "points",
        xyz_validation[:, 0],
        xyz_validation[:, 1],
        xyz_validation[:, 2],
    )
    result = np.asarray(prediction, dtype=np.float64)
    if result.shape != (len(validation_features),) or not np.all(np.isfinite(result)):
        raise FloatingPointError("PyKrige produced invalid P24 predictions")
    return result


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "valid_voxels": int(len(error)),
    }


def _common_inputs(
    *,
    data_dir: Path,
    stage3_root: Path,
) -> tuple[base.DevInputPaths, base.OOFDevelopment, tuple[p17.FoldSamples, ...], dict[str, Any]]:
    inputs = base.resolve_dev_inputs(data_dir)
    oof = base.load_oof_development(stage3_root)
    folds, fold_audit = p17.load_fold_samples(
        stage3_root=stage3_root,
        train_h5=inputs.train_h5,
        oof=oof,
    )
    return inputs, oof, folds, fold_audit


def preflight(
    *,
    data_dir: Path,
    stage3_root: Path,
    rms_zip: Path,
    eclipse_zip: Path,
    pykrige_site: Path,
) -> dict[str, Any]:
    prereg = _json(PREREGISTRATION)
    _validate_preregistration(prereg)
    source = prereg["source_lock"]
    if _sha256(rms_zip) != source["rms_zip"]["sha256"]:
        raise RuntimeError("RMS archive hash drift")
    if _sha256(eclipse_zip) != source["eclipse_zip"]["sha256"]:
        raise RuntimeError("Eclipse archive hash drift")
    with ZipFile(rms_zip) as zf:
        current, current_record = _read_geomatic(zf, source["final_reference_member"])
        historical_meta = _member_record(zf, source["historical_target_member"]["path"])
    lock = source["historical_target_member"]
    if historical_meta["zip_crc32"] != lock["zip_crc32"]:
        raise RuntimeError("historical target member metadata drift")
    current_kji = _read_eclipse_ascii_poro(eclipse_zip)
    current_ijk = np.transpose(current_kji, (2, 1, 0)).ravel()
    if not np.array_equal(current_ijk[current_ijk > 0.0], current):
        raise RuntimeError("frozen RMS-to-Eclipse mapping preflight failed")

    inputs, oof, folds, fold_audit = _common_inputs(
        data_dir=data_dir,
        stage3_root=stage3_root,
    )
    ordinary_kriging_3d, version = _pykrige_class(pykrige_site)
    fold_rows: list[dict[str, Any]] = []
    maximum_delta = 0.0
    for fold in folds:
        train_x, train_y, validation_x, validation_y = _fold_normalized_features(
            stage3_root, fold.fold_id
        )
        np.testing.assert_allclose(train_y, fold.train_target, rtol=0.0, atol=0.0)
        mask = oof.fold_ids == fold.fold_id
        np.testing.assert_allclose(validation_y, oof.target[mask], rtol=0.0, atol=1e-7)
        prediction = _fit_pykrige(
            ordinary_kriging_3d, train_x, train_y, validation_x
        )
        delta = float(np.max(np.abs(prediction - oof.baseline[mask])))
        maximum_delta = max(maximum_delta, delta)
        fold_rows.append({"fold_id": fold.fold_id, "maximum_absolute_delta": delta})
    tolerance = float(
        prereg["evaluation_lock"]["baseline_reproduction_gate"][
            "maximum_absolute_prediction_delta"
        ]
    )
    if maximum_delta > tolerance:
        raise RuntimeError("PyKrige reproduction exceeds the preregistered tolerance")
    requested = p17._unique_indices(folds)  # noqa: SLF001
    _, _, feature_audit = p18._load_feature_cache(  # noqa: SLF001
        p18.DEFAULT_FEATURE_CACHE,
        train_h5=inputs.train_h5,
        expected_indices=requested,
    )
    return {
        "status": "PASSED",
        "historical_target_values_opened": False,
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "pykrige_version": version,
        "baseline_reproduction_maximum_absolute_delta": maximum_delta,
        "baseline_reproduction_tolerance": tolerance,
        "folds": fold_rows,
        "fold_loading_audit": fold_audit,
        "feature_loading_audit": feature_audit,
        "final_reference_member": current_record,
        "historical_target_member_metadata": historical_meta,
    }


def _comparison(
    target: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    fold_ids: np.ndarray,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_metrics = _metrics(target, baseline_prediction)
    candidate_metrics = _metrics(target, candidate_prediction)
    improvement = (
        float(baseline_metrics["rmse"]) - float(candidate_metrics["rmse"])
    ) / float(baseline_metrics["rmse"])
    rows: list[dict[str, Any]] = []
    outcomes = {"win": 0, "loss": 0, "tie": 0}
    for fold_id in base.FOLD_IDS:
        mask = fold_ids == fold_id
        baseline_fold = _metrics(target[mask], baseline_prediction[mask])
        candidate_fold = _metrics(target[mask], candidate_prediction[mask])
        delta = float(candidate_fold["rmse"]) - float(baseline_fold["rmse"])
        outcome = "win" if delta < -1e-12 else "loss" if delta > 1e-12 else "tie"
        outcomes[outcome] += 1
        rows.append(
            {
                "fold_id": fold_id,
                "pykrige": baseline_fold,
                "candidate": candidate_fold,
                "rmse_delta_candidate_minus_pykrige": delta,
                "outcome": outcome,
            }
        )
    passed = (
        improvement >= float(gate["minimum_relative_rmse_improvement_vs_pykrige"])
        and outcomes["loss"] <= int(gate["maximum_fold_losses_vs_pykrige"])
    )
    return {
        "pykrige": baseline_metrics,
        "candidate": candidate_metrics,
        "relative_rmse_improvement_vs_pykrige": improvement,
        "rmse_delta_candidate_minus_pykrige": float(candidate_metrics["rmse"])
        - float(baseline_metrics["rmse"]),
        "per_fold": rows,
        "outcomes_vs_pykrige": outcomes,
        "success_gate": dict(gate),
        "gate_passed": passed,
    }


def run(
    *,
    data_dir: Path,
    stage3_root: Path,
    rms_zip: Path,
    eclipse_zip: Path,
    pykrige_site: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if (output_dir / "opening_record.json").exists() or (output_dir / "summary.json").exists():
        raise RuntimeError("P24 target was already opened; use --verify-only")
    audit = preflight(
        data_dir=data_dir,
        stage3_root=stage3_root,
        rms_zip=rms_zip,
        eclipse_zip=eclipse_zip,
        pykrige_site=pykrige_site,
    )
    prereg = _json(PREREGISTRATION)
    source = prereg["source_lock"]
    prereg_commit = _git("log", "-1", "--format=%H", "--", str(PREREGISTRATION.relative_to(PROJECT_ROOT)))
    code_commit = _git("rev-parse", "HEAD")
    if not prereg_commit or not code_commit:
        raise RuntimeError("P24 git provenance is unavailable")
    output_dir.mkdir(parents=True, exist_ok=False)
    opening = {
        "schema_version": "reconstruction-p24-opening-record/v1",
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "preregistration_git_commit": prereg_commit,
        "execution_code_git_commit": code_commit,
        "historical_target_member_sha256": source["historical_target_member"]["member_sha256"],
        "hyperparameter_search_after_opening": False,
    }
    (output_dir / "opening_record.json").write_text(
        json.dumps(opening, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    current_kji = _read_eclipse_ascii_poro(eclipse_zip)
    with ZipFile(rms_zip) as zf:
        current_rms, current_record = _read_geomatic(zf, source["final_reference_member"])
        historical_rms, historical_record = _read_geomatic(
            zf, source["historical_target_member"]
        )
    historical_kji, mapping_audit = _historical_kji_volume(
        current_kji, current_rms, historical_rms
    )
    inputs, oof, folds, fold_audit = _common_inputs(
        data_dir=data_dir,
        stage3_root=stage3_root,
    )
    ordinary_kriging_3d, version = _pykrige_class(pykrige_site)
    historical_folds: list[p17.FoldSamples] = []
    baseline = np.full(len(oof.target), np.nan, dtype=np.float64)
    for fold in folds:
        historical_train = _sample_kji(historical_kji, fold.train_indices_kji)
        historical_validation = _sample_kji(
            historical_kji, fold.validation_indices_kji
        )
        train_x, _, validation_x, _ = _fold_normalized_features(stage3_root, fold.fold_id)
        fold_prediction = _fit_pykrige(
            ordinary_kriging_3d,
            train_x,
            historical_train,
            validation_x,
        )
        mask = oof.fold_ids == fold.fold_id
        np.testing.assert_array_equal(oof.indices_kji[mask], fold.validation_indices_kji)
        np.testing.assert_allclose(
            historical_validation,
            _sample_kji(historical_kji, oof.indices_kji[mask]),
            rtol=0.0,
            atol=0.0,
        )
        baseline[mask] = fold_prediction
        historical_folds.append(replace(fold, train_target=historical_train))
    if not np.all(np.isfinite(baseline)):
        raise RuntimeError("P24 PyKrige OOF baseline is incomplete")
    target = _sample_kji(historical_kji, oof.indices_kji)
    historical_oof = replace(oof, target=target, baseline=baseline)
    requested = p17._unique_indices(historical_folds)  # noqa: SLF001
    cache_indices, foundation, feature_audit = p18._load_feature_cache(  # noqa: SLF001
        p18.DEFAULT_FEATURE_CACHE,
        train_h5=inputs.train_h5,
        expected_indices=requested,
    )
    candidate, candidate_names, transform_audits = p21._fixed_candidate(  # noqa: SLF001
        oof=historical_oof,
        folds=historical_folds,
        requested_indices=cache_indices,
        foundation_features=foundation,
    )
    comparison = _comparison(
        target,
        baseline,
        candidate,
        oof.fold_ids,
        prereg["evaluation_lock"]["success_gate"],
    )
    bootstrap = p17._whole_fold_bootstrap(  # noqa: SLF001
        target=target,
        baseline=baseline,
        candidate=candidate,
        fold_ids=oof.fold_ids,
    )
    prediction_path = output_dir / "predictions.npz"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            indices_kji=oof.indices_kji,
            fold_ids=oof.fold_ids,
            target=target,
            pykrige_prediction=baseline,
            candidate_prediction=candidate,
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "objective": prereg["objective"],
        "evidence_class": prereg["evidence_class"],
        "opening_record": opening,
        "preflight": audit,
        "mapping_audit": mapping_audit,
        "source_records": {
            "current_reference": current_record,
            "historical_target": historical_record,
            "rms_zip_sha256": source["rms_zip"]["sha256"],
            "eclipse_zip_sha256": source["eclipse_zip"]["sha256"],
        },
        "protocol": prereg["evaluation_lock"],
        "pykrige_version": version,
        "candidate_names": candidate_names,
        "fold_loading_audit": fold_audit,
        "feature_loading_audit": feature_audit,
        "transform_audits": transform_audits,
        "comparison": {**comparison, "whole_fold_bootstrap": bootstrap},
        "decision": {
            "state": "same_field_transfer_supported"
            if comparison["gate_passed"]
            else "development_only_transfer_gate_failed",
            "gate_passed": comparison["gate_passed"],
            "claim_boundary": prereg["evidence_class"]["allowed_claim"],
            "external_blind_validation_still_required": True,
            "post_opening_tuning_performed": False,
        },
        "prediction_artifact": {
            "path": str(prediction_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(prediction_path),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence_lines = [
        "# P24 historical-version transfer result",
        "",
        f"- Evidence class: `{result['evidence_class']['name']}` (not fresh blind).",
        f"- PyKrige RMSE: `{comparison['pykrige']['rmse']:.12f}`.",
        f"- Frozen P21 RMSE: `{comparison['candidate']['rmse']:.12f}`.",
        f"- Relative RMSE improvement: `{comparison['relative_rmse_improvement_vs_pykrige']:.6%}`.",
        f"- Fold outcomes: `{comparison['outcomes_vs_pykrige']}`.",
        f"- Preregistered gate passed: `{comparison['gate_passed']}`.",
        "",
        "No P21 hyperparameter or mapping rule was changed after opening the historical target.",
        "A genuinely external field or competition-hidden test is still required for a blind claim.",
    ]
    (output_dir / "evidence.md").write_text(
        "\n".join(evidence_lines) + "\n", encoding="utf-8"
    )
    return result


def verify_evidence(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    summary = _json(summary_path)
    prediction_path = PROJECT_ROOT / summary["prediction_artifact"]["path"]
    if _sha256(prediction_path) != summary["prediction_artifact"]["sha256"]:
        raise RuntimeError("P24 prediction hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as payload:
        target = np.asarray(payload["target"], dtype=np.float64)
        baseline = np.asarray(payload["pykrige_prediction"], dtype=np.float64)
        candidate = np.asarray(payload["candidate_prediction"], dtype=np.float64)
        fold_ids = np.asarray(payload["fold_ids"], dtype=np.int64)
    recomputed = _comparison(
        target,
        baseline,
        candidate,
        fold_ids,
        summary["protocol"]["success_gate"],
    )
    np.testing.assert_allclose(
        recomputed["pykrige"]["rmse"],
        summary["comparison"]["pykrige"]["rmse"],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        recomputed["candidate"]["rmse"],
        summary["comparison"]["candidate"]["rmse"],
        rtol=0.0,
        atol=1e-15,
    )
    if recomputed["gate_passed"] != summary["decision"]["gate_passed"]:
        raise RuntimeError("P24 decision gate drift")
    verification = {
        "status": "PASSED",
        "summary_sha256": _sha256(summary_path),
        "prediction_artifact_sha256": _sha256(prediction_path),
        "comparison_recomputed": recomputed,
        "claim_boundary_preserved": (
            summary["evidence_class"]["fresh_blind"] is False
            and summary["evidence_class"]["cross_field"] is False
        ),
    }
    (output_dir / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return verification


def write_artifact_manifest(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        PREREGISTRATION,
        output_dir / "opening_record.json",
        output_dir / "predictions.npz",
        output_dir / "summary.json",
        output_dir / "evidence.md",
        output_dir / "verification.json",
    ]
    manifest = {
        "schema_version": "reconstruction-p24-artifact-manifest/v1",
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
    parser.add_argument("--rms-zip", type=Path, required=True)
    parser.add_argument("--eclipse-zip", type=Path, required=True)
    parser.add_argument("--pykrige-site", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for name in (
        "data_dir",
        "stage3_root",
        "rms_zip",
        "eclipse_zip",
        "pykrige_site",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.verify_only:
        verification = verify_evidence(args.output_dir)
        write_artifact_manifest(args.output_dir)
        print(json.dumps(verification, sort_keys=True))
        return 0
    if args.preflight:
        result = preflight(
            data_dir=args.data_dir,
            stage3_root=args.stage3_root,
            rms_zip=args.rms_zip,
            eclipse_zip=args.eclipse_zip,
            pykrige_site=args.pykrige_site,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    result = run(
        data_dir=args.data_dir,
        stage3_root=args.stage3_root,
        rms_zip=args.rms_zip,
        eclipse_zip=args.eclipse_zip,
        pykrige_site=args.pykrige_site,
        output_dir=args.output_dir,
    )
    verification = verify_evidence(args.output_dir)
    write_artifact_manifest(args.output_dir)
    print(
        json.dumps(
            {
                "state": result["decision"]["state"],
                "gate_passed": result["decision"]["gate_passed"],
                "pykrige_rmse": result["comparison"]["pykrige"]["rmse"],
                "candidate_rmse": result["comparison"]["candidate"]["rmse"],
                "verification": verification["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
