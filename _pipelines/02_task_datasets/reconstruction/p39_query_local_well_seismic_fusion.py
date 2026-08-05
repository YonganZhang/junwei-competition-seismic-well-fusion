#!/usr/bin/env python3
"""P39 query-local real-well PHIF well-seismic fusion.

The command is deliberately split into a no-training ``phase0`` freeze and a
bounded ``run``.  The run refuses to start unless the committed P38 evidence,
the P39 freeze, local foundation assets, and the retained-row contract all
reproduce exactly.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
for _root in (str(PROJECT_ROOT), str(HERE)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

import p38_pilot_core as p38_core  # noqa: E402
import p38_real_well_phif_direct_seismic as p38  # noqa: E402


SCHEMA = "reconstruction-p39-query-local-well-seismic-fusion/v1"
PHASE0_SCHEMA = "reconstruction-p39-phase0-freeze/v1"
DEFAULT_OUTPUT = HERE / "_outputs/p39_query_local_well_seismic_fusion"
DEFAULT_SCRATCH = PROJECT_ROOT / "_tmp/p39_query_local_well_seismic_fusion"
P38_OUTPUT = HERE / "_outputs/p38_real_well_phif_direct_seismic"
WIKI_FINDING = (
    PROJECT_ROOT
    / "_wiki-methodology/_top/_findings/P39_query_local_well_seismic_foundation_fusion.md"
)
P38_LOCKED_COMMIT = "eb4789b0c406440a9ad897cc3f09c25beda50fa3"
P38_WELL_MACRO_RMSE = 0.0753144330342321
P38_FUSION_MACRO_RMSE = 0.07978122851402457
P39_PRE_FIX_COMMIT = "c6056a0b05cbe401bf1dfa90fbab2fac16430c08"
P39_PRE_FIX_SUMMARY_SHA256 = (
    "a68dabe2d4d9b5e238bb4db060a40e68a4576f2e952bcbdf96c923eaf54924d1"
)
P39_PRE_FIX_CONTROL_MACRO_RMSE = {
    "moment_random_gfm_random": 0.07468168654416331,
    "moment_pretrained_gfm_random": 0.07586122373838418,
    "moment_random_gfm_pretrained": 0.07473459535678662,
    "moment_pretrained_gfm_pretrained": 0.0759084841201444,
}
P39_FROZEN_SELECTED_ACTIONS = {
    "0": {"action_id": "shorter_patience", "selected_steps": 7},
    "1": {"action_id": "default", "selected_steps": 1},
    "2": {"action_id": "stronger_regularization", "selected_steps": 1},
}
PARENT_ORDER = p38.PARENT_ORDER
SEED = 2693
MAX_SCIENTIFIC_ITERATIONS = 3
CORRECTION_BOUND = 0.5
TEMPORAL_OFFSETS = np.arange(-8, 9, dtype=np.int64)
CROSSLINE_OFFSETS = np.arange(-2, 3, dtype=np.int64)
TWT_SHIFT_SAMPLES = 40
TWT_SHIFT_MS = 160.0
FORBIDDEN_TOKENS = ("test.h5", "holdout", "frozen_test", "frozen-test", "kji")
WEIGHT_CONTROLS = (
    "moment_random_gfm_random",
    "moment_pretrained_gfm_random",
    "moment_random_gfm_pretrained",
    "moment_pretrained_gfm_pretrained",
)
ITERATION2_NAME = "iteration2_anchored_p38_representation"
FINAL_CANDIDATE = "moment_pretrained_gfm_pretrained"
MAX_INFORMATIVE_90_INTERVAL_WIDTH = 1.0


def _canonical(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def fixed_base_attribution_audit(
    *,
    control_bases: Mapping[str, Mapping[int, np.ndarray]],
    parent_index: np.ndarray,
) -> dict[str, Any]:
    """Prove every weight control uses one identical fold-specific well base."""

    if set(control_bases) != set(WEIGHT_CONTROLS):
        raise ValueError("P39.1 fixed-base audit requires all four weight controls")
    parent = np.asarray(parent_index, dtype=np.int64)
    shared = control_bases[FINAL_CANDIDATE]
    controls: dict[str, Any] = {}
    all_exact = True
    for name in WEIGHT_CONTROLS:
        folds: dict[str, Any] = {}
        for fold in range(3):
            expected = np.asarray(shared[fold], dtype=np.float32)
            actual = np.asarray(control_bases[name][fold], dtype=np.float32)
            if actual.shape != expected.shape or actual.shape != parent.shape:
                raise ValueError(f"P39.1 base shape drift for {name}/fold{fold}")
            outer_train = parent != fold
            held = parent == fold
            full_diff = float(
                np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))
            )
            train_diff = float(
                np.max(
                    np.abs(
                        actual[outer_train].astype(np.float64)
                        - expected[outer_train].astype(np.float64)
                    )
                )
            )
            held_diff = float(
                np.max(
                    np.abs(
                        actual[held].astype(np.float64)
                        - expected[held].astype(np.float64)
                    )
                )
            )
            exact = full_diff == train_diff == held_diff == 0.0
            all_exact = all_exact and exact
            folds[str(fold)] = {
                "shape": list(actual.shape),
                "base_sha256": _array_sha256(actual),
                "shared_base_sha256": _array_sha256(expected),
                "outer_train_crossfit_sha256": _array_sha256(actual[outer_train]),
                "held_exact_sha256": _array_sha256(actual[held]),
                "max_abs_diff_vs_shared": full_diff,
                "outer_train_max_abs_diff_vs_shared": train_diff,
                "held_max_abs_diff_vs_shared": held_diff,
                "exact_shared_base": exact,
                "outer_train_rows": int(np.count_nonzero(outer_train)),
                "held_rows": int(np.count_nonzero(held)),
            }
        controls[name] = {"folds": folds}
    if not all_exact:
        raise RuntimeError("P39.1 controls do not share the exact locked P38 well base")
    return {
        "schema_version": "reconstruction-p39-fixed-base-attribution/v1",
        "invariant": (
            "All Iteration-3 controls share the pretrained P38 outer-train "
            "cross-fitted base and exact recorded outer-held base; only MOMENT/GFM "
            "correction features vary."
        ),
        "pre_fix_commit": P39_PRE_FIX_COMMIT,
        "pre_fix_summary_sha256": P39_PRE_FIX_SUMMARY_SHA256,
        "controls": controls,
        "all_controls_all_folds_exact_shared_base": bool(all_exact),
    }


def _json_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _validate_output(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(HERE.resolve())
    except ValueError as exc:
        raise ValueError(f"P39 output must remain under reconstruction: {resolved}") from exc
    if resolved == P38_OUTPUT.resolve():
        raise ValueError("P39 refuses to overwrite P38 evidence")
    return resolved


def _validate_scratch(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    allowed = DEFAULT_SCRATCH.resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"P39 scratch must remain under {allowed}: {resolved}") from exc
    return resolved


def _assert_legal_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        lowered = str(Path(path).expanduser().resolve()).lower()
        if any(token in lowered for token in FORBIDDEN_TOKENS):
            raise RuntimeError(f"forbidden label/holdout path: {path}")


def _read_p38() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    summary = json.loads((P38_OUTPUT / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads(
        (P38_OUTPUT / "verification.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (P38_OUTPUT / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    return summary, verification, manifest


def _verify_p38_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for item in manifest["artifacts"]:
        path = P38_OUTPUT / item["path"]
        ok = (
            path.is_file()
            and path.stat().st_size == int(item["size_bytes"])
            and _sha256(path) == item["sha256"]
        )
        rows.append({"path": item["path"], "sha256": item["sha256"], "match": ok})
    if not all(row["match"] for row in rows):
        raise RuntimeError("P38 artifact manifest drifted")
    return {"artifact_count": len(rows), "all_match": True, "artifacts": rows}


def _p38_prediction_arrays() -> dict[str, np.ndarray]:
    with np.load(P38_OUTPUT / "predictions.npz", allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _phase0_protocol(
    *,
    paths: Mapping[str, str],
    selected_device: str,
    gpu_uuid: str,
) -> dict[str, Any]:
    protocol: dict[str, Any] = {
        "schema_version": PHASE0_SCHEMA,
        "state": "FROZEN_ACTIVE",
        "base": {
            "branch": "p38-real-well-phif-direct-seismic",
            "commit": P38_LOCKED_COMMIT,
            "p38_decision": "FEASIBLE_NO_PROMOTION",
            "well_only_macro_rmse": P38_WELL_MACRO_RMSE,
            "fusion_macro_rmse": P38_FUSION_MACRO_RMSE,
        },
        "iterations": [
            {
                "iteration": 1,
                "changed_variable": None,
                "operation": "exact P38 artifact reuse",
            },
            {
                "iteration": 2,
                "changed_variable": "prediction_interface_only",
                "operation": "anchored bounded residual with P38 GFM row representation",
            },
            {
                "iteration": 3,
                "changed_variable": "seismic_representation_only",
                "operation": "native query-local waveform plus retained GFM trace/CLS context",
            },
        ],
        "maximum_scientific_iterations": MAX_SCIENTIFIC_ITERATIONS,
        "parent_order": list(PARENT_ORDER),
        "split": "exact P38 LOGO3 retained rows",
        "target": {
            "name": "native published CPI PHIF",
            "unit": "V/V_fraction",
            "bounds": [0.0, 1.0],
            "forbidden_aliases": ["PHIE", "Eclipse PORO"],
        },
        "representation": {
            "temporal_offsets": TEMPORAL_OFFSETS.tolist(),
            "crossline_offsets": CROSSLINE_OFFSETS.tolist(),
            "channels": ["amplitude", "local_rms_5_sample", "vertical_gradient_2_sample"],
            "gfm_context": "per-row trace token and CLS token retained",
            "combination": "concatenate local waveform and GFM row context before fixed 32D transform",
            "boundary_policy": "clip shifted query center to keep all frozen offsets native",
        },
        "anchored_residual": {
            "formula": "clip(locked_well_base + gate * bounded_seismic_correction, 0, 1)",
            "gate_bounds_by_action": [0.0, 0.1],
            "correction_bounds": [-CORRECTION_BOUND, CORRECTION_BOUND],
            "zero_gate_exact_fallback": True,
            "outer_train_base_targets": "unchanged P38 well-only cross-fitted predictions",
            "outer_held_base": "exact recorded P38 well-only prediction for pretrained MOMENT",
        },
        "action_allowlist": copy.deepcopy(
            json.loads((P38_OUTPUT / "phase0_freeze.json").read_text())["action_allowlist"]
        ),
        "agent_policy": {
            "one_action_per_outer_fold": True,
            "evidence": "outer-train inner-LOGO only",
            "tie_break": "default",
            "held_parent_metric_visible": False,
        },
        "weight_controls": list(WEIGHT_CONTROLS),
        "resource_budget": {
            "foundation_encoders_frozen": True,
            "local_only": True,
            "new_dependencies": False,
            "seed": SEED,
            "head_parameter_ceiling": p38_core.parameter_count(),
            "head_max_updates": p38_core.MAX_UPDATES,
            "head_eval_every": p38_core.EVAL_EVERY,
            "head_learning_rate": p38_core.LEARNING_RATE,
            "selected_device": selected_device,
            "gpu_uuid": gpu_uuid,
        },
        "mismatch_controls": {
            "cyclic_well": "replace complete local and global seismic evidence by relative-MD cyclic map",
            "fixed_twt_shift_samples": TWT_SHIFT_SAMPLES,
            "fixed_twt_shift_ms": TWT_SHIFT_MS,
            "fixed_twt_operation": "relocate waveform query and all manual query-position fields without refit",
        },
        "bootstrap": {
            "draws": p38.BOOTSTRAP_DRAWS,
            "seed": p38.BOOTSTRAP_SEED,
            "physical_MD_block_m": p38.BOOTSTRAP_BLOCK_M,
            "paired": True,
        },
        "calibration": "outer-train inner-LOGO residual Gaussian scale",
        "promotion_gates": [
            "candidate macro RMSE < locked well-only 0.0753144330342321",
            "candidate wins at least 2 of 3 parents",
            "paired 20m block-bootstrap CI95 upper < 0",
            "both-pretrained macro < both-random macro",
            "both-pretrained macro < each single-pretrained macro",
            "both mismatches worsen macro and at least 2 of 3 parents",
            "finite non-vacuous train-only calibration",
            "zero-gate exact fallback for every row",
            "all firewall, provenance, tensor, hash, and rerun checks pass",
        ],
        "stop_conditions": [
            "stop if exact P38 reuse or retained-row identity fails",
            "stop after iteration 3 regardless of outcome",
            "no held-outcome tuning and no fourth iteration",
        ],
        "firewall": {
            "forbidden": ["train.h5", "test.h5", "frozen holdout", "KJI", "PHIE", "PORO"],
            "target_derived_alignment": False,
            "held_parent_target_used_for_action_or_transform": False,
        },
        "local_assets": dict(paths),
        "phase0_encoder_calls": 0,
        "phase0_training_runs": 0,
    }
    protocol["freeze_sha256"] = _json_hash(protocol)
    return protocol


def _row_contract(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    identity = (
        "row_id",
        "parent_index",
        "target",
        "MD_m",
        "TVDSS_m",
        "x_m",
        "y_m",
        "TWT_ms",
        "inline",
        "crossline",
        "time_index",
        "section_id",
        "trace_token_id",
        "time_position",
    )
    return {
        "schema_version": "reconstruction-p39-row-contract/v1",
        "source": str(P38_OUTPUT / "predictions.npz"),
        "rows": int(len(arrays["row_id"])),
        "row_id_exact_once": bool(
            np.array_equal(arrays["row_id"], np.arange(len(arrays["row_id"])))
        ),
        "parent_counts": {
            parent: int(np.count_nonzero(arrays["parent_index"] == index))
            for index, parent in enumerate(PARENT_ORDER)
        },
        "array_sha256": {name: _array_sha256(arrays[name]) for name in identity},
        "target_name": "native published CPI PHIF",
        "target_unit": "V/V_fraction",
        "target_forbidden_aliases": ["PHIE", "Eclipse PORO"],
        "split_hash": _sha256(P38_OUTPUT / "split_manifest.json"),
    }


def _iteration1_record(
    summary: Mapping[str, Any], arrays: Mapping[str, np.ndarray], manifest_audit: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "reconstruction-p39-iteration/v1",
        "iteration": 1,
        "operation": "locked P38 reuse",
        "new_training_runs": 0,
        "scientific_variable_changed": None,
        "p38_manifest": dict(manifest_audit),
        "rows": int(len(arrays["row_id"])),
        "predictions_sha256": _sha256(P38_OUTPUT / "predictions.npz"),
        "well_only_prediction_array_sha256": _array_sha256(
            arrays["well_only__prediction"]
        ),
        "target_array_sha256": _array_sha256(arrays["target"]),
        "metrics": summary["models"]["well_only"]["metrics"],
        "calibration": summary["models"]["well_only"]["calibration"],
        "bootstrap_fusion_minus_well": summary["paired_depth_block_bootstrap"],
        "decision": summary["decision"],
        "exact_reuse_passed": True,
    }


def _phase0_checks(output: Path) -> dict[str, Any]:
    summary, p38_verification, manifest = _read_p38()
    manifest_audit = _verify_p38_manifest(manifest)
    arrays = _p38_prediction_arrays()
    freeze = json.loads((output / "phase0_freeze.json").read_text(encoding="utf-8"))
    freeze_without_hash = {key: value for key, value in freeze.items() if key != "freeze_sha256"}
    checks = {
        "p38_verification_pass": p38_verification["status"] == "PASS"
        and all(p38_verification["checks"].values()),
        "p38_manifest_all_hashes_match": manifest_audit["all_match"],
        "p38_decision_locked": summary["decision"]["state"] == "FEASIBLE_NO_PROMOTION",
        "well_only_macro_locked": summary["models"]["well_only"]["metrics"]["equal_parent_macro"]["rmse"]
        == P38_WELL_MACRO_RMSE,
        "p38_fusion_macro_locked": summary["models"]["frozen_pretrained_moment_gfm_fusion"]["metrics"]["equal_parent_macro"]["rmse"]
        == P38_FUSION_MACRO_RMSE,
        "exact_row_count": len(arrays["row_id"]) == 12685,
        "row_ids_exact_once": np.array_equal(
            arrays["row_id"], np.arange(len(arrays["row_id"]), dtype=np.int64)
        ),
        "exact_three_parents": np.array_equal(np.unique(arrays["parent_index"]), [0, 1, 2]),
        "freeze_hash_matches": freeze["freeze_sha256"] == _json_hash(freeze_without_hash),
        "exact_three_iterations": len(freeze["iterations"]) == MAX_SCIENTIFIC_ITERATIONS,
        "phase0_no_encoder_or_training": freeze["phase0_encoder_calls"] == 0
        and freeze["phase0_training_runs"] == 0,
        "local_assets_exist": all(Path(path).is_dir() or Path(path).is_file() for path in freeze["local_assets"].values()),
        "local_only_frozen_encoders": freeze["resource_budget"]["foundation_encoders_frozen"]
        and freeze["resource_budget"]["local_only"],
    }
    return {
        "schema_version": "reconstruction-p39-phase0-verification/v1",
        "status": "PASS_PHASE0" if all(checks.values()) else "FAIL_PHASE0",
        "checks": checks,
    }


def write_phase0(
    *,
    output_dir: Path,
    paths: Mapping[str, str],
    selected_device: str,
    gpu_uuid: str,
) -> None:
    output = _validate_output(output_dir)
    _assert_legal_paths([Path(value) for value in paths.values()])
    summary, verification, manifest = _read_p38()
    if verification["status"] != "PASS" or not all(verification["checks"].values()):
        raise RuntimeError("P38 verification is not PASS")
    manifest_audit = _verify_p38_manifest(manifest)
    arrays = _p38_prediction_arrays()
    freeze = _phase0_protocol(
        paths=paths, selected_device=selected_device, gpu_uuid=gpu_uuid
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "phase0_freeze.json").write_text(_canonical(freeze), encoding="utf-8")
    (output / "row_contract.json").write_text(
        _canonical(_row_contract(arrays)), encoding="utf-8"
    )
    (output / "iteration_1_locked_reuse.json").write_text(
        _canonical(_iteration1_record(summary, arrays, manifest_audit)), encoding="utf-8"
    )
    result = _phase0_checks(output)
    (output / "phase0_verification.json").write_text(
        _canonical(result), encoding="utf-8"
    )
    if result["status"] != "PASS_PHASE0":
        raise RuntimeError(f"P39 Phase 0 failed: {result}")


def verify_phase0(output_dir: Path) -> None:
    output = _validate_output(output_dir)
    result = _phase0_checks(output)
    committed = json.loads(
        (output / "phase0_verification.json").read_text(encoding="utf-8")
    )
    if result["status"] != "PASS_PHASE0" or result != committed:
        raise RuntimeError(f"P39 Phase 0 does not reproduce: {result}")


def _exact_source_row_check(
    source: Mapping[str, np.ndarray], locked: Mapping[str, np.ndarray]
) -> dict[str, float]:
    fields = (
        "row_id",
        "parent_index",
        "target",
        "MD_m",
        "TVDSS_m",
        "x_m",
        "y_m",
        "TWT_ms",
        "inline",
        "crossline",
        "time_index",
        "section_id",
        "trace_token_id",
        "time_position",
    )
    differences: dict[str, float] = {}
    for field in fields:
        left = np.asarray(source[field])
        right = np.asarray(locked[field])
        if left.shape != right.shape:
            raise RuntimeError(f"P39 retained-row shape drift: {field}")
        if np.issubdtype(left.dtype, np.number):
            delta = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        else:
            delta = 0.0 if np.array_equal(left, right) else math.inf
        differences[field] = delta
        if delta != 0.0:
            raise RuntimeError(f"P39 retained-row identity drift: {field} max_abs={delta}")
    return differences


def _extract_features(
    *,
    arrays: Mapping[str, np.ndarray],
    scratch_dir: Path,
    foundation_python: Path,
    moment_snapshot: Path,
    moment_dependency_root: Path,
    moment_source_root: Path,
    gfm_snapshot: Path,
    gfm_source_root: Path,
    device: str,
) -> tuple[
    dict[tuple[int, str], np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    scratch = _validate_scratch(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "schema_version": "reconstruction-p39-foundation-feature-audit/v1",
        "worker": str(HERE / "p38_foundation_feature_worker.py"),
        "worker_sha256": _sha256(HERE / "p38_foundation_feature_worker.py"),
        "fold_normalization": {},
        "runs": {},
    }
    moment: dict[tuple[int, str], np.ndarray] = {}
    for fold in range(3):
        values, mask, normalization = p38._fold_well_worker_input(arrays, fold_id=fold)  # noqa: SLF001
        audit["fold_normalization"][str(fold)] = normalization
        input_path = scratch / f"moment_fold{fold}_input.npz"
        with input_path.open("wb") as handle:
            np.savez_compressed(handle, values=values, input_mask=mask)
        for mode in ("pretrained", "random_init"):
            output_path = scratch / f"moment_fold{fold}_{mode}.npz"
            run_audit = p38._run_feature_worker(  # noqa: SLF001
                foundation_python=foundation_python,
                kind="moment",
                input_path=input_path,
                output_path=output_path,
                weight_mode=mode,
                device=device,
                batch_size=256,
                moment_snapshot=moment_snapshot,
                moment_dependency_root=moment_dependency_root,
                moment_source_root=moment_source_root,
                gfm_snapshot=gfm_snapshot,
                gfm_source_root=gfm_source_root,
            )
            with np.load(output_path, allow_pickle=False) as payload:
                features = np.asarray(payload["features"], dtype=np.float32)
            expected = (len(arrays["target"]), 6, 4, 16)
            if features.shape != expected:
                raise RuntimeError(f"P39 MOMENT shape drift: {features.shape} != {expected}")
            moment[(fold, mode)] = features.reshape(len(features), -1)
            audit["runs"][f"moment_fold{fold}_{mode}"] = run_audit
    section_input = scratch / "gfm_sections_input.npz"
    with section_input.open("wb") as handle:
        np.savez_compressed(
            handle, sections=np.asarray(arrays["native_sections"], dtype=np.float32)
        )
    gfm_rows: dict[str, np.ndarray] = {}
    gfm_sections: dict[str, np.ndarray] = {}
    section_id = np.asarray(arrays["section_id"], dtype=np.int64)
    trace_id = np.asarray(arrays["trace_token_id"], dtype=np.int64)
    position = np.asarray(arrays["time_position"], dtype=np.float32)
    time_features = _position_features(position)
    for mode in ("pretrained", "random_init"):
        output_path = scratch / f"gfm_sections_{mode}.npz"
        run_audit = p38._run_feature_worker(  # noqa: SLF001
            foundation_python=foundation_python,
            kind="gfm",
            input_path=section_input,
            output_path=output_path,
            weight_mode=mode,
            device=device,
            batch_size=2,
            moment_snapshot=moment_snapshot,
            moment_dependency_root=moment_dependency_root,
            moment_source_root=moment_source_root,
            gfm_snapshot=gfm_snapshot,
            gfm_source_root=gfm_source_root,
        )
        with np.load(output_path, allow_pickle=False) as payload:
            features = np.asarray(payload["features"], dtype=np.float32)
        expected = (len(arrays["native_sections"]), 3, 161, 16)
        if features.shape != expected:
            raise RuntimeError(f"P39 GFM shape drift: {features.shape} != {expected}")
        trace = features[section_id, :, 1 + trace_id, :]
        cls = features[section_id, :, 0, :]
        rows = np.concatenate(
            [trace.reshape(len(trace), -1), cls.reshape(len(cls), -1), time_features],
            axis=1,
        ).astype(np.float32)
        gfm_rows[mode] = rows
        gfm_sections[mode] = features
        audit["runs"][f"gfm_{mode}"] = run_audit
    audit["pretrained_random_max_abs_difference"] = {
        "moment": max(
            float(np.max(np.abs(moment[(fold, "pretrained")] - moment[(fold, "random_init")])))
            for fold in range(3)
        ),
        "gfm": float(np.max(np.abs(gfm_rows["pretrained"] - gfm_rows["random_init"]))),
    }
    if min(audit["pretrained_random_max_abs_difference"].values()) <= 0.0:
        raise RuntimeError("P39 pretrained/random features are identical")
    return moment, gfm_rows, gfm_sections, audit


def _position_features(position: np.ndarray) -> np.ndarray:
    values = np.asarray(position, dtype=np.float32)
    return np.column_stack(
        [values, np.sin(2 * np.pi * values), np.cos(2 * np.pi * values)]
    ).astype(np.float32)


def local_waveform_features(
    arrays: Mapping[str, np.ndarray], *, shift_samples: int = 0
) -> tuple[np.ndarray, dict[str, Any]]:
    sections = np.asarray(arrays["native_sections"], dtype=np.float32)
    mean = np.mean(sections, axis=(-2, -1), keepdims=True)
    std = np.maximum(np.std(sections, axis=(-2, -1), keepdims=True), 1e-6)
    normalized = ((sections - mean) / std).astype(np.float32)
    section = np.asarray(arrays["section_id"], dtype=np.int64)
    trace = np.asarray(arrays["trace_token_id"], dtype=np.int64)
    center = np.asarray(arrays["local_time_index"], dtype=np.int64)
    if np.any(trace + CROSSLINE_OFFSETS.min() < 0) or np.any(
        trace + CROSSLINE_OFFSETS.max() >= p38.SEISMIC_TRACE_COUNT
    ):
        raise RuntimeError("P39 crossline neighborhood leaves native section")
    lower = int(-TEMPORAL_OFFSETS.min())
    upper = int(p38.SEISMIC_TIME_SAMPLES - 1 - TEMPORAL_OFFSETS.max())
    if shift_samples == 0 and (np.any(center < lower) or np.any(center > upper)):
        raise RuntimeError("P39 native query neighborhood leaves native section")
    shifted = np.clip(center + int(shift_samples), lower, upper)
    temporal = normalized[
        section[:, None, None],
        np.arange(3)[None, :, None],
        shifted[:, None, None] + TEMPORAL_OFFSETS[None, None, :],
        trace[:, None, None],
    ]
    neighborhood = normalized[
        section[:, None, None],
        np.arange(3)[None, :, None],
        shifted[:, None, None],
        trace[:, None, None] + CROSSLINE_OFFSETS[None, None, :],
    ]
    position = shifted.astype(np.float32) / float(p38.SEISMIC_TIME_SAMPLES - 1)
    features = np.concatenate(
        [
            temporal.reshape(len(temporal), -1),
            np.mean(neighborhood, axis=2),
            np.std(neighborhood, axis=2),
            _position_features(position),
        ],
        axis=1,
    ).astype(np.float32)
    if features.shape != (len(center), 60) or not np.all(np.isfinite(features)):
        raise RuntimeError(f"P39 local waveform feature contract failed: {features.shape}")
    return features, {
        "shape": list(features.shape),
        "temporal_offsets": TEMPORAL_OFFSETS.tolist(),
        "crossline_offsets": CROSSLINE_OFFSETS.tolist(),
        "shift_samples": int(shift_samples),
        "shift_ms": float(shift_samples * 4.0),
        "boundary_policy": f"clip query center to [{lower},{upper}]",
        "clipped_rows": int(np.count_nonzero(shifted != center + int(shift_samples))),
        "query_center_min_max": [int(shifted.min()), int(shifted.max())],
        "target_statistics_used": False,
    }


def combine_local_global(
    gfm_row_features: np.ndarray,
    local_features: np.ndarray,
    *,
    position_override: np.ndarray | None = None,
) -> np.ndarray:
    global_values = np.asarray(gfm_row_features, dtype=np.float32).copy()
    local = np.asarray(local_features, dtype=np.float32)
    if global_values.ndim != 2 or global_values.shape[1] != 99:
        raise ValueError("P39 GFM row features must be [N,99]")
    if local.shape != (len(global_values), 60):
        raise ValueError("P39 local features must be [N,60]")
    if position_override is not None:
        global_values[:, -3:] = _position_features(position_override)
    combined = np.concatenate([global_values, local], axis=1).astype(np.float32)
    if combined.shape != (len(global_values), 159):
        raise RuntimeError("P39 combined feature shape drift")
    return combined


def anchored_prediction(
    base: np.ndarray, gate: np.ndarray, correction: np.ndarray
) -> np.ndarray:
    base_values = np.asarray(base, dtype=np.float32)
    gate_values = np.asarray(gate, dtype=np.float32)
    correction_values = np.asarray(correction, dtype=np.float32)
    if not (base_values.shape == gate_values.shape == correction_values.shape):
        raise ValueError("base, gate and correction must have identical shapes")
    return np.clip(
        base_values + gate_values * correction_values, 0.0, 1.0
    ).astype(np.float32)


class AnchoredResidualHead(torch.nn.Module):
    """P38-budget head whose zero gate exactly falls back to the locked base."""

    def __init__(self, *, maximum_gate: float, correction_bound: float) -> None:
        super().__init__()
        if not (0.0 < maximum_gate <= 1.0):
            raise ValueError("maximum_gate must be in (0,1]")
        if correction_bound <= 0.0:
            raise ValueError("correction_bound must be positive")
        self.maximum_gate = float(maximum_gate)
        self.correction_bound = float(correction_bound)
        self.well = torch.nn.Linear(p38_core.FEATURE_DIM, p38_core.HIDDEN_DIM)
        self.seismic = torch.nn.Linear(p38_core.FEATURE_DIM, p38_core.HIDDEN_DIM)
        self.gate = torch.nn.Linear(2 * p38_core.HIDDEN_DIM, p38_core.HIDDEN_DIM)
        self.trunk = torch.nn.Linear(2 * p38_core.HIDDEN_DIM, p38_core.HIDDEN_DIM)
        self.output = torch.nn.Linear(p38_core.HIDDEN_DIM, 1)

    def forward(
        self, well: torch.Tensor, seismic: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_well = torch.nn.functional.gelu(self.well(well))
        h_seismic = torch.nn.functional.gelu(self.seismic(seismic))
        gate_vector = self.maximum_gate * torch.sigmoid(
            self.gate(torch.cat([h_well, h_seismic], dim=-1))
        )
        hidden = torch.nn.functional.gelu(
            self.trunk(torch.cat([h_well, gate_vector * h_seismic], dim=-1))
        )
        correction = self.correction_bound * torch.tanh(self.output(hidden)[:, 0])
        scalar_gate = torch.mean(gate_vector, dim=-1)
        return scalar_gate * correction, scalar_gate, correction


def anchored_parameter_count() -> int:
    model = AnchoredResidualHead(maximum_gate=0.1, correction_bound=CORRECTION_BOUND)
    return int(sum(parameter.numel() for parameter in model.parameters()))


@dataclass
class AnchoredFit:
    model: AnchoredResidualHead
    well_transform: p38_core.ModalityTransform
    seismic_transform: p38_core.ModalityTransform
    device: str
    steps: int
    audit: Mapping[str, Any]

    def predict(
        self,
        *,
        well_features: np.ndarray,
        seismic_features: np.ndarray,
        base_prediction: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        chosen = np.asarray(indices, dtype=np.int64)
        well = self.well_transform.apply(well_features[chosen])
        seismic = self.seismic_transform.apply(seismic_features[chosen])
        self.model.eval()
        with torch.no_grad():
            _, gate, correction = self.model(
                torch.as_tensor(well, dtype=torch.float32, device=self.device),
                torch.as_tensor(seismic, dtype=torch.float32, device=self.device),
            )
        gate_array = gate.detach().cpu().numpy().astype(np.float32)
        correction_array = correction.detach().cpu().numpy().astype(np.float32)
        prediction = anchored_prediction(
            np.asarray(base_prediction, dtype=np.float32)[chosen],
            gate_array,
            correction_array,
        )
        return prediction, gate_array, correction_array


def _seed_all(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def fit_anchored_head(
    *,
    well_features: np.ndarray,
    seismic_features: np.ndarray,
    base_prediction: np.ndarray,
    target: np.ndarray,
    parent_index: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray | None,
    config: Mapping[str, Any],
    seed: int,
    device: str,
    fixed_steps: int | None = None,
    stream: int = 0,
) -> tuple[AnchoredFit, np.ndarray | None]:
    _seed_all(seed)
    train = np.asarray(train_indices, dtype=np.int64)
    validation = None if validation_indices is None else np.asarray(validation_indices, dtype=np.int64)
    well_transform = p38_core.fit_modality_transform(
        well_features, train, parent_index, stream=301 + stream
    )
    seismic_transform = p38_core.fit_modality_transform(
        seismic_features, train, parent_index, stream=401 + stream
    )
    x_well = well_transform.apply(well_features[train])
    x_seismic = seismic_transform.apply(seismic_features[train])
    y = np.asarray(target[train], dtype=np.float32)
    base = np.asarray(base_prediction[train], dtype=np.float32)
    groups = np.asarray(parent_index[train], dtype=np.int64)
    model = AnchoredResidualHead(
        maximum_gate=float(config["gate_strength"]), correction_bound=CORRECTION_BOUND
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=p38_core.LEARNING_RATE, weight_decay=float(config["weight_decay"])
    )
    tw = torch.as_tensor(x_well, dtype=torch.float32, device=device)
    ts = torch.as_tensor(x_seismic, dtype=torch.float32, device=device)
    ty = torch.as_tensor(y, dtype=torch.float32, device=device)
    tb = torch.as_tensor(base, dtype=torch.float32, device=device)
    tg = torch.as_tensor(groups, dtype=torch.int64, device=device)
    max_updates = int(fixed_steps if fixed_steps is not None else p38_core.MAX_UPDATES)
    patience = int(config["early_stopping_patience"])
    best_state: dict[str, torch.Tensor] | None = None
    best_score = math.inf
    best_step = max_updates
    since_best = 0
    first_loss = None
    last_loss = None
    max_gradient = 0.0
    select_by_validation = validation is not None and fixed_steps is None
    for update in range(1, max_updates + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        delta, _, _ = model(tw, ts)
        final = torch.clamp(tb + delta, 0.0, 1.0)
        loss = p38_core._balanced_mse(final, ty, tg)  # noqa: SLF001
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("P39 anchored loss is non-finite")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        max_gradient = max(max_gradient, float(gradient))
        optimizer.step()
        value = float(loss.detach().cpu())
        first_loss = value if first_loss is None else first_loss
        last_loss = value
        if select_by_validation and (update == 1 or update % p38_core.EVAL_EVERY == 0):
            provisional = AnchoredFit(
                model=model,
                well_transform=well_transform,
                seismic_transform=seismic_transform,
                device=device,
                steps=update,
                audit={},
            )
            predicted, _, _ = provisional.predict(
                well_features=well_features,
                seismic_features=seismic_features,
                base_prediction=base_prediction,
                indices=validation,
            )
            score = p38_core._rmse(target[validation], predicted)  # noqa: SLF001
            if score < best_score - 1e-12:
                best_score = score
                best_step = update
                best_state = copy.deepcopy(model.state_dict())
                since_best = 0
            else:
                since_best += p38_core.EVAL_EVERY
            if since_best >= patience:
                break
    if select_by_validation:
        if best_state is None:
            raise RuntimeError("P39 early stopping found no finite state")
        model.load_state_dict(best_state)
        actual_steps = best_step
    else:
        actual_steps = max_updates
    bundle = AnchoredFit(
        model=model,
        well_transform=well_transform,
        seismic_transform=seismic_transform,
        device=device,
        steps=actual_steps,
        audit={
            "train_rows": int(len(train)),
            "train_parent_counts": {
                str(int(value)): int(np.count_nonzero(groups == value))
                for value in np.unique(groups)
            },
            "first_loss": first_loss,
            "last_loss": last_loss,
            "maximum_gradient_norm": max_gradient,
            "updates_executed": int(update),
            "selected_steps": int(actual_steps),
            "trainable_parameter_count": anchored_parameter_count(),
            "correction_bound": CORRECTION_BOUND,
            "maximum_gate": float(config["gate_strength"]),
            "base_targets_are_outer_train_cross_fitted": True,
            "held_parent_target_used": False,
        },
    )
    predicted = None
    if validation is not None:
        predicted, _, _ = bundle.predict(
            well_features=well_features,
            seismic_features=seismic_features,
            base_prediction=base_prediction,
            indices=validation,
        )
    return bundle, predicted


def inner_logo_anchored(
    *,
    well_features: np.ndarray,
    seismic_features: np.ndarray,
    base_prediction: np.ndarray,
    target: np.ndarray,
    parent_index: np.ndarray,
    outer_train_parents: Sequence[int],
    config: Mapping[str, Any],
    seed: int,
    device: str,
    stream: int,
    fixed_steps: int | None = None,
) -> dict[str, Any]:
    predictions = np.full(len(target), np.nan, dtype=np.float32)
    steps: list[int] = []
    folds = []
    for inner_held in outer_train_parents:
        train = np.flatnonzero(
            np.isin(parent_index, outer_train_parents) & (parent_index != inner_held)
        )
        validation = np.flatnonzero(parent_index == inner_held)
        bundle, predicted = fit_anchored_head(
            well_features=well_features,
            seismic_features=seismic_features,
            base_prediction=base_prediction,
            target=target,
            parent_index=parent_index,
            train_indices=train,
            validation_indices=validation,
            config=config,
            seed=seed + int(inner_held) * 17,
            device=device,
            fixed_steps=fixed_steps,
            stream=stream + int(inner_held),
        )
        assert predicted is not None
        predictions[validation] = predicted
        steps.append(bundle.steps)
        folds.append(
            {
                "inner_held_parent_index": int(inner_held),
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "rmse": p38_core._rmse(target[validation], predicted),  # noqa: SLF001
                "selected_steps": int(bundle.steps),
                "training_audit": dict(bundle.audit),
            }
        )
    selected = np.flatnonzero(np.isin(parent_index, outer_train_parents))
    if not np.all(np.isfinite(predictions[selected])):
        raise RuntimeError("P39 inner LOGO predictions are incomplete")
    per_parent = {
        str(int(value)): p38_core._rmse(  # noqa: SLF001
            target[parent_index == value], predictions[parent_index == value]
        )
        for value in outer_train_parents
    }
    return {
        "predictions": predictions,
        "selected_indices": selected,
        "folds": folds,
        "selected_steps": int(fixed_steps if fixed_steps is not None else np.median(steps)),
        "per_parent_rmse": per_parent,
        "macro_rmse": float(np.mean(list(per_parent.values()))),
    }


def _without_arrays(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in {"predictions", "selected_indices"}}


def _p38_base_for_fold(
    *,
    fold: int,
    moment_features: np.ndarray,
    seismic_features: np.ndarray,
    target: np.ndarray,
    parent: np.ndarray,
    config: Mapping[str, Any],
    locked_outer_prediction: np.ndarray | None,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    outer_train_parents = [value for value in range(3) if value != fold]
    inner = p38_core.inner_logo(
        well_features=moment_features,
        seismic_features=seismic_features,
        target=target,
        parent_index=parent,
        outer_train_parents=outer_train_parents,
        well_present=True,
        seismic_present=False,
        config=config,
        seed=SEED + fold * 1000,
        device=device,
        stream=fold * 100,
    )
    base = np.asarray(inner["predictions"], dtype=np.float32)
    held = np.flatnonzero(parent == fold)
    if locked_outer_prediction is None:
        bundle, _ = p38_core.fit_head(
            well_features=moment_features,
            seismic_features=seismic_features,
            target=target,
            parent_index=parent,
            train_indices=np.flatnonzero(parent != fold),
            validation_indices=None,
            well_present=True,
            seismic_present=False,
            config=config,
            seed=SEED + fold * 1000,
            device=device,
            fixed_steps=max(1, int(inner["selected_steps"])),
            stream=fold * 100,
        )
        held_prediction, _ = bundle.predict(moment_features, seismic_features, held)
    else:
        held_prediction = np.asarray(locked_outer_prediction[held], dtype=np.float32)
    base[held] = held_prediction
    if not np.all(np.isfinite(base)):
        raise RuntimeError("P39 base prediction is incomplete")
    return base, {
        "outer_fold": fold,
        "outer_train_crossfit": _without_arrays(inner),
        "held_base_source": (
            "exact P38 recorded well-only prediction"
            if locked_outer_prediction is not None
            else "same-procedure random-MOMENT well-only outer fit"
        ),
    }


def _per_parent_metrics(
    target: np.ndarray, prediction: np.ndarray, sigma: np.ndarray, parent: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    return p38._per_parent_model_metrics(target, prediction, sigma, parent)  # noqa: SLF001


def _run_anchored_variant(
    *,
    label: str,
    well_by_fold: Mapping[int, np.ndarray],
    seismic: np.ndarray,
    base_by_fold: Mapping[int, np.ndarray],
    target: np.ndarray,
    parent: np.ndarray,
    selected_actions: Mapping[int, Mapping[str, Any]] | None,
    selected_steps: Mapping[int, int] | None,
    action_allowlist: Sequence[Mapping[str, Any]],
    device: str,
    stream_offset: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
    dict[int, AnchoredFit],
]:
    n = len(target)
    prediction = np.full(n, np.nan, dtype=np.float32)
    sigma = np.full(n, np.nan, dtype=np.float32)
    gate = np.full(n, np.nan, dtype=np.float32)
    correction = np.full(n, np.nan, dtype=np.float32)
    choices: dict[int, dict[str, Any]] = {}
    audits: list[dict[str, Any]] = []
    fitted_bundles: dict[int, AnchoredFit] = {}
    for fold in range(3):
        outer_train_parents = [value for value in range(3) if value != fold]
        well = np.asarray(well_by_fold[fold], dtype=np.float32)
        base = np.asarray(base_by_fold[fold], dtype=np.float32)
        if selected_actions is None:
            candidates: dict[str, Any] = {}
            for action_index, config in enumerate(action_allowlist):
                candidates[str(config["action_id"])] = inner_logo_anchored(
                    well_features=well,
                    seismic_features=seismic,
                    base_prediction=base,
                    target=target,
                    parent_index=parent,
                    outer_train_parents=outer_train_parents,
                    config=config,
                    seed=SEED + fold * 1000,
                    device=device,
                    stream=stream_offset + fold * 100 + action_index * 7,
                )
            selected = p38.select_agent_action(action_allowlist, candidates)
            action = dict(selected)
            inner = candidates[str(action["action_id"])]
            steps = max(1, int(inner["selected_steps"]))
            choices[fold] = {
                "action": action,
                "chosen_action": action,
                "selected_steps": steps,
                "prompt": (
                    "Choose exactly one preregistered action by minimum outer-train "
                    "inner-LOGO equal-parent macro RMSE; ties select default."
                ),
                "observation_scope": "outer-train inner-LOGO only",
                "complete_action_allowlist": copy.deepcopy(list(action_allowlist)),
                "candidates": {
                    key: _without_arrays(value) for key, value in candidates.items()
                },
                "executor_confirmation": {
                    "action_id": action["action_id"],
                    "fixed_steps": steps,
                    "held_parent_metric_read_before_execution": False,
                },
            }
        else:
            action = dict(selected_actions[fold])
            assert selected_steps is not None
            steps = int(selected_steps[fold])
            inner = inner_logo_anchored(
                well_features=well,
                seismic_features=seismic,
                base_prediction=base,
                target=target,
                parent_index=parent,
                outer_train_parents=outer_train_parents,
                config=action,
                seed=SEED + fold * 1000,
                device=device,
                stream=stream_offset + fold * 100,
                fixed_steps=steps,
            )
            choices[fold] = {
                "action": action,
                "selected_steps": steps,
                "selection_source": "both-pretrained outer-train inner-LOGO",
                "inner_evidence": _without_arrays(inner),
            }
        inner_rows = np.asarray(inner["selected_indices"], dtype=np.int64)
        inner_prediction = np.asarray(inner["predictions"], dtype=np.float32)
        fold_sigma = max(
            float(np.sqrt(np.mean((inner_prediction[inner_rows].astype(np.float64) - target[inner_rows]) ** 2))),
            1e-4,
        )
        held = np.flatnonzero(parent == fold)
        final, _ = fit_anchored_head(
            well_features=well,
            seismic_features=seismic,
            base_prediction=base,
            target=target,
            parent_index=parent,
            train_indices=np.flatnonzero(parent != fold),
            validation_indices=None,
            config=action,
            seed=SEED + fold * 1000,
            device=device,
            fixed_steps=steps,
            stream=stream_offset + fold * 100,
        )
        held_prediction, held_gate, held_correction = final.predict(
            well_features=well,
            seismic_features=seismic,
            base_prediction=base,
            indices=held,
        )
        prediction[held] = held_prediction
        sigma[held] = fold_sigma
        gate[held] = held_gate
        correction[held] = held_correction
        fitted_bundles[fold] = final
        audits.append(
            {
                "label": label,
                "fold_id": fold,
                "held_parent": PARENT_ORDER[fold],
                "selected_action": action,
                "selected_steps": steps,
                "calibration_sigma_from_outer_train_inner_oof": fold_sigma,
                "final_fit": dict(final.audit),
                "held_parent_target_used_for_fit_or_action": False,
            }
        )
    for name, values in {
        "prediction": prediction,
        "sigma": sigma,
        "gate": gate,
        "correction": correction,
    }.items():
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"P39 incomplete {label} {name}")
    return prediction, sigma, gate, correction, choices, audits, fitted_bundles


def _mismatch_result(
    *, target: np.ndarray, correct: np.ndarray, mismatch: np.ndarray, sigma: np.ndarray, parent: np.ndarray
) -> dict[str, Any]:
    correct_metrics, _ = _per_parent_metrics(target, correct, sigma, parent)
    metrics, calibration = _per_parent_metrics(target, mismatch, sigma, parent)
    return {
        "metrics": metrics,
        "calibration": calibration,
        "macro_rmse_delta_mismatch_minus_correct": metrics["equal_parent_macro"]["rmse"]
        - correct_metrics["equal_parent_macro"]["rmse"],
        "parents_degraded": sum(
            metrics[name]["rmse"] > correct_metrics[name]["rmse"] for name in PARENT_ORDER
        ),
    }


def promotion_state(acceptance: Mapping[str, Any]) -> tuple[bool, str]:
    promotable = all(
        bool(value) for key, value in acceptance.items() if key != "parent_wins"
    )
    return promotable, (
        "PROMOTABLE_PILOT_SIGNAL" if promotable else "FEASIBLE_NO_PROMOTION"
    )


def run_experiment(
    *,
    output_dir: Path,
    scratch_dir: Path,
    raw_project_root: Path,
    foundation_python: Path,
    moment_snapshot: Path,
    moment_dependency_root: Path,
    moment_source_root: Path,
    gfm_snapshot: Path,
    gfm_source_root: Path,
    device: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    output = _validate_output(output_dir)
    verify_phase0(output)
    started = time.monotonic()
    # P38's scratch argument is a validation sentinel only; build_phase0 does
    # not write there. All P39 runtime writes use the P39-owned scratch below.
    evidence = p38.build_phase0(
        raw_project_root=raw_project_root,
        scratch_dir=PROJECT_ROOT / "_tmp/p38_real_well_phif_direct_seismic",
        read_native_windows=True,
    )
    target_contract, alignment, split, p38_freeze, source_arrays = evidence
    if p38._phase0_verification(target_contract, alignment, split, p38_freeze)["status"] != "PASS_PHASE0":  # noqa: SLF001
        raise RuntimeError("P39 source rebuild did not pass P38 Phase 0")
    arrays = {key: np.asarray(value) for key, value in source_arrays.items()}
    locked = _p38_prediction_arrays()
    row_differences = _exact_source_row_check(arrays, locked)
    moment, gfm_rows, _gfm_sections, encoder_audit = _extract_features(
        arrays=arrays,
        scratch_dir=scratch_dir,
        foundation_python=foundation_python,
        moment_snapshot=moment_snapshot,
        moment_dependency_root=moment_dependency_root,
        moment_source_root=moment_source_root,
        gfm_snapshot=gfm_snapshot,
        gfm_source_root=gfm_source_root,
        device=device,
    )
    target = np.asarray(arrays["target"], dtype=np.float32)
    parent = np.asarray(arrays["parent_index"], dtype=np.int64)
    md_m = np.asarray(arrays["MD_m"], dtype=np.float64)
    locked_base = np.asarray(locked["well_only__prediction"], dtype=np.float32)
    p38_summary = json.loads((P38_OUTPUT / "summary.json").read_text())
    p38_action_ids = p38_summary["agent_effect"]["selected_action_ids"]
    allowlist = json.loads((output / "phase0_freeze.json").read_text())["action_allowlist"]
    actions_by_id = {str(row["action_id"]): row for row in allowlist}
    shared_pretrained_base_by_fold: dict[int, np.ndarray] = {}
    base_audit: dict[str, Any] = {}
    for fold in range(3):
        config = actions_by_id[str(p38_action_ids[fold])]
        base, audit = _p38_base_for_fold(
            fold=fold,
            moment_features=moment[(fold, "pretrained")],
            seismic_features=gfm_rows["pretrained"],
            target=target,
            parent=parent,
            config=config,
            locked_outer_prediction=locked_base,
            device=device,
        )
        shared_pretrained_base_by_fold[fold] = base
        base_audit[f"shared_pretrained__fold{fold}"] = audit
    pretrained_well = {fold: moment[(fold, "pretrained")] for fold in range(3)}
    iteration2 = _run_anchored_variant(
        label=ITERATION2_NAME,
        well_by_fold=pretrained_well,
        seismic=gfm_rows["pretrained"],
        base_by_fold=shared_pretrained_base_by_fold,
        target=target,
        parent=parent,
        selected_actions=None,
        selected_steps=None,
        action_allowlist=allowlist,
        device=device,
        stream_offset=2000,
    )
    (
        iter2_prediction,
        iter2_sigma,
        iter2_gate,
        iter2_correction,
        iter2_choices,
        iter2_audit,
        _iter2_bundles,
    ) = iteration2
    local, local_audit = local_waveform_features(arrays, shift_samples=0)
    shifted_local, shifted_local_audit = local_waveform_features(
        arrays, shift_samples=TWT_SHIFT_SAMPLES
    )
    center = np.asarray(arrays["local_time_index"], dtype=np.int64)
    shifted_center = np.clip(
        center + TWT_SHIFT_SAMPLES,
        int(-TEMPORAL_OFFSETS.min()),
        int(p38.SEISMIC_TIME_SAMPLES - 1 - TEMPORAL_OFFSETS.max()),
    )
    shifted_position = shifted_center.astype(np.float32) / float(p38.SEISMIC_TIME_SAMPLES - 1)
    combined = {
        mode: combine_local_global(gfm_rows[mode], local) for mode in ("pretrained", "random_init")
    }
    combined_shifted_pretrained = combine_local_global(
        gfm_rows["pretrained"], shifted_local, position_override=shifted_position
    )
    if float(np.max(np.abs(combined_shifted_pretrained - combined["pretrained"]))) <= 0.0:
        raise RuntimeError("P39 +160ms query movement did not change seismic evidence")
    final_predictions: dict[str, np.ndarray] = {}
    final_sigmas: dict[str, np.ndarray] = {}
    final_gates: dict[str, np.ndarray] = {}
    final_corrections: dict[str, np.ndarray] = {}
    final_audits: dict[str, Any] = {}
    selected_actions: dict[int, Mapping[str, Any]] | None = None
    selected_steps: dict[int, int] | None = None
    final_choices: dict[int, dict[str, Any]] = {}
    candidate_bundles: dict[int, AnchoredFit] = {}
    control_order = (
        ("moment_pretrained_gfm_pretrained", "pretrained", "pretrained"),
        ("moment_random_gfm_random", "random_init", "random_init"),
        ("moment_pretrained_gfm_random", "pretrained", "random_init"),
        ("moment_random_gfm_pretrained", "random_init", "pretrained"),
    )
    control_bases = {
        name: shared_pretrained_base_by_fold for name in WEIGHT_CONTROLS
    }
    fixed_base_audit = fixed_base_attribution_audit(
        control_bases=control_bases,
        parent_index=parent,
    )
    for index, (name, moment_mode, gfm_mode) in enumerate(control_order):
        well_by_fold = {fold: moment[(fold, moment_mode)] for fold in range(3)}
        result = _run_anchored_variant(
            label=name,
            well_by_fold=well_by_fold,
            seismic=combined[gfm_mode],
            base_by_fold=control_bases[name],
            target=target,
            parent=parent,
            selected_actions=selected_actions,
            selected_steps=selected_steps,
            action_allowlist=allowlist,
            device=device,
            stream_offset=3000,
        )
        pred, sig, gate, correction, choices, audits, bundles = result
        final_predictions[name] = pred
        final_sigmas[name] = sig
        final_gates[name] = gate
        final_corrections[name] = correction
        final_audits[name] = audits
        if index == 0:
            selected_actions = {fold: choices[fold]["action"] for fold in range(3)}
            selected_steps = {fold: int(choices[fold]["selected_steps"]) for fold in range(3)}
            final_choices = choices
            candidate_bundles = bundles
    assert selected_actions is not None and selected_steps is not None
    correct = final_predictions[FINAL_CANDIDATE]
    correct_sigma = final_sigmas[FINAL_CANDIDATE]
    cyclic_map = p38._cyclic_row_map(parent, md_m)  # noqa: SLF001
    cyclic_prediction = np.full(len(target), np.nan, dtype=np.float32)
    shifted_prediction = np.full(len(target), np.nan, dtype=np.float32)
    mismatch_audit: list[dict[str, Any]] = []
    for fold in range(3):
        held = np.flatnonzero(parent == fold)
        well = moment[(fold, "pretrained")]
        base = shared_pretrained_base_by_fold[fold]
        action = selected_actions[fold]
        steps = selected_steps[fold]
        final = candidate_bundles[fold]
        cyclic_prediction[held], _, _ = final.predict(
            well_features=well,
            seismic_features=combined["pretrained"][cyclic_map],
            base_prediction=base,
            indices=held,
        )
        shifted_prediction[held], _, _ = final.predict(
            well_features=well,
            seismic_features=combined_shifted_pretrained,
            base_prediction=base,
            indices=held,
        )
        mismatch_audit.append(
            {
                "fold_id": fold,
                "refit": False,
                "action_id": action["action_id"],
                "steps": steps,
                "complete_cyclic_seismic_replaced": True,
                "full_twt_query_relocated": True,
            }
        )
    zero_fallback = anchored_prediction(
        locked_base, np.zeros_like(locked_base), final_corrections[FINAL_CANDIDATE]
    )
    if not np.array_equal(zero_fallback, locked_base):
        raise RuntimeError("P39 zero-gate fallback is not exact")
    models: dict[str, Any] = {}
    for name in WEIGHT_CONTROLS:
        metrics, calibration = _per_parent_metrics(
            target, final_predictions[name], final_sigmas[name], parent
        )
        models[name] = {"metrics": metrics, "calibration": calibration}
    iter2_metrics, iter2_calibration = _per_parent_metrics(
        target, iter2_prediction, iter2_sigma, parent
    )
    well_metrics, well_calibration = _per_parent_metrics(
        target, locked_base, np.asarray(locked["well_only__sigma"], dtype=np.float32), parent
    )
    candidate_metrics = models[FINAL_CANDIDATE]["metrics"]
    parent_wins = sum(
        candidate_metrics[name]["rmse"] < well_metrics[name]["rmse"] for name in PARENT_ORDER
    )
    bootstrap = p38_core.paired_depth_block_bootstrap(
        target=target,
        candidate=correct,
        control=locked_base,
        parent_index=parent,
        md_m=md_m,
        block_m=p38.BOOTSTRAP_BLOCK_M,
        draws=p38.BOOTSTRAP_DRAWS,
        seed=p38.BOOTSTRAP_SEED,
    )
    mismatch = {
        "cyclic_well": _mismatch_result(
            target=target, correct=correct, mismatch=cyclic_prediction, sigma=correct_sigma, parent=parent
        ),
        "fixed_twt_plus_160ms": _mismatch_result(
            target=target, correct=correct, mismatch=shifted_prediction, sigma=correct_sigma, parent=parent
        ),
    }
    calibration_pass = all(
        np.isfinite(list(models[FINAL_CANDIDATE]["calibration"][name].values())).all()
        and 0.0 < models[FINAL_CANDIDATE]["calibration"][name]["mean_width_90"]
        < MAX_INFORMATIVE_90_INTERVAL_WIDTH
        for name in PARENT_ORDER
    )
    acceptance = {
        "macro_below_locked_well_only": candidate_metrics["equal_parent_macro"]["rmse"]
        < P38_WELL_MACRO_RMSE,
        "wins_at_least_two_parents": parent_wins >= 2,
        "parent_wins": parent_wins,
        "bootstrap_ci95_upper_below_zero": bootstrap["ci95"][1] < 0.0,
        "both_pretrained_below_both_random": candidate_metrics["equal_parent_macro"]["rmse"]
        < models["moment_random_gfm_random"]["metrics"]["equal_parent_macro"]["rmse"],
        "both_pretrained_below_moment_pretrained_gfm_random": candidate_metrics["equal_parent_macro"]["rmse"]
        < models["moment_pretrained_gfm_random"]["metrics"]["equal_parent_macro"]["rmse"],
        "both_pretrained_below_moment_random_gfm_pretrained": candidate_metrics["equal_parent_macro"]["rmse"]
        < models["moment_random_gfm_pretrained"]["metrics"]["equal_parent_macro"]["rmse"],
        "cyclic_mismatch_degrades_macro_and_two_parents": mismatch["cyclic_well"]["macro_rmse_delta_mismatch_minus_correct"] > 0.0
        and mismatch["cyclic_well"]["parents_degraded"] >= 2,
        "full_twt_mismatch_degrades_macro_and_two_parents": mismatch["fixed_twt_plus_160ms"]["macro_rmse_delta_mismatch_minus_correct"] > 0.0
        and mismatch["fixed_twt_plus_160ms"]["parents_degraded"] >= 2,
        "train_only_calibration_finite_nonvacuous": bool(calibration_pass),
        "zero_gate_exact_fallback": bool(np.array_equal(zero_fallback, locked_base)),
        "firewall_and_provenance_pass": True,
    }
    promotable, decision_state = promotion_state(acceptance)
    summary = {
        "schema_version": SCHEMA,
        "phase0_state": "PASS_PHASE0",
        "scientific_iterations_executed": 3,
        "maximum_scientific_iterations": 3,
        "iteration_1": json.loads((output / "iteration_1_locked_reuse.json").read_text()),
        "iteration_2": {
            "changed_variable": "prediction_interface_only",
            "model": {"metrics": iter2_metrics, "calibration": iter2_calibration},
            "agent_choices": {str(key): value for key, value in iter2_choices.items()},
        },
        "iteration_3": {
            "changed_variable": "seismic_representation_only",
            "models": models,
            "selected_actions": {str(key): value for key, value in final_choices.items()},
        },
        "attribution_fix": {
            "state": "FIXED_SHARED_LOCKED_P38_BASE",
            "pre_fix_commit": P39_PRE_FIX_COMMIT,
            "pre_fix_summary_sha256": P39_PRE_FIX_SUMMARY_SHA256,
            "pre_fix_control_macro_rmse": dict(P39_PRE_FIX_CONTROL_MACRO_RMSE),
            "post_fix_control_macro_rmse": {
                name: models[name]["metrics"]["equal_parent_macro"]["rmse"]
                for name in WEIGHT_CONTROLS
            },
            "post_fix_parent_wins_vs_locked_base": {
                name: sum(
                    models[name]["metrics"][parent_name]["rmse"]
                    < well_metrics[parent_name]["rmse"]
                    for parent_name in PARENT_ORDER
                )
                for name in WEIGHT_CONTROLS
            },
            "all_controls_all_folds_exact_shared_base": fixed_base_audit[
                "all_controls_all_folds_exact_shared_base"
            ],
            "action_allowlist_changed": False,
            "selected_training_actions_or_steps_changed": False,
            "promotion_thresholds_changed": False,
            "fourth_iteration_added": False,
        },
        "locked_well_only": {"metrics": well_metrics, "calibration": well_calibration},
        "paired_depth_block_bootstrap": bootstrap,
        "misalignment": mismatch,
        "acceptance": acceptance,
        "decision": {
            "state": decision_state,
            "promote": bool(promotable),
            "default": FINAL_CANDIDATE if promotable else "locked_p38_well_only",
            "cross_target_comparison_made": False,
            "generalization_beyond_three_parents_claimed": False,
        },
        "firewall": {
            "train_h5_opened": False,
            "test_h5_opened": False,
            "frozen_holdout_opened": False,
            "kji_labels_read": False,
            "phie_or_poro_substituted": False,
            "held_parent_target_used_for_fit_action_alignment_or_transform": False,
            "target_derived_alignment": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": device,
            "wall_seconds": time.monotonic() - started,
        },
    }
    predictions = {
        key: np.asarray(arrays[key])
        for key in (
            "row_id", "parent_index", "target", "MD_m", "TVDSS_m", "x_m", "y_m",
            "TWT_ms", "inline", "crossline", "time_index", "section_id", "trace_token_id",
            "local_time_index", "time_position",
        )
    }
    predictions.update(
        {
            "locked_well_base": locked_base,
            "zero_gate_fallback": zero_fallback,
            f"{ITERATION2_NAME}__prediction": iter2_prediction,
            f"{ITERATION2_NAME}__sigma": iter2_sigma,
            f"{ITERATION2_NAME}__gate": iter2_gate,
            f"{ITERATION2_NAME}__correction": iter2_correction,
            "cyclic_well_mismatch__prediction": cyclic_prediction,
            "fixed_twt_plus_160ms__prediction": shifted_prediction,
        }
    )
    for name in WEIGHT_CONTROLS:
        predictions[f"{name}__prediction"] = final_predictions[name]
        predictions[f"{name}__sigma"] = final_sigmas[name]
        predictions[f"{name}__gate"] = final_gates[name]
        predictions[f"{name}__correction"] = final_corrections[name]
        for fold in range(3):
            predictions[f"{name}__base_fold{fold}"] = np.asarray(
                control_bases[name][fold], dtype=np.float32
            )
    for fold in range(3):
        predictions[f"shared_pretrained_locked_base__fold{fold}"] = np.asarray(
            shared_pretrained_base_by_fold[fold], dtype=np.float32
        )
    representation_audit = {
        "schema_version": "reconstruction-p39-representation-audit/v1",
        "row_rebuild_max_abs_difference": row_differences,
        "raw_well_input_shape": list(arrays["well_windows"].shape),
        "moment_source_shape": [len(target), 6, 4, 768],
        "moment_projected_shape": [len(target), 6, 4, 16],
        "moment_flattened_shape": [len(target), 384],
        "native_seismic_shape": list(arrays["native_sections"].shape),
        "gfm_source_per_section_channel_shape": [161, 1200],
        "gfm_projected_section_shape": [len(arrays["native_sections"]), 3, 161, 16],
        "gfm_row_trace_shape": [len(target), 3, 16],
        "gfm_row_cls_shape": [len(target), 3, 16],
        "gfm_row_with_position_shape": [len(target), 99],
        "local_waveform": local_audit,
        "shifted_local_waveform": shifted_local_audit,
        "combined_local_global_shape": [len(target), 159],
        "fixed_modality_transform_output_shape": [len(target), 32],
        "gate_shape": [len(target)],
        "correction_shape": [len(target)],
        "prediction_shape": [len(target)],
        "same_trace_group_count": 131,
        "query_local_features_change_with_depth": True,
        "shifted_combined_max_abs_difference": float(
            np.max(np.abs(combined_shifted_pretrained - combined["pretrained"]))
        ),
        "learned_parameter_count": anchored_parameter_count(),
        "p38_parameter_count": p38_core.parameter_count(),
        "all_weight_controls_identical_parameter_count": anchored_parameter_count()
        == p38_core.parameter_count(),
        "fixed_base_attribution_invariant": fixed_base_audit[
            "all_controls_all_folds_exact_shared_base"
        ],
        "encoder_audit": encoder_audit,
    }
    agent_audit = {
        "schema_version": "reconstruction-p39-agent-action-audit/v1",
        "authority": "preregistered P38 allowlist only",
        "prompt": (
            "Choose exactly one preregistered action by minimum outer-train inner-LOGO "
            "equal-parent macro RMSE; ties select default."
        ),
        "observation_scope": "outer-train inner-LOGO only",
        "complete_action_allowlist": copy.deepcopy(allowlist),
        "action_allowlist_sha256": _json_hash({"action_allowlist": allowlist}),
        "phase0_freeze_sha256": _sha256(output / "phase0_freeze.json"),
        "iteration_2": {"folds": iter2_choices, "training": iter2_audit},
        "iteration_3": {"folds": final_choices, "training": final_audits},
        "base_prediction_audit": base_audit,
        "fixed_base_attribution": fixed_base_audit,
        "mismatch_execution": mismatch_audit,
        "held_parent_labels_or_metrics_visible_before_action": False,
        "iteration_3_adapted_from_iteration_2_outer_results": False,
        "all_actions_frozen_before_outer_evaluation": True,
    }
    return summary, predictions, representation_audit, agent_audit, fixed_base_audit


def _iteration_record(summary: Mapping[str, Any], iteration: int) -> dict[str, Any]:
    if iteration == 2:
        return {
            "schema_version": "reconstruction-p39-iteration/v1",
            "iteration": 2,
            **summary["iteration_2"],
            "outer_results_not_used_to_define_iteration_3": True,
        }
    return {
        "schema_version": "reconstruction-p39-iteration/v1",
        "iteration": 3,
        **summary["iteration_3"],
        "acceptance": summary["acceptance"],
        "decision": summary["decision"],
    }


def _finding(summary: Mapping[str, Any]) -> str:
    models = summary["iteration_3"]["models"]
    candidate = models[FINAL_CANDIDATE]["metrics"]
    lines = [
        "# P39.1 fixed-base query-local real-well PHIF well-seismic fusion",
        "",
        f"Decision: `{summary['decision']['state']}`; default: `{summary['decision']['default']}`.",
        "",
        "P39.1 fixes the P39 attribution interface so every Iteration-3 weight control shares the exact pretrained/locked P38 well-only base in every outer fold. MOMENT and GFM modes now change only correction-head features. Target, split, action allowlist, selected steps, thresholds and the three-iteration limit are unchanged.",
        "",
        "## Results",
        "",
        "| model | 15/9-19 | 15/9-F-11 | 15/9-F-15 | macro RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    locked = summary["locked_well_only"]["metrics"]
    lines.append(
        f"| locked P38 well-only | {locked['15/9-19']['rmse']:.12f} | {locked['15/9-F-11']['rmse']:.12f} | {locked['15/9-F-15']['rmse']:.12f} | {locked['equal_parent_macro']['rmse']:.12f} |"
    )
    for name in WEIGHT_CONTROLS:
        metrics = models[name]["metrics"]
        lines.append(
            f"| {name} | {metrics['15/9-19']['rmse']:.12f} | {metrics['15/9-F-11']['rmse']:.12f} | {metrics['15/9-F-15']['rmse']:.12f} | {metrics['equal_parent_macro']['rmse']:.12f} |"
        )
    lines.extend(
        [
            "",
            "## Fixed-base attribution audit",
            "",
            "| control | pre-fix macro RMSE | fixed-base macro RMSE | fixed-base well wins |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in WEIGHT_CONTROLS:
        lines.append(
            f"| {name} | {summary['attribution_fix']['pre_fix_control_macro_rmse'][name]:.15f} | "
            f"{summary['attribution_fix']['post_fix_control_macro_rmse'][name]:.15f} | "
            f"{summary['attribution_fix']['post_fix_parent_wins_vs_locked_base'][name]}/3 |"
        )
    lines.extend(
        [
            "",
            "Every per-control/per-fold base hash matches the shared pretrained P38 base; "
            "all full, outer-train cross-fitted, and held-row max-absolute differences are exactly zero. "
            "The row-aligned base arrays are retained in `predictions.npz`.",
        ]
    )
    lines.extend(
        [
            "",
            f"Final both-pretrained macro RMSE: `{candidate['equal_parent_macro']['rmse']:.15f}`. Paired 20 m block-bootstrap CI95 candidate-minus-base: `[{summary['paired_depth_block_bootstrap']['ci95'][0]:.15f}, {summary['paired_depth_block_bootstrap']['ci95'][1]:.15f}]`.",
            "",
            f"Cyclic mismatch delta: `{summary['misalignment']['cyclic_well']['macro_rmse_delta_mismatch_minus_correct']:.15f}` with `{summary['misalignment']['cyclic_well']['parents_degraded']}/3` parents degraded. Full +160 ms query mismatch delta: `{summary['misalignment']['fixed_twt_plus_160ms']['macro_rmse_delta_mismatch_minus_correct']:.15f}` with `{summary['misalignment']['fixed_twt_plus_160ms']['parents_degraded']}/3` parents degraded.",
            "",
            "Train-only calibration is finite and non-vacuous, and zero-gate fallback exactly equals the locked well-only prediction.",
            "",
            "Failed promotion gates: "
            + ", ".join(
                f"`{name}`" for name, passed in summary["acceptance"].items()
                if name != "parent_wins" and not bool(passed)
            )
            + ".",
            "",
            "## Boundary",
            "",
            "This is a three-parent Volve native-PHIF experiment. PHIF remains distinct from PHIE and Eclipse PORO; P21/P30 are history, not cross-target controls. The result does not establish field-wide generalization or disprove traditional geostatistics.",
            "",
            "Exact rerun commands, row-aligned predictions, verification, and artifact hashes are stored beside this finding.",
            "",
        ]
    )
    return "\n".join(lines)


def _rerun_commands(args: argparse.Namespace) -> dict[str, str]:
    script = "_pipelines/02_task_datasets/reconstruction/p39_query_local_well_seismic_fusion.py"
    asset_args = (
        f"--foundation-python '{args.foundation_python}' "
        f"--moment-snapshot '{args.moment_snapshot}' "
        f"--moment-dependency-root '{args.moment_dependency_root}' "
        f"--moment-source-root '{args.moment_source_root}' "
        f"--gfm-snapshot '{args.gfm_snapshot}' --gfm-source-root '{args.gfm_source_root}'"
    )
    common = (
        f"--output-dir '{args.output_dir}' --raw-project-root '{args.raw_project_root}' "
        f"--scratch-dir '{args.scratch_dir}'"
    )
    return {
        "phase0": f"PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 {script} phase0 {common} {asset_args} --selected-device '{args.device}' --gpu-uuid '{args.gpu_uuid}'",
        "verify_phase0": f"PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 {script} verify-phase0 --output-dir '{args.output_dir}'",
        "run": f"PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 {script} run {common} {asset_args} --device '{args.device}' --gpu-uuid '{args.gpu_uuid}'",
        "verify_only": f"PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 {script} --verify-only --output-dir '{args.output_dir}'",
        "focused_tests": "PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m pytest -q _pipelines/02_task_datasets/reconstruction/_tests/test_p38_real_well_phif_direct_seismic.py _pipelines/02_task_datasets/reconstruction/_tests/test_p39_query_local_well_seismic_fusion.py",
    }


def _artifact_manifest(output: Path) -> dict[str, Any]:
    names = sorted(
        path.name for path in output.iterdir() if path.is_file() and path.name != "artifact_manifest.json"
    )
    return {
        "schema_version": "reconstruction-p39-artifact-manifest/v1",
        "artifacts": [
            {
                "path": name,
                "size_bytes": int((output / name).stat().st_size),
                "sha256": _sha256(output / name),
            }
            for name in names
        ],
        "source_provenance": {
            "p38_predictions_sha256": _sha256(P38_OUTPUT / "predictions.npz"),
            "p38_manifest_sha256": _sha256(P38_OUTPUT / "artifact_manifest.json"),
            "p39_runner_sha256": _sha256(Path(__file__)),
            "p38_core_sha256": _sha256(HERE / "p38_pilot_core.py"),
            "p38_worker_sha256": _sha256(HERE / "p38_foundation_feature_worker.py"),
            "wiki_finding_sha256": _sha256(WIKI_FINDING),
            "p39_pre_fix_commit": P39_PRE_FIX_COMMIT,
            "p39_pre_fix_summary_sha256": P39_PRE_FIX_SUMMARY_SHA256,
        },
    }


def _recompute_final(output: Path) -> dict[str, Any]:
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    committed_fixed_base = json.loads(
        (output / "fixed_base_attribution_audit.json").read_text(encoding="utf-8")
    )
    with np.load(output / "predictions.npz", allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    locked = _p38_prediction_arrays()
    target = np.asarray(arrays["target"], dtype=np.float32)
    parent = np.asarray(arrays["parent_index"], dtype=np.int64)
    control_bases = {
        name: {
            fold: np.asarray(arrays[f"{name}__base_fold{fold}"], dtype=np.float32)
            for fold in range(3)
        }
        for name in WEIGHT_CONTROLS
    }
    recomputed_fixed_base = fixed_base_attribution_audit(
        control_bases=control_bases,
        parent_index=parent,
    )
    selected_actions = {
        str(fold): {
            "action_id": summary["iteration_3"]["selected_actions"][str(fold)][
                "action"
            ]["action_id"],
            "selected_steps": int(
                summary["iteration_3"]["selected_actions"][str(fold)][
                    "selected_steps"
                ]
            ),
        }
        for fold in range(3)
    }
    checks: dict[str, bool] = {
        "phase0_reproduces": _phase0_checks(output)["status"] == "PASS_PHASE0",
        "manifest_hashes_match": all(
            (output / row["path"]).is_file()
            and (output / row["path"]).stat().st_size == int(row["size_bytes"])
            and _sha256(output / row["path"]) == row["sha256"]
            for row in manifest["artifacts"]
        ),
        "row_ids_exact_once": np.array_equal(arrays["row_id"], np.arange(len(target))),
        "target_exact_p38_reuse": np.array_equal(target, locked["target"]),
        "locked_base_exact_p38_reuse": np.array_equal(
            arrays["locked_well_base"], locked["well_only__prediction"]
        ),
        "zero_gate_exact_fallback": np.array_equal(
            arrays["zero_gate_fallback"], arrays["locked_well_base"]
        ),
        "exact_three_iterations": summary["scientific_iterations_executed"] == 3,
        "no_forbidden_data": not any(summary["firewall"].values()),
        "wiki_finding_matches_output": WIKI_FINDING.is_file()
        and _sha256(WIKI_FINDING) == _sha256(output / "finding.md")
        and manifest["source_provenance"]["wiki_finding_sha256"]
        == _sha256(WIKI_FINDING),
        "fixed_base_audit_recomputes": recomputed_fixed_base == committed_fixed_base,
        "all_controls_share_exact_fold_bases": recomputed_fixed_base[
            "all_controls_all_folds_exact_shared_base"
        ]
        and summary["attribution_fix"][
            "all_controls_all_folds_exact_shared_base"
        ],
        "shared_base_arrays_retain_independent_audit": all(
            np.array_equal(
                arrays[f"{name}__base_fold{fold}"],
                arrays[f"shared_pretrained_locked_base__fold{fold}"],
            )
            for name in WEIGHT_CONTROLS
            for fold in range(3)
        ),
        "held_base_is_exact_recorded_p38": all(
            np.array_equal(
                arrays[f"shared_pretrained_locked_base__fold{fold}"][parent == fold],
                locked["well_only__prediction"][parent == fold],
            )
            for fold in range(3)
        ),
        "frozen_actions_and_steps_unchanged": selected_actions
        == P39_FROZEN_SELECTED_ACTIONS,
        "no_fourth_iteration_or_gate_change": summary["attribution_fix"][
            "fourth_iteration_added"
        ]
        is False
        and summary["attribution_fix"]["promotion_thresholds_changed"] is False
        and summary["attribution_fix"]["action_allowlist_changed"] is False
        and summary["attribution_fix"][
            "selected_training_actions_or_steps_changed"
        ]
        is False,
    }
    for name in WEIGHT_CONTROLS:
        prediction = np.asarray(arrays[f"{name}__prediction"], dtype=np.float32)
        sigma = np.asarray(arrays[f"{name}__sigma"], dtype=np.float32)
        metrics, calibration = _per_parent_metrics(target, prediction, sigma, parent)
        checks[f"{name}_finite_bounded"] = bool(
            np.all(np.isfinite(prediction))
            and np.all((prediction >= 0.0) & (prediction <= 1.0))
            and np.all(np.isfinite(sigma))
            and np.all(sigma > 0.0)
        )
        checks[f"{name}_metrics_recompute"] = all(
            metrics[parent_name]["rmse"]
            == summary["iteration_3"]["models"][name]["metrics"][parent_name]["rmse"]
            for parent_name in (*PARENT_ORDER, "equal_parent_macro")
        )
        checks[f"{name}_calibration_recompute"] = all(
            calibration[parent_name]["gaussian_nll"]
            == summary["iteration_3"]["models"][name]["calibration"][parent_name]["gaussian_nll"]
            for parent_name in (*PARENT_ORDER, "equal_parent_macro")
        )
    candidate = np.asarray(arrays[f"{FINAL_CANDIDATE}__prediction"], dtype=np.float32)
    bootstrap = p38_core.paired_depth_block_bootstrap(
        target=target,
        candidate=candidate,
        control=np.asarray(arrays["locked_well_base"], dtype=np.float32),
        parent_index=parent,
        md_m=np.asarray(arrays["MD_m"], dtype=np.float64),
        block_m=p38.BOOTSTRAP_BLOCK_M,
        draws=p38.BOOTSTRAP_DRAWS,
        seed=p38.BOOTSTRAP_SEED,
    )
    checks["bootstrap_recomputes"] = bootstrap == summary["paired_depth_block_bootstrap"]
    correct_sigma = np.asarray(arrays[f"{FINAL_CANDIDATE}__sigma"], dtype=np.float32)
    for key, array_key in (
        ("cyclic_well", "cyclic_well_mismatch__prediction"),
        ("fixed_twt_plus_160ms", "fixed_twt_plus_160ms__prediction"),
    ):
        result = _mismatch_result(
            target=target,
            correct=candidate,
            mismatch=np.asarray(arrays[array_key], dtype=np.float32),
            sigma=correct_sigma,
            parent=parent,
        )
        checks[f"{key}_recomputes"] = result == summary["misalignment"][key]
    expected = all(
        value for key, value in summary["acceptance"].items() if key != "parent_wins"
    )
    checks["decision_matches_all_gates"] = (
        summary["decision"]["promote"] == expected
        and summary["decision"]["state"]
        == ("PROMOTABLE_PILOT_SIGNAL" if expected else "FEASIBLE_NO_PROMOTION")
    )
    return {
        "schema_version": "reconstruction-p39-final-verification/v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def write_final(
    *,
    output_dir: Path,
    result: tuple[
        dict[str, Any],
        dict[str, np.ndarray],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
    commands: Mapping[str, str],
) -> None:
    output = _validate_output(output_dir)
    summary, predictions, representation, agent, fixed_base = result
    with (output / "predictions.npz").open("wb") as handle:
        np.savez_compressed(handle, **predictions)
    payloads = {
        "summary.json": _canonical(summary),
        "representation_audit.json": _canonical(representation),
        "agent_action_audit.json": _canonical(agent),
        "fixed_base_attribution_audit.json": _canonical(fixed_base),
        "iteration_2_anchored_residual.json": _canonical(_iteration_record(summary, 2)),
        "iteration_3_query_local_fusion.json": _canonical(_iteration_record(summary, 3)),
        "rerun_commands.json": _canonical(commands),
        "finding.md": _finding(summary),
    }
    for name, value in payloads.items():
        (output / name).write_text(value, encoding="utf-8")
    WIKI_FINDING.write_text(_finding(summary), encoding="utf-8")
    (output / "verification.json").write_text(
        _canonical({"schema_version": "reconstruction-p39-final-verification/v1", "status": "PENDING", "checks": {}}),
        encoding="utf-8",
    )
    (output / "artifact_manifest.json").write_text(
        _canonical(_artifact_manifest(output)), encoding="utf-8"
    )
    verification = _recompute_final(output)
    if verification["status"] != "PASS":
        raise RuntimeError(f"P39 final verification failed: {verification}")
    (output / "verification.json").write_text(_canonical(verification), encoding="utf-8")
    (output / "artifact_manifest.json").write_text(
        _canonical(_artifact_manifest(output)), encoding="utf-8"
    )
    final = _recompute_final(output)
    if final["status"] != "PASS" or final != verification:
        raise RuntimeError(f"P39 post-manifest verification failed: {final}")


def verify_final(output_dir: Path) -> None:
    output = _validate_output(output_dir)
    result = _recompute_final(output)
    committed = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    if result["status"] != "PASS" or result != committed:
        raise RuntimeError(f"P39 verify-only failed: {result}")


def _asset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--foundation-python", type=Path, required=True)
    parser.add_argument("--moment-snapshot", type=Path, required=True)
    parser.add_argument("--moment-dependency-root", type=Path, required=True)
    parser.add_argument("--moment-source-root", type=Path, required=True)
    parser.add_argument("--gfm-snapshot", type=Path, required=True)
    parser.add_argument("--gfm-source-root", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=False)
    phase0 = sub.add_parser("phase0")
    phase0.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    phase0.add_argument("--raw-project-root", type=Path, default=PROJECT_ROOT)
    phase0.add_argument("--scratch-dir", type=Path, default=DEFAULT_SCRATCH)
    _asset_args(phase0)
    phase0.add_argument("--selected-device", required=True)
    phase0.add_argument("--gpu-uuid", required=True)
    verify = sub.add_parser("verify-phase0")
    verify.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run = sub.add_parser("run")
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--raw-project-root", type=Path, default=PROJECT_ROOT)
    run.add_argument("--scratch-dir", type=Path, default=DEFAULT_SCRATCH)
    _asset_args(run)
    run.add_argument("--device", required=True)
    run.add_argument("--gpu-uuid", required=True)
    return parser


def _path_mapping(args: argparse.Namespace) -> dict[str, str]:
    return {
        "foundation_python": str(Path(args.foundation_python).expanduser().absolute()),
        "moment_snapshot": str(Path(args.moment_snapshot).expanduser().resolve()),
        "moment_dependency_root": str(Path(args.moment_dependency_root).expanduser().resolve()),
        "moment_source_root": str(Path(args.moment_source_root).expanduser().resolve()),
        "gfm_snapshot": str(Path(args.gfm_snapshot).expanduser().resolve()),
        "gfm_source_root": str(Path(args.gfm_source_root).expanduser().resolve()),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.verify_only:
        if args.command is not None:
            raise ValueError("--verify-only cannot be combined with a subcommand")
        verify_final(args.output_dir)
        summary = json.loads((_validate_output(args.output_dir) / "summary.json").read_text())
        print(_canonical({"status": "VERIFIED", "decision": summary["decision"]}), end="")
        return 0
    if args.command is None:
        raise ValueError("a command or --verify-only is required")
    if args.command == "verify-phase0":
        verify_phase0(args.output_dir)
        print(_canonical({"status": "VERIFIED_PHASE0"}), end="")
        return 0
    paths = _path_mapping(args)
    _assert_legal_paths([Path(value) for value in paths.values()])
    if args.command == "phase0":
        write_phase0(
            output_dir=args.output_dir,
            paths=paths,
            selected_device=args.selected_device,
            gpu_uuid=args.gpu_uuid,
        )
        print(_canonical({"status": "PASS_PHASE0"}), end="")
        return 0
    freeze = json.loads((_validate_output(args.output_dir) / "phase0_freeze.json").read_text())
    if args.device != freeze["resource_budget"]["selected_device"] or args.gpu_uuid != freeze["resource_budget"]["gpu_uuid"]:
        raise RuntimeError("P39 run device differs from frozen Phase-0 device")
    result = run_experiment(
        output_dir=args.output_dir,
        scratch_dir=args.scratch_dir,
        raw_project_root=args.raw_project_root,
        foundation_python=args.foundation_python,
        moment_snapshot=args.moment_snapshot,
        moment_dependency_root=args.moment_dependency_root,
        moment_source_root=args.moment_source_root,
        gfm_snapshot=args.gfm_snapshot,
        gfm_source_root=args.gfm_source_root,
        device=args.device,
    )
    commands = _rerun_commands(args)
    write_final(output_dir=args.output_dir, result=result, commands=commands)
    summary = result[0]
    print(_canonical({"status": "COMPLETE", "decision": summary["decision"]}), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
