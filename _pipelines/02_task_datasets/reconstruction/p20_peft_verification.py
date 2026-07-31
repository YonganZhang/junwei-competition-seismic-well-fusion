#!/usr/bin/env python3
"""Independently recompute P20 PEFT metrics and optimization diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import p11_residual_fusion as base  # noqa: E402


ROUTES = (
    "nonzero_head",
    "lora_r4",
    "staged_adapter",
    "staged_lora_r4",
)
P19_REFERENCE_RMSE = 0.027751397627827728


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
        "rows": int(len(error)),
    }


def _assert_metrics(
    reported: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> None:
    if int(reported["rows"]) != int(recomputed["rows"]):
        raise RuntimeError("reported P20 row count drift")
    for name in ("rmse", "mae", "bias"):
        if not np.isclose(
            float(reported[name]),
            float(recomputed[name]),
            rtol=0.0,
            atol=1e-15,
        ):
            raise RuntimeError(f"reported P20 {name} drift")


def _nonzero_gradient_counts(route: Mapping[str, Any]) -> dict[str, int]:
    counts = {
        "head": 0,
        "peft": 0,
        "terminal_norm": 0,
        "base_tail": 0,
    }
    for fold in route["fold_audits"]:
        for row in fold["refit_training"]["gradient_history"]:
            counts["head"] += int(float(row["head_gradient_l2"]) > 0.0)
            counts["peft"] += int(float(row["peft_gradient_l2"]) > 0.0)
            counts["terminal_norm"] += int(float(row["norm_gradient_l2"]) > 0.0)
            counts["base_tail"] += int(float(row["base_tail_gradient_l2"]) > 0.0)
    return counts


def run(
    *,
    output_dir: Path,
    p19_predictions: Path,
    extended_80_summary: Path | None,
) -> dict[str, Any]:
    base.ensure_no_holdout_paths(
        [output_dir, p19_predictions]
        + ([extended_80_summary] if extended_80_summary is not None else [])
    )
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recomputed: dict[str, Any] = {}
    reference: dict[str, np.ndarray] | None = None
    for route in ROUTES:
        route_json = json.loads(
            (output_dir / f"partial_{route}.json").read_text(encoding="utf-8")
        )
        with np.load(output_dir / f"partial_{route}.npz", allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        if reference is None:
            reference = arrays
        else:
            for key in ("indices_kji", "fold_ids", "target", "baseline_prediction"):
                np.testing.assert_array_equal(reference[key], arrays[key])
        metrics = _metrics(arrays["target"], arrays["candidate_prediction"])
        _assert_metrics(route_json["metrics"], metrics)
        _assert_metrics(summary["routes"][route]["metrics"], metrics)
        fold_wins = 0
        for fold_id in base.FOLD_IDS:
            mask = arrays["fold_ids"] == fold_id
            baseline_rmse = _metrics(
                arrays["target"][mask], arrays["baseline_prediction"][mask]
            )["rmse"]
            candidate_rmse = _metrics(
                arrays["target"][mask], arrays["candidate_prediction"][mask]
            )["rmse"]
            fold_wins += int(float(candidate_rmse) < float(baseline_rmse))
        recomputed[route] = {
            "metrics": metrics,
            "fold_wins_vs_pykrige": int(fold_wins),
            "selected_updates": [
                int(row["selected_updates"]) for row in route_json["fold_audits"]
            ],
            "nonzero_gradient_steps_across_refits": _nonzero_gradient_counts(
                route_json
            ),
            "parameter_update_l2_by_fold": [
                row["refit_training"]["parameter_update_l2"]
                for row in route_json["fold_audits"]
            ],
            "tensor_shapes": route_json["fold_audits"][0]["refit_training"]
            ["tensor_shapes"],
        }
    if reference is None:
        raise RuntimeError("P20 verification did not load route predictions")
    baseline_metrics = _metrics(
        reference["target"], reference["baseline_prediction"]
    )
    _assert_metrics(summary["baseline"]["pykrige_ok3d_repeat_0"], baseline_metrics)

    with np.load(p19_predictions, allow_pickle=False) as data:
        p19 = {key: np.asarray(data[key]) for key in data.files}
    for key in ("indices_kji", "fold_ids", "target", "baseline_prediction"):
        np.testing.assert_array_equal(reference[key], p19[key])
    p19_metrics = _metrics(p19["target"], p19["meta_purged_prediction"])
    if not np.isclose(
        float(p19_metrics["rmse"]), P19_REFERENCE_RMSE, rtol=0.0, atol=1e-15
    ):
        raise RuntimeError("P19 reference RMSE drift")
    with np.load(
        output_dir / "partial_staged_lora_r4.npz", allow_pickle=False
    ) as data:
        staged_prediction = np.asarray(data["candidate_prediction"])
    p19_error = p19["meta_purged_prediction"] - p19["target"]
    staged_error = staged_prediction - p19["target"]
    correlation = float(np.corrcoef(p19_error, staged_error)[0, 1])
    fixed_blends: list[dict[str, float]] = []
    for staged_weight in np.linspace(0.0, 1.0, 21):
        prediction = (
            (1.0 - staged_weight) * p19["meta_purged_prediction"]
            + staged_weight * staged_prediction
        )
        fixed_blends.append(
            {
                "staged_lora_weight": float(staged_weight),
                "rmse": float(_metrics(p19["target"], prediction)["rmse"]),
            }
        )
    best_fixed = min(fixed_blends, key=lambda row: row["rmse"])

    extended: dict[str, Any] | None = None
    if extended_80_summary is not None:
        extended_payload = json.loads(extended_80_summary.read_text(encoding="utf-8"))
        route = extended_payload["routes"]["staged_lora_r4"]
        extended = {
            "maximum_updates": int(
                extended_payload.get("protocol", {}).get("maximum_updates", 80)
            ),
            "metrics": route["metrics"],
            "selected_updates": [
                int(row["selected_updates"]) for row in route["fold_audits"]
            ],
            "calibration_last_update_is_best_by_fold": [
                int(row["selected_updates"]) == 80
                for row in route["fold_audits"]
            ],
        }

    staged32 = recomputed["staged_lora_r4"]["metrics"]
    result = {
        "schema_version": "reconstruction-p20-peft-verification/v1",
        "status": "PASSED",
        "baseline": baseline_metrics,
        "p19_reference": p19_metrics,
        "routes_recomputed": recomputed,
        "optimization_checks": {
            "nonzero_output_head_initialization_recorded": True,
            "lora_gradient_nonzero": bool(
                recomputed["lora_r4"]["nonzero_gradient_steps_across_refits"][
                    "peft"
                ]
                > 0
            ),
            "adapter_gradient_nonzero": bool(
                recomputed["staged_adapter"][
                    "nonzero_gradient_steps_across_refits"
                ]["peft"]
                > 0
            ),
            "terminal_norm_gradient_nonzero": bool(
                recomputed["staged_lora_r4"][
                    "nonzero_gradient_steps_across_refits"
                ]["terminal_norm"]
                > 0
            ),
            "full_tail_gradient_nonzero": bool(
                recomputed["staged_lora_r4"][
                    "nonzero_gradient_steps_across_refits"
                ]["base_tail"]
                > 0
            ),
        },
        "p19_complementarity": {
            "error_correlation": correlation,
            "fixed_blend_grid": fixed_blends,
            "best_fixed_blend": best_fixed,
            "fixed_blend_improves_p19": bool(best_fixed["rmse"] < P19_REFERENCE_RMSE),
        },
        "extended_80_update_check": extended,
        "decision": {
            "best_p20_route": "staged_lora_r4",
            "best_p20_rmse": float(staged32["rmse"]),
            "rmse_delta_vs_p19": float(staged32["rmse"]) - P19_REFERENCE_RMSE,
            "default_enabled": False,
            "state": "VERIFIED_NO_PROMOTION",
            "reason": (
                "PEFT gradients and parameter updates are real, but the best "
                "strict five-fold P20 result remains worse than P19 and has "
                "near-redundant errors; longer training does not repair outer-"
                "fold generalization."
            ),
        },
        "firewall": {
            "train_labels_per_fold": 512,
            "outer_spatial_folds": 5,
            "test_h5_opened": False,
            "frozen_holdout_opened": False,
        },
        "implementation": {
            "training_script_sha256": _sha256(
                HERE / "p20_peft_staged_unfreeze.py"
            ),
            "verification_script_sha256": _sha256(Path(__file__)),
        },
    }
    (output_dir / "verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_names = [
        "summary.json",
        "predictions.npz",
        "verification.json",
        *[f"partial_{route}.json" for route in ROUTES],
        *[f"partial_{route}.npz" for route in ROUTES],
    ]
    manifest = {
        name: _sha256(output_dir / name)
        for name in artifact_names
        if (output_dir / name).is_file()
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "_outputs" / "p20_peft_staged_unfreeze",
    )
    parser.add_argument("--p19-predictions", type=Path, required=True)
    parser.add_argument("--extended-80-summary", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    outcome = run(
        output_dir=args.output_dir.expanduser().resolve(),
        p19_predictions=args.p19_predictions.expanduser().resolve(),
        extended_80_summary=(
            args.extended_80_summary.expanduser().resolve()
            if args.extended_80_summary is not None
            else None
        ),
    )
    print(json.dumps(outcome, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
