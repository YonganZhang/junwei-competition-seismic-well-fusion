#!/usr/bin/env python3
"""P6 private Gaia/DAGT evidence pack for the lithofacies track.

This module does not retrain models. It turns the already-verified LOGO4 /
fixed-nine development evidence into a reproducible private bundle, while using
the shared read-only Gaia/DAGT contract objects for provenance, agent mode
selection, and dry-run packaging.

The final research state is intentionally one of the five approved P6 states:
`verified_gain`, `foundation_gain_only`, `agent_signal_but_no_baseline_win`,
`no_verified_gain`, or `blocked_by_data`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _models.gaia_dagt import (  # noqa: E402
    DEFAULT_PROMPT_VERSION,
    DEFAULT_SOURCE_MANIFEST,
    GaiaDAGTAdapter,
    ModelBatch,
    ModelOutput,
    TrackSpec,
    agent_unavailable,
    supervisory_qc_agent,
)
from _models.gaia_dagt.contracts import canonical_json, sha256_json  # noqa: E402

OUTPUT_ROOT = TRACK_DIR / "_outputs" / "p6_gaia_dagt"
FIGURE_ROOT = OUTPUT_ROOT / "figures"

STAGE3_ROOT = TRACK_DIR / "_outputs" / "p5_stage3"
STAGE3_SUMMARY = STAGE3_ROOT / "p5_stage3_summary.json"
STAGE3_LEADERBOARD = STAGE3_ROOT / "p5_stage3_gm09_p_leaderboard.json"
STAGE3_RESULTS = STAGE3_ROOT / "p5_stage3_results.jsonl"
SPLIT_MANIFEST = TRACK_DIR / "_outputs" / "split_manifest.json"
NORMALIZATION_STATS = TRACK_DIR / "_outputs" / "normalization_stats.json"
COMPLETION_AUDIT = TRACK_DIR / "_outputs" / "completion_audit.json"
P5_SOURCE_LOCK = TRACK_DIR / "p5_source_lock.json"

ROOT_SEED = 2693
TASK_ID = "gm09_genetic_facies_9class"
TRACK_ID = "lithofacies"
ALLOWED_FINAL_STATES = (
    "verified_gain",
    "foundation_gain_only",
    "agent_signal_but_no_baseline_win",
    "no_verified_gain",
    "blocked_by_data",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _require_existing(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    for base in (PROJECT_ROOT.resolve(), TRACK_DIR.resolve(), OUTPUT_ROOT.resolve()):
        try:
            return resolved.relative_to(base).as_posix()
        except ValueError:
            continue
    return resolved.name


def _load_stage3_evidence() -> dict[str, Any]:
    summary = _read_json(_require_existing(STAGE3_SUMMARY))
    leaderboard = _read_json(_require_existing(STAGE3_LEADERBOARD))
    results = _read_jsonl(_require_existing(STAGE3_RESULTS))
    split_manifest = _read_json(_require_existing(SPLIT_MANIFEST))
    normalization_stats = _read_json(_require_existing(NORMALIZATION_STATS))
    completion_audit = _read_json(_require_existing(COMPLETION_AUDIT))
    source_lock = _read_json(_require_existing(P5_SOURCE_LOCK))
    return {
        "summary": summary,
        "leaderboard": leaderboard,
        "results": results,
        "split_manifest": split_manifest,
        "normalization_stats": normalization_stats,
        "completion_audit": completion_audit,
        "source_lock": source_lock,
    }


def _stage3_entries(leaderboard: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = list(leaderboard["entries"])
    entries.sort(key=lambda item: int(item["rank"]) if item["rank"] is not None else 99)
    return entries


def _baseline_table(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for entry in entries:
        table.append(
            {
                "model_id": str(entry["model_id"]),
                "rankable": str(entry["status"]) == "eligible",
                "completion_rate": float(entry["completion_rate"]),
                "fixed_schema_macro_f1_mean": float(entry["fixed_schema_macro_f1_mean"]),
                "fixed_schema_macro_f1_ci95": [float(x) for x in entry["fixed_schema_macro_f1_ci95"]],
                "worst_fold_fixed_schema_macro_f1": None
                if entry["worst_fold_fixed_schema_macro_f1"] is None
                else float(entry["worst_fold_fixed_schema_macro_f1"]),
                "supported_class_macro_f1_mean_diagnostic": float(entry["supported_class_macro_f1_mean_diagnostic"]),
                "seed_mean_std": float(entry["seed_mean_std"]),
                "seed_means": {str(k): float(v) for k, v in entry["seed_means"].items()},
                "fold_means": {str(k): float(v) for k, v in entry["fold_means"].items()},
                "wall_seconds_mean": float(entry["wall_seconds_mean"]),
                "wall_seconds_max": float(entry["wall_seconds_max"]),
            }
        )
    return table


def _best_ranked_baseline(entries: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [entry for entry in entries if entry["status"] == "eligible"]
    if not eligible:
        raise ValueError("no rankable development baseline is available")
    return max(eligible, key=lambda item: (float(item["fixed_schema_macro_f1_mean"]), float(item["worst_fold_fixed_schema_macro_f1"])))


def _research_state(evidence: Mapping[str, Any]) -> tuple[str, str]:
    summary = evidence["summary"]
    leaderboard = evidence["leaderboard"]
    entries = _stage3_entries(leaderboard)
    best = _best_ranked_baseline(entries)
    text_paths = evidence["agent"]["sample_level_text_paths"]
    predictive_available = bool(text_paths)
    if not predictive_available:
        state = "no_verified_gain"
        reason = (
            f"development winner remains {best['model_id']} at "
            f"fixed_schema_macro_f1_mean={float(best['fixed_schema_macro_f1_mean']):.6f}; "
            "no honest sample-level text exists, so predictive F2/C1/C2 remain unavailable; "
            "S lane stays not_feasible/not_rankable without finite center_md_m."
        )
    else:
        state = "agent_signal_but_no_baseline_win"
        reason = "sample-level text exists but the agent signal does not exceed the best baseline."
    if state not in ALLOWED_FINAL_STATES:
        raise ValueError(f"unexpected final research state: {state}")
    if summary["status"] not in {"ranked", "PASS", "CONFIRMATION_COMPLETE"}:
        reason = f"stage3 summary status={summary['status']} is not sufficient for a claim"
    return state, reason


def _build_track_spec(evidence: Mapping[str, Any]) -> TrackSpec:
    return TrackSpec(
        track_id="lithofacies_p6_gaia_dagt",
        task_type="classification",
        modality="offline_development_evidence",
        input_fields=("stage3_summary", "stage3_leaderboard", "split_manifest", "normalization_stats"),
        target_fields=("final_research_state",),
        allowed_paths=(
            "p5_stage3/p5_stage3_summary.json",
            "p5_stage3/p5_stage3_gm09_p_leaderboard.json",
            "p5_stage3/p5_stage3_results.jsonl",
            "split_manifest.json",
            "normalization_stats.json",
            "completion_audit.json",
        ),
        forbidden_paths=("test.h5", "frozen_test", "PRIVATE_WORKTREE_ROOT", "ABSOLUTE_DATA_ROOT"),
        metric_names=("fixed_schema_macro_f1", "balanced_accuracy"),
        base_seed=ROOT_SEED,
        prompt_version=DEFAULT_PROMPT_VERSION,
        source_manifest_digest=DEFAULT_SOURCE_MANIFEST.digest(),
        provenance={
            "track_id": TRACK_ID,
            "task_id": TASK_ID,
            "development_only": True,
            "frozen_test_accessed": False,
            "sample_level_text_paths": [],
            "source_lock_sha256": evidence["source_lock_sha256"],
        },
    )


def _build_agent_evidence(track_spec: TrackSpec, report_text: str) -> Any:
    return supervisory_qc_agent(
        report_text,
        source_manifest_digest=track_spec.source_manifest_digest,
        control_mode="real",
        seed=ROOT_SEED,
    )


def _build_gd_adapter(track_spec: TrackSpec) -> GaiaDAGTAdapter:
    return GaiaDAGTAdapter.from_default_manifest(track_spec, control_mode="real", seed=ROOT_SEED)


def _build_model_batch(track_spec: TrackSpec, evidence: Mapping[str, Any], agent_evidence: Any) -> ModelBatch:
    table = _baseline_table(_stage3_entries(evidence["leaderboard"]))
    features = [
        [
            row["fixed_schema_macro_f1_mean"],
            row["fixed_schema_macro_f1_ci95"][0],
            row["fixed_schema_macro_f1_ci95"][1],
            row["completion_rate"],
        ]
        for row in table
    ]
    # Target is a synthetic audit label, not a research score; the shared adapter
    # only uses it for its deterministic dry-run bookkeeping.
    target = [0, 0, 0]
    mask = [1, 1, 1]
    return ModelBatch(
        track_spec=track_spec,
        features=features,
        target=target,
        mask=mask,
        metadata={
            "bundle_kind": "lithofacies_p6_gaia_dagt",
            "final_states": ALLOWED_FINAL_STATES,
            "baseline_rows": len(table),
        },
        agent_evidence=agent_evidence,
    )


def _plot_development_comparison(evidence: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    leaderboard = evidence["leaderboard"]
    entries = _stage3_entries(leaderboard)
    selected = entries[:3]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ax = axes[0]
    xs = np.arange(len(selected))
    means = np.asarray([float(entry["fixed_schema_macro_f1_mean"]) for entry in selected], dtype=np.float64)
    lowers = np.asarray([float(entry["fixed_schema_macro_f1_ci95"][0]) for entry in selected], dtype=np.float64)
    uppers = np.asarray([float(entry["fixed_schema_macro_f1_ci95"][1]) for entry in selected], dtype=np.float64)
    yerr = np.vstack((means - lowers, uppers - means))
    colors = ["#2C7FB8" if entry["status"] == "eligible" else "#9E9E9E" for entry in selected]
    ax.bar(xs, means, color=colors, edgecolor="black", linewidth=0.8)
    ax.errorbar(xs, means, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.1, capsize=4)
    ax.set_xticks(xs, [entry["model_id"] for entry in selected], rotation=20, ha="right")
    ax.set_ylabel("fixed-schema Macro-F1 (unitless)")
    ax.set_xlabel("model family")
    ax.set_ylim(0.0, max(0.25, float(uppers.max()) + 0.03))
    ax.set_title("LOGO4 development comparison")
    for index, entry in enumerate(selected):
        ax.text(
            index,
            float(entry["fixed_schema_macro_f1_mean"]) + 0.008,
            f"seeds=3\ncomp={float(entry['completion_rate']):.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax2 = axes[1]
    xgboost = selected[0]
    fold_means = np.asarray([float(value) for value in xgboost["fold_means"].values()], dtype=np.float64)
    fold_ids = np.asarray([int(key) for key in xgboost["fold_means"].keys()], dtype=np.int64)
    ax2.plot(fold_ids, fold_means, marker="o", color="#1B9E77", linewidth=2.0)
    ax2.set_xlabel("LOGO4 fold id")
    ax2.set_ylabel("fixed-schema Macro-F1 (unitless)")
    ax2.set_title("Best baseline fold stability")
    ax2.set_xticks(fold_ids)
    ax2.set_ylim(0.0, max(0.25, float(fold_means.max()) + 0.03))
    ax2.grid(True, axis="y", alpha=0.3)

    caveat = textwrap.dedent(
        """
        Development-only evidence.
        No frozen test loader was opened.
        CatBoost is diagnostic-only here because completion = 0.75.
        S lane remains not_feasible because finite center_md_m = 0.
        Predictive text / F2 / C1 / C2 remain unavailable without honest sample-level text.
        """
    ).strip()
    fig.suptitle("Lithofacies P6 Gaia/DAGT: fixed-nine LOGO4 evidence", fontsize=14)
    fig.text(0.5, -0.03, caveat, ha="center", va="top", fontsize=9)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "development_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure_id": "development_comparison",
        "status": "PASS",
        "path": _portable_path(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "labels": {
            "x": "model family / fold id",
            "y": "fixed-schema Macro-F1 (unitless)",
            "caveat": "development-only; no frozen test access",
        },
    }


def _write_caption(output_dir: Path, final_state: str) -> Path:
    caption = textwrap.dedent(
        f"""
        # Lithofacies P6 development comparison

        This figure summarizes the real LOGO4 / fixed-nine development evidence only.
        The y-axis is fixed-schema Macro-F1, which is unitless.

        Caveats:

        - no frozen test loader was opened;
        - the CatBoost cell is diagnostic-only because completion is 0.75;
        - S lane remains not_feasible / not_rankable because finite center_md_m is 0;
        - predictive text, F2, C1, and C2 remain unavailable because no honest sample-level text path exists.

        Final research state: `{final_state}`.
        """
    ).strip() + "\n"
    path = output_dir / "figure_caption.md"
    _atomic_text(path, caption)
    return path


def build_bundle(*, output_dir: Path = OUTPUT_ROOT) -> dict[str, Any]:
    evidence = _load_stage3_evidence()
    evidence["source_lock_sha256"] = _sha256(P5_SOURCE_LOCK)
    evidence["stage3_summary_sha256"] = _sha256(STAGE3_SUMMARY)
    evidence["stage3_leaderboard_sha256"] = _sha256(STAGE3_LEADERBOARD)
    evidence["stage3_results_sha256"] = _sha256(STAGE3_RESULTS)
    evidence["split_manifest_sha256"] = _sha256(SPLIT_MANIFEST)
    evidence["normalization_stats_sha256"] = _sha256(NORMALIZATION_STATS)
    evidence["completion_audit_sha256"] = _sha256(COMPLETION_AUDIT)

    state, reason = _research_state(
        {
            **evidence,
            "agent": {"sample_level_text_paths": []},
        }
    )
    track_spec = _build_track_spec(evidence)
    report_text = (
        "Lithofacies P6 Gaia/DAGT QC report: development-only LOGO4 evidence, "
        "fixed-nine schema, three seeds where valid, no sample-level text path, "
        f"final state candidate={state}."
    )
    agent_evidence = _build_agent_evidence(track_spec, report_text)
    adapter = _build_gd_adapter(track_spec)
    batch = _build_model_batch(track_spec, evidence, agent_evidence)
    dry_run = adapter.dry_run(report_text, batch, mode="supervisory_qc_agent")
    figures_dir = output_dir / "figures"
    figure_manifest = _plot_development_comparison(evidence, figures_dir)
    caption_path = _write_caption(output_dir, state)

    bundle = {
        "schema_version": "lithofacies-p6-gaia-dagt-v1",
        "track_id": TRACK_ID,
        "task_id": TASK_ID,
        "final_research_state": state,
        "final_state_reason": reason,
        "allowed_states": list(ALLOWED_FINAL_STATES),
        "agent_mode": agent_evidence.agent_mode,
        "predictive_text_available": False,
        "f2_c1_c2_available": False,
        "sample_level_text_paths": [],
        "comparisons": _baseline_table(_stage3_entries(evidence["leaderboard"])),
        "ranked_best_baseline": {
            "model_id": _best_ranked_baseline(_stage3_entries(evidence["leaderboard"]))["model_id"],
            "fixed_schema_macro_f1_mean": float(_best_ranked_baseline(_stage3_entries(evidence["leaderboard"]))["fixed_schema_macro_f1_mean"]),
            "worst_fold_fixed_schema_macro_f1": float(_best_ranked_baseline(_stage3_entries(evidence["leaderboard"]))["worst_fold_fixed_schema_macro_f1"]),
        },
        "source_hashes": {
            "stage3_summary": evidence["stage3_summary_sha256"],
            "stage3_leaderboard": evidence["stage3_leaderboard_sha256"],
            "stage3_results": evidence["stage3_results_sha256"],
            "split_manifest": evidence["split_manifest_sha256"],
            "normalization_stats": evidence["normalization_stats_sha256"],
            "completion_audit": evidence["completion_audit_sha256"],
            "source_lock": evidence["source_lock_sha256"],
        },
        "resource_hashes": {
            "track_spec_cache_key": track_spec.cache_key(),
            "gaia_source_manifest_digest": track_spec.source_manifest_digest,
            "dry_run_svg_sha256": _sha256_text(dry_run.svg),
            "figure_sha256": figure_manifest["sha256"],
            "figure_caption_sha256": _sha256(caption_path),
        },
        "failure_hashes": {
            "no_sample_level_text": sha256_json(
                {
                    "agent_mode": agent_evidence.agent_mode,
                    "reason": "predictive_text_agent unavailable because no honest sample-level text path exists",
                    "sample_level_text_paths": [],
                    "f2_c1_c2_available": False,
                    "s_lane": "not_feasible",
                }
            ),
            "catboost_not_rankable": sha256_json(
                {
                    "model_id": "catboost_multiclass_window",
                    "completion_rate": 0.75,
                    "status": "not_rankable",
                }
            ),
        },
        "provenance_hashes": {
            "bundle_sha256": sha256_json(
                {
                    "state": state,
                    "reason": reason,
                    "comparisons": "pending",
                }
            )
        },
        "gaia_contract": {
            "track_spec": track_spec.to_dict(),
            "track_spec_cache_key": track_spec.cache_key(),
            "agent_evidence": agent_evidence.to_dict(),
            "dry_run": {
                "metric": dry_run.metric,
                "svg": dry_run.svg,
                "svg_sha256": _sha256_text(dry_run.svg),
                "batch": batch.to_dict(),
                "output": dry_run.output.to_dict(),
            },
        },
        "visualization_manifest": {
            "schema_version": "lithofacies-p6-gaia-dagt-visualization-manifest-v1",
            "figures": [figure_manifest],
            "caption_path": _portable_path(caption_path),
            "caveats": [
                "development-only LOGO4 evidence",
                "no frozen test access",
                "fixed-nine macro-F1 is unitless",
                "S lane not_feasible because finite center_md_m = 0",
            ],
        },
    }
    # The provenance hash above must be computed from the fully materialized bundle.
    bundle["provenance_hashes"]["bundle_sha256"] = sha256_json(
        {
            "final_research_state": bundle["final_research_state"],
            "comparisons": bundle["comparisons"],
            "source_hashes": bundle["source_hashes"],
            "resource_hashes": bundle["resource_hashes"],
            "failure_hashes": bundle["failure_hashes"],
            "gaia_contract": {
                "track_spec_cache_key": bundle["gaia_contract"]["track_spec_cache_key"],
                "agent_evidence_cache_key": bundle["gaia_contract"]["agent_evidence"]["cache_key"],
                "dry_run_svg_sha256": bundle["gaia_contract"]["dry_run"]["svg_sha256"],
            },
        }
    )
    return bundle


def write_bundle(*, output_dir: Path = OUTPUT_ROOT) -> dict[str, Any]:
    bundle = build_bundle(output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_manifest = bundle["visualization_manifest"]["figures"][0]
    report_path = output_dir / "summary.json"
    provenance_path = output_dir / "provenance.json"
    state_path = output_dir / "research_state.json"
    agent_path = output_dir / "gaia_agent_evidence.json"
    contract_path = output_dir / "gaia_contract.json"
    dry_run_svg_path = output_dir / "figures" / "gaia_dagt_dry_run.svg"
    _atomic_json(report_path, bundle)
    _atomic_json(
        provenance_path,
        {
            "schema_version": bundle["schema_version"],
            "source_hashes": bundle["source_hashes"],
            "resource_hashes": bundle["resource_hashes"],
            "failure_hashes": bundle["failure_hashes"],
            "provenance_hashes": bundle["provenance_hashes"],
        },
    )
    _atomic_json(
        state_path,
        {
            "schema_version": "lithofacies-p6-gaia-dagt-state-v1",
            "final_research_state": bundle["final_research_state"],
            "reason": bundle["final_state_reason"],
            "allowed_states": bundle["allowed_states"],
            "agent_mode": bundle["agent_mode"],
            "predictive_text_available": bundle["predictive_text_available"],
            "f2_c1_c2_available": bundle["f2_c1_c2_available"],
            "source_hashes": bundle["source_hashes"],
            "resource_hashes": bundle["resource_hashes"],
            "failure_hashes": bundle["failure_hashes"],
            "provenance_hashes": bundle["provenance_hashes"],
        },
    )
    _atomic_json(agent_path, bundle["gaia_contract"]["agent_evidence"])
    _atomic_json(contract_path, bundle["gaia_contract"])
    _atomic_text(dry_run_svg_path, bundle["gaia_contract"]["dry_run"]["svg"])
    _atomic_json(output_dir / "visualization_manifest.json", bundle["visualization_manifest"])
    return {
        "summary_path": _portable_path(report_path),
        "provenance_path": _portable_path(provenance_path),
        "research_state_path": _portable_path(state_path),
        "agent_evidence_path": _portable_path(agent_path),
        "gaia_contract_path": _portable_path(contract_path),
        "visualization_manifest_path": _portable_path(output_dir / "visualization_manifest.json"),
        "figure_path": fig_manifest["path"],
        "figure_sha256": fig_manifest["sha256"],
        "final_research_state": bundle["final_research_state"],
    }


def verify_bundle(*, output_dir: Path = OUTPUT_ROOT) -> dict[str, Any]:
    summary = _read_json(_require_existing(output_dir / "summary.json"))
    provenance = _read_json(_require_existing(output_dir / "provenance.json"))
    state = _read_json(_require_existing(output_dir / "research_state.json"))
    manifest = _read_json(_require_existing(output_dir / "visualization_manifest.json"))
    figure = output_dir / "figures" / "development_comparison.png"
    caption = output_dir / "figure_caption.md"
    dry_run_svg = output_dir / "figures" / "gaia_dagt_dry_run.svg"
    checks = {
        "state": state["final_research_state"] in ALLOWED_FINAL_STATES,
        "summary_state_match": summary["final_research_state"] == state["final_research_state"],
        "figure_exists": figure.is_file(),
        "figure_hash_match": manifest["figures"][0]["sha256"] == _sha256(figure),
        "caption_exists": caption.is_file(),
        "dry_run_svg_exists": dry_run_svg.is_file(),
        "provenance_hash_match": provenance["provenance_hashes"]["bundle_sha256"] == summary["provenance_hashes"]["bundle_sha256"],
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "summary_sha256": _sha256(output_dir / "summary.json"),
        "provenance_sha256": _sha256(output_dir / "provenance.json"),
        "state_sha256": _sha256(output_dir / "research_state.json"),
        "manifest_sha256": _sha256(output_dir / "visualization_manifest.json"),
        "final_research_state": state["final_research_state"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="materialize the private Gaia/DAGT evidence bundle")
    build.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)

    verify = subparsers.add_parser("verify", help="verify the private Gaia/DAGT evidence bundle")
    verify.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        payload = write_bundle(output_dir=args.output_dir)
    elif args.command == "verify":
        payload = verify_bundle(output_dir=args.output_dir)
    else:  # pragma: no cover - argparse prevents this
        raise SystemExit(2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
