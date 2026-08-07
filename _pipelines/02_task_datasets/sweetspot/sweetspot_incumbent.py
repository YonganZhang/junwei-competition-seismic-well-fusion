"""Single source of truth for what currently wins on the sweetspot track.

The track runs three layers — small models (P5), a time-series foundation model
(P7/P8), and an LLM action-selection agent (P28/P29). Before this runner existed
the pipeline adapter only referenced the agent layer, so `verify` could pass
green while the actual best result (Chronos-2, a 30% MAE reduction) sat outside
the pipeline entirely. Anyone opening the repo fresh had no single place to read
what currently wins and which routes are already closed.

This module reads the archived evidence of all three layers, decides the current
incumbent per target, and writes `_outputs/incumbent/incumbent.json`. It trains
nothing, opens no frozen test artifact, and depends only on the standard library
so it runs under any interpreter.

Read incumbent.json first. `rejected_routes` lists what has already been tried
and refuted — do not restart those without new evidence.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "sweetspot-incumbent/v1"
TRACK_ID = "sweetspot"
HERE = Path(__file__).resolve().parent


def _git_output(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def _checkout_root() -> Path:
    return Path(_git_output(["rev-parse", "--show-toplevel"], HERE)).resolve()


ROOT = _checkout_root()
TRACK_DIR = ROOT / "_pipelines/02_task_datasets/sweetspot"
OUTPUT_DIR = TRACK_DIR / "_outputs" / "incumbent"

P5_LEADERBOARD_DIR = TRACK_DIR / "p5/_outputs/stage3_cv/leaderboards"
P7_SUMMARY = TRACK_DIR / "p7/_outputs/t3_chronos2_cv/summary.json"
P8_SUMMARY = TRACK_DIR / "p8/_outputs/t3_chronos2_calendar_cv/summary.json"
P28_SUMMARY = TRACK_DIR / "_outputs/p28_agentic_optimization/summary.json"
P29_SUMMARY = TRACK_DIR / "_outputs/p29_agent_action_effect/summary.json"
P6_CONCLUSION = TRACK_DIR / "p6/_outputs/private_gaia_dagt/p6_gaia_dagt_conclusion.json"

TARGET_NAMES = {
    "T1": "reservoir_quality",
    "T2": "hydrocarbon_pay",
    "T3": "productivity",
    "T4": "water_breakthrough",
    "T5": "remaining_oil_infill",
    "T6": "porosity",
    "T7": "permeability",
}

# Targets whose labels are CPI interpretation products rather than field truth.
# P44 showed RHOB alone reaches R2=0.9696 on T6, so model comparison on these
# targets measures how well an analytical relation is reproduced, not skill.
INTERPRETATION_PRODUCT_TARGETS = {"T1", "T2", "T6", "T7"}


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _small_model_layer() -> dict[str, Any]:
    """P5 Stage-3 multi-model cross-validation leaderboards."""
    targets: dict[str, Any] = {}
    for tid, slug in TARGET_NAMES.items():
        board = _load(P5_LEADERBOARD_DIR / f"{tid}.json")
        if board is None:
            targets[tid] = {"slug": slug, "status": "absent"}
            continue
        entry: dict[str, Any] = {
            "slug": slug,
            "status": board.get("status"),
            "label_is_interpretation_product": tid in INTERPRETATION_PRODUCT_TARGETS,
        }
        if board.get("status") == "rankable":
            entries = board.get("entries") or []
            if entries:
                best = min(entries, key=lambda e: e.get("rank", 1 << 30))
                entry.update({
                    "best_model": best.get("model_id"),
                    "primary_metric": board.get("metric") or board.get("primary_metric"),
                    "direction": board.get("direction"),
                    "primary_mean": best.get("primary_mean"),
                    "bootstrap_95ci": best.get("primary_bootstrap_95ci"),
                    "worst_fold": best.get("worst_fold"),
                })
        else:
            entry["reason"] = board.get("reason")
        targets[tid] = entry
    return {"source": _rel(P5_LEADERBOARD_DIR), "targets": targets}


def _foundation_layer() -> dict[str, Any]:
    """P7/P8 Chronos-2 time-series foundation lane."""
    layer: dict[str, Any] = {}
    p7 = _load(P7_SUMMARY)
    if p7 is not None:
        decision = p7.get("decision", {})
        foundation = p7.get("foundation", {})
        layer["p7_t3"] = {
            "source": _rel(P7_SUMMARY),
            "model_id": foundation.get("model_id"),
            "real_pretrained_weights_loaded": foundation.get("real_pretrained_weights_loaded"),
            "promotion_status": decision.get("promotion_status"),
            "selected_method": decision.get("selected_method"),
            "selected_macro_fold_mae": decision.get("selected_macro_fold_mae"),
            "archived_xgboost_macro_fold_mae": decision.get("archived_xgboost_macro_fold_mae"),
            "causal_history_mean_macro_fold_mae": decision.get("causal_history_mean_macro_fold_mae"),
            "mae_reduction_vs_archived_xgboost_percent": decision.get("mae_reduction_vs_archived_xgboost_percent"),
            "mae_reduction_vs_history_mean_percent": decision.get("mae_reduction_vs_history_mean_percent"),
            "t4_status": decision.get("t4_status"),
        }
    p8 = _load(P8_SUMMARY)
    if p8 is not None:
        decision = p8.get("decision", {})
        layer["p8_t3_calendar"] = {
            "source": _rel(P8_SUMMARY),
            "state": decision.get("state"),
            "effect_supported": decision.get("effect_supported"),
            "default_enabled": decision.get("default_enabled"),
            "blocking_control": decision.get("reason"),
        }
    p6 = _load(P6_CONCLUSION)
    if p6 is not None:
        layer["p6_private_gaia_dagt"] = {
            "source": _rel(P6_CONCLUSION),
            "status": p6.get("status"),
            "note": (
                "Private geoscience foundation lane. Not trained, no API calls, no downloads — it contributes "
                "no prediction on this track. Listed so its presence is not mistaken for an active model."
            ),
        }
    return layer


def _agent_layer() -> dict[str, Any]:
    """P28/P29 LLM action-selection agent."""
    layer: dict[str, Any] = {}
    for key, path in (("p28", P28_SUMMARY), ("p29", P29_SUMMARY)):
        summary = _load(path)
        if summary is None:
            continue
        a2l = summary.get("a2l") or {}
        layer[key] = {
            "source": _rel(path),
            "verdict": summary.get("verdict"),
            "retain_llm": summary.get("retain_llm"),
            "a2l_status": a2l.get("status"),
            "baseline_kind": summary.get("baseline_kind"),
            "a0_action_id": summary.get("a0_action_id"),
        }
    return layer


def _resolve_incumbents(small: dict[str, Any], foundation: dict[str, Any]) -> dict[str, Any]:
    """Pick the current best method per target across all three layers."""
    incumbents: dict[str, Any] = {}
    for tid, slug in TARGET_NAMES.items():
        board = small["targets"].get(tid, {})
        if board.get("status") != "rankable":
            incumbents[tid] = {
                "slug": slug,
                "incumbent": None,
                "reason": board.get("reason") or board.get("status"),
            }
            continue
        record = {
            "slug": slug,
            "incumbent": board.get("best_model"),
            "layer": "small_model",
            "metric": board.get("primary_metric"),
            "value": board.get("primary_mean"),
            "evidence": small["source"] + f"/{tid}.json",
        }
        # The foundation lane only contests T3, and only when it formally promoted.
        p7 = foundation.get("p7_t3") or {}
        if tid == "T3" and p7.get("promotion_status") == "PROMOTE":
            record = {
                "slug": slug,
                "incumbent": p7.get("selected_method"),
                "layer": "foundation_model",
                "model_id": p7.get("model_id"),
                "metric": "mae",
                "value": p7.get("selected_macro_fold_mae"),
                "superseded": {
                    "method": board.get("best_model"),
                    "value": board.get("primary_mean"),
                    "improvement_percent": p7.get("mae_reduction_vs_archived_xgboost_percent"),
                },
                "evidence": p7.get("source"),
            }
        if tid in INTERPRETATION_PRODUCT_TARGETS:
            record["label_caveat"] = (
                "label is a CPI interpretation product; a higher score reflects reproducing an "
                "analytical relation, not predictive skill (see P44)"
            )
        incumbents[tid] = record
    return incumbents


def _rejected_routes(agent: dict[str, Any], foundation: dict[str, Any]) -> list[dict[str, Any]]:
    """Routes already tried and refuted. Do not restart without new evidence."""
    routes: list[dict[str, Any]] = []
    p28, p29 = agent.get("p28") or {}, agent.get("p29") or {}
    if p28:
        routes.append({
            "route": "LLM directly selecting T3 XGBoost hyperparameter actions (P28)",
            "outcome": p28.get("verdict"),
            "detail": "DeepSeek issued a legal stop; no candidate beat the frozen A0 baseline.",
            "evidence": p28.get("source"),
            "restart_condition": "Only with a different incumbent to optimise against, or a target with lower fold variance.",
        })
    if p29:
        routes.append({
            "route": "LLM action selection against a same-fold same-executor A0 baseline (P29)",
            "outcome": p29.get("verdict"),
            "detail": (
                "Feedback-baseline bug fixed and rerun; all candidate deltas were non-positive. "
                "Note the A0 baseline is the archived XGBoost, which the foundation lane has since superseded."
            ),
            "evidence": p29.get("source"),
            "restart_condition": "Re-run only after A0 is switched to the current incumbent (see open_work).",
        })
    p7 = foundation.get("p7_t3") or {}
    if p7.get("t4_status"):
        routes.append({
            "route": "Chronos-2 direct forecasting for T4 water breakthrough",
            "outcome": p7.get("t4_status"),
            "detail": "Foundation average precision fell below the archived CatBoost leader; not promoted.",
            "evidence": p7.get("source"),
            "restart_condition": "Requires more water-breakthrough events, not a different model.",
        })
    routes.append({
        "route": "Model optimisation on T1/T2/T6/T7",
        "outcome": "NOT_MEANINGFUL",
        "detail": (
            "Labels are CPI interpretation products. RHOB alone reaches R2=0.9696 on T6 against "
            "0.9709 for all sixteen curves, so there is no unknown quantity left to learn."
        ),
        "evidence": "_wiki-methodology/_top/_findings/P44_sweetspot_label_provenance_collapse.md",
        "restart_condition": "Only if a field-truth label replaces the interpretation-derived one.",
    })
    return routes


# Rulings made by the project owner on 2026-08-07, recorded here rather than by
# editing the frozen P4 artifacts. label_mapping.v1.json and the T6/T7 split
# manifests each have their sha256 quoted by archived protocol/manifest files
# (5 and 7 referencing files respectively), so editing them would break the
# provenance chain of evidence that is already on record. Current truth lives
# here; history stays intact.
USER_DECISIONS = [
    {
        "id": "D1_t5_simulation_labels",
        "ruling": "Simulation output is accepted as a legitimate label source for T5 remaining-oil/infill.",
        "supersedes": "Stage-3 currently refuses T5 with 'no label is defined; simulation case must not be presented as field truth'.",
        "status": "ACCEPTED_NOT_YET_IMPLEMENTED",
        "implementation_note": (
            "T5 has no data pipeline at all (completed_cells 0/0). Honouring this ruling means building "
            "the target end to end — locating the Eclipse/RMS dynamic saturation volumes, defining the "
            "label, building a leak-free spatial split, then training. It is not a gate-flag change. "
            "The gate must also keep simulation-derived results labelled as such rather than as field truth."
        ),
    },
    {
        "id": "D2_t4_continue",
        "ruling": "Keep investing in T4 water-breakthrough rather than downgrading it.",
        "status": "ACCEPTED_NOT_YET_IMPLEMENTED",
        "implementation_note": (
            "T4's problem is event scarcity, not model choice: CatBoost macro AP 0.654 but the 95% CI runs "
            "[0.396, 0.867] and the worst fold is 0.143. Chronos-2 direct forecasting was already tried and "
            "rejected (AP 0.509). The actionable step is more breakthrough events or a longer causal window, "
            "not another architecture."
        ),
    },
    {
        "id": "D3_remove_well_15_9_19",
        "ruling": "Drop well family 15/9-19 from T6/T7 rather than sourcing its raw curves.",
        "status": "ACCEPTED_RECORDED",
        "implementation_note": (
            "The frozen T6/T7 split manifests still name it and are left untouched — their sha256 is quoted "
            "by seven archived files each. In practice the well never participated: it has three CPI tables "
            "carrying PHIF+KLOGH but no companion raw curves among its four authorized members, so passing "
            "three or four families to the loader yields an identical table set. Any future split must not "
            "include it."
        ),
    },
    {
        "id": "D4_t6_t7_split_rework",
        "ruling": "Originally approved, then withdrawn on the same day after P44.",
        "status": "WITHDRAWN",
        "implementation_note": (
            "The rework was approved on the premise that T6/T7 were untrainable. They were not — both "
            "completed P4 and passed frozen test (T6 R2=0.93411). P44 then showed RHOB alone reaches "
            "R2=0.9696 on T6, so comparing models on these targets measures reproduction of an analytical "
            "relation. Reworking the split would have bought a new, incomparable baseline for a target with "
            "nothing left to learn. Not doing it is the correct outcome."
        ),
    },
]

# Declared in the frozen T6/T7 split manifests but absent from every rebuilt table set.
DATA_NOTES = [
    {
        "subject": "well family 15/9-19",
        "note": (
            "Declared in the T6/T7 development_groups but carries no raw log curves in the authorized "
            "archive members, so it cannot enter a RAW_LOG_FEATURES matrix. Measured: passing three or four "
            "families to _load_development_petrophysical_tables returns the same tables. See decision D3."
        ),
    },
]


def _open_work(foundation: dict[str, Any]) -> list[dict[str, Any]]:
    work: list[dict[str, Any]] = []
    p8 = foundation.get("p8_t3_calendar") or {}
    if p8.get("state") == "EFFECT_SUPPORTED_NOT_PROMOTED":
        work.append({
            "item": "Add the same-architecture random-init control for P8",
            "why": (
                "P8 already beats the causal mean, same-grid ExtraTrees, target-shuffle and "
                "history-order-shuffle controls. The remaining control separates the value of the "
                "pretrained weights from the value of the architecture — it is the decisive test of "
                "whether foundation pretraining helps here."
            ),
            "blocks": "P8 default promotion",
        })
    work.append({
        "item": "Switch the P28/P29 agent A0 baseline to the current T3 incumbent",
        "why": (
            "The agent currently optimises against the archived XGBoost, which the foundation lane "
            "superseded by 30%. Its rejection verdicts are therefore measured against a stale frontier."
        ),
        "blocks": "Any meaningful agent-layer result on T3",
    })
    return work


def build_incumbent_record() -> dict[str, Any]:
    small = _small_model_layer()
    foundation = _foundation_layer()
    agent = _agent_layer()
    return {
        "schema_version": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "generation_base_commit": _git_output(["rev-parse", "HEAD"], ROOT),
        "read_this_first": (
            "This file is the sweetspot track's single source of truth. `incumbents` is what "
            "currently wins per target, `rejected_routes` is what has already been refuted, "
            "`user_decisions` is what the project owner has ruled, and `open_work` is what is worth "
            "doing next. Do not restart a rejected route without new evidence. Frozen P4 artifacts are "
            "never edited to reflect a later ruling — their sha256 is quoted by archived evidence."
        ),
        "layers": {
            "small_model": small,
            "foundation_model": foundation,
            "agent": agent,
        },
        "incumbents": _resolve_incumbents(small, foundation),
        "rejected_routes": _rejected_routes(agent, foundation),
        "user_decisions": USER_DECISIONS,
        "data_notes": DATA_NOTES,
        "open_work": _open_work(foundation),
    }


def generate(output_dir: Path | None = None) -> dict[str, Any]:
    record = build_incumbent_record()
    target_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (target_dir / "incumbent.json").write_text(payload, encoding="utf-8")
    return record


def main() -> None:
    record = generate()
    print(f"track: {record['track_id']}  schema: {record['schema_version']}")
    for tid, item in sorted(record["incumbents"].items()):
        if item.get("incumbent") is None:
            print(f"  {tid} {item['slug']:22} —  ({item.get('reason')})")
            continue
        line = f"  {tid} {item['slug']:22} {item['incumbent']}  [{item['layer']}]  {item['metric']}={item['value']}"
        if item.get("superseded"):
            sup = item["superseded"]
            line += f"  (supersedes {sup['method']} {sup['value']:.3f}, -{sup['improvement_percent']:.2f}%)"
        print(line)
    pending = [d for d in record["user_decisions"] if d["status"] == "ACCEPTED_NOT_YET_IMPLEMENTED"]
    print(f"  rejected routes: {len(record['rejected_routes'])}   open work: {len(record['open_work'])}"
          f"   owner decisions: {len(record['user_decisions'])} ({len(pending)} pending)")
    for d in pending:
        print(f"    pending: {d['id']} — {d['ruling']}")
    print(f"  written: {_rel(OUTPUT_DIR / 'incumbent.json')}")


if __name__ == "__main__":
    main()
