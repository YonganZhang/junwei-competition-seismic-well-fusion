from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE = Path(__file__).resolve().parents[1] / "lithofacies_p33_hybrid_agent_optimizer.py"
SPEC = importlib.util.spec_from_file_location("lithofacies_p33", MODULE)
p33 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = p33
SPEC.loader.exec_module(p33)


def _candidate(**overrides):
    value = {
        "max_depth": 4,
        "eta": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "rationale": "bounded joint regularization",
    }
    value.update(overrides)
    return value


def test_candidate_validation_is_bounded() -> None:
    item = p33.validate_candidate(_candidate(), 0, "test")
    assert p33.executable_config(item)["rounds"] == 60
    with pytest.raises(ValueError, match="max_depth"):
        p33.validate_candidate(_candidate(max_depth=8), 0, "test")
    with pytest.raises(ValueError, match="subsample"):
        p33.validate_candidate(_candidate(subsample=0.5), 0, "test")


def test_pool_requires_four_unique_configs() -> None:
    payload = {"candidates": [_candidate(max_depth=depth) for depth in (2, 3, 4, 5)]}
    assert len(p33.validate_agent_candidates(payload)) == 4
    payload["candidates"][3] = dict(payload["candidates"][2])
    with pytest.raises(ValueError, match="duplicate"):
        p33.validate_agent_candidates(payload)


def test_prompt_hides_promotion_and_metrics() -> None:
    prompt = p33.build_candidate_prompt()
    assert prompt["budget"]["promotion_fold_hidden"] is True
    assert prompt["data_boundary"]["promotion_metrics_visible"] is False
    assert prompt["allowlist"]["rounds"] == 60


def test_gate_requires_delta_and_two_seed_wins() -> None:
    def strategy(values):
        return {
            "promotion": {
                "rows": [
                    {"metrics": {p33.PRIMARY_METRIC: value}} for value in values
                ]
            }
        }

    good = p33.promotion_gate(
        strategy([0.22, 0.23, 0.24]),
        strategy([0.21] * 3),
        strategy([0.215] * 3)["promotion"],
    )
    assert good["decision"] == "RETAIN_HYBRID"
    flat = p33.promotion_gate(
        strategy([0.211, 0.212, 0.209]),
        strategy([0.20] * 3),
        strategy([0.21] * 3)["promotion"],
    )
    assert flat["decision"] == "KEEP_CURRENT_DEFAULT"


def test_verify_rejects_budget_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "schema_version": p33.SCHEMA_VERSION,
        "data": {
            "selection_promotion_overlap": False,
            "frozen_test_accessed": False,
            "known_holdout_accessed": False,
        },
        "matched_budget": {"equal": False},
    }
    (tmp_path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(p33, "_owned_output", lambda path: path)
    with pytest.raises(ValueError, match="not matched"):
        p33.verify(tmp_path)


def test_signature_ignores_rationale_and_id() -> None:
    first = p33.Candidate("a", 4, 0.08, 0.9, 0.85, "first", "provider")
    second = p33.Candidate("b", 4, 0.08, 0.9, 0.85, "second", "replay")
    assert p33.candidate_signature(first) == p33.candidate_signature(second)


def test_independent_verifier_ignores_runtime_but_not_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def summary(response_id: str, runtime: float, metric: float):
        promotion = {
            "executable_config": {"max_depth": 3, "eta": 0.2, "rounds": 60, "subsample": 1.0, "colsample_bytree": 1.0},
            "mean_fixed_schema_macro_f1": metric,
            "runtime_s": runtime,
            "rows": [
                {
                    "fold_id": 3,
                    "repeat_id": 0,
                    "seed": 1,
                    "metrics": {p33.PRIMARY_METRIC: metric},
                    "prediction_sha256": "abc",
                }
            ],
        }
        selection = [{"candidate_signature": "same"}]
        strategy = {
            "selected_executable_config": promotion["executable_config"],
            "promotion": promotion,
            "selection": selection,
        }
        return {
            "provider": {"response_id": response_id},
            "agent": strategy,
            "promotion_gate": {"decision": "RETAIN_HYBRID"},
        }

    primary = tmp_path / "primary"
    replay = primary / "independent_replay"
    replay.mkdir(parents=True)
    (primary / "summary.json").write_text(json.dumps(summary("a", 1.0, 0.2)), encoding="utf-8")
    (replay / "summary.json").write_text(json.dumps(summary("b", 9.0, 0.2)), encoding="utf-8")
    monkeypatch.setattr(p33, "_write_json", lambda path, value: None)
    result = p33.verify_independent_replay(primary, replay)
    assert result["endpoint_metrics_stable"] is True
    assert result["verified"] is True
