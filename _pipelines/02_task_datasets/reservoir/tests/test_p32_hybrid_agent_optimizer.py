from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import p32_hybrid_agent_optimizer as p32  # noqa: E402


def _candidate(model_name: str, model_kwargs: dict, rationale: str = "bounded probe") -> dict:
    return {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "rationale": rationale,
    }


def test_agent_candidate_contract_accepts_four_unique_two_family_pool() -> None:
    payload = {
        "candidates": [
            _candidate("reservoir_linear", {"learning_rate": 0.01, "l2_strength": 0.0}),
            _candidate("reservoir_ridge", {"learning_rate": 0.01, "l2_strength": 1e-4}),
            _candidate("tiny_mlp", {"hidden_dim": 16, "learning_rate": 0.003, "weight_decay": 1e-5}),
            _candidate("tiny_mlp", {"hidden_dim": 32, "learning_rate": 0.001, "weight_decay": 1e-4}),
        ]
    }
    candidates = p32.validate_agent_candidates(payload)
    assert len(candidates) == 4
    assert len({item.model_name for item in candidates}) == 3


def test_agent_candidate_contract_rejects_escape_and_duplicates() -> None:
    with pytest.raises(ValueError, match="allowlist"):
        p32.validate_candidate(
            _candidate("transformer", {"learning_rate": 0.001}), 0, "test"
        )
    duplicate = _candidate(
        "reservoir_linear", {"learning_rate": 0.01, "l2_strength": 0.0}
    )
    with pytest.raises(ValueError, match="duplicate"):
        p32.validate_agent_candidates({"candidates": [duplicate] * 4})


def test_prompt_exposes_no_metrics_labels_or_paths() -> None:
    prompt = p32.build_candidate_prompt(sample_count=100, feature_count=153)
    text = json.dumps(prompt, sort_keys=True).lower()
    assert "selection_metrics_visible\": false" in text
    assert "promotion_metrics_visible\": false" in text
    assert "test_data_visible\": false" in text
    assert "/mnt/" not in text
    assert "sample_id" not in text


def test_promotion_gate_requires_improvement_nondegradation_and_seed_wins() -> None:
    base = {
        "endpoint": {
            "promotion_primary_median": 1.0,
            "promotion_primary_values": [1.0, 1.0, 1.0],
            "promotion_worst_group_RMSE_median": {"PHIF": 1.0, "KLOGH": 1.0, "SW": 1.0},
        }
    }
    agent = {
        "endpoint": {
            "promotion_primary_median": 0.98,
            "promotion_primary_values": [0.98, 0.99, 1.01],
            "promotion_worst_group_RMSE_median": {"PHIF": 1.0, "KLOGH": 1.01, "SW": 0.99},
        }
    }
    gate = p32.promotion_gate(agent, base)
    assert gate["decision"] == "RETAIN_HYBRID"
    agent["endpoint"]["promotion_worst_group_RMSE_median"]["SW"] = 1.03
    assert p32.promotion_gate(agent, base)["decision"] == "KEEP_DETERMINISTIC"


def test_verify_rejects_budget_drift(tmp_path: Path) -> None:
    payload = {
        "schema_version": p32.SCHEMA_VERSION,
        "matched_budget": {"equal": False},
        "data": {"selection_promotion_overlap": False, "frozen_test_accessed": False},
    }
    (tmp_path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not matched"):
        p32.verify(tmp_path)


def test_candidate_signature_ignores_prose_order_and_source() -> None:
    first = {
        "candidate_id": "agent_1",
        "model_name": "reservoir_linear",
        "model_kwargs": {"learning_rate": 0.01, "l2_strength": 0.0},
        "rationale": "first wording",
        "source": "provider",
    }
    second = {
        **first,
        "candidate_id": "agent_4",
        "rationale": "different wording",
    }
    assert p32.candidate_signature(first) == p32.candidate_signature(second)
