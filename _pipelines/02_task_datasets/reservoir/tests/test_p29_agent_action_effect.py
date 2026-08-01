from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(PROJECT_ROOT / "_code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "_code"))


import p29_agent_action_effect as p29  # noqa: E402


def test_split_is_family_disjoint_and_guard_only() -> None:
    split = p29.split_records()
    assert {record.family_id for record in split["train"]} == {"15/9-19", "15/9-F-1", "15/9-F-11"}
    assert {record.family_id for record in split["selection_dev"]} == {"15/9-F-12"}
    assert {record.family_id for record in split["promotion_dev"]} == {"15/9-F-12"}
    assert not ({record.family_id for record in split["train"]} & {record.family_id for record in split["selection_dev"]})
    assert not ({record.family_id for record in split["train"]} & {record.family_id for record in split["promotion_dev"]})
    assert not ({record.sample_id for record in split["selection_dev"]} & {record.sample_id for record in split["promotion_dev"]})


def test_prompt_exposes_route_semantics_and_safe_deltas() -> None:
    route_pilots = [
        {
            "route_id": "tiny_mlp_default",
            "semantics": "tiny_mlp: A0-compatible tiny MLP with frozen default regularization",
            "lane": "tabular-cpu",
            "blocked": False,
            "feedback": "flat",
            "selection": {"composite_mean_train_std_normalized_RMSE": 0.9},
            "promotion": {"composite_mean_train_std_normalized_RMSE": 0.95},
        },
        {
            "route_id": "tiny_mlp_l2",
            "semantics": "tiny_mlp: Same architecture with stronger L2",
            "lane": "tabular-cpu",
            "blocked": False,
            "feedback": "improved",
            "selection": {"composite_mean_train_std_normalized_RMSE": 0.8},
            "promotion": {"composite_mean_train_std_normalized_RMSE": 0.7},
        },
    ]
    prompt = p29.build_prompt(
        route_pilots,
        {"composite_mean_train_std_normalized_RMSE": 1.0},
        {"composite_mean_train_std_normalized_RMSE": 1.0},
        "abc123",
    )
    prompt_text = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
    assert "tiny_mlp: A0-compatible tiny MLP" in prompt_text
    assert "selection_primary_delta_rel" in prompt_text
    assert "promotion_primary_delta_rel" in prompt_text
    assert "test.h5" not in prompt_text
    assert prompt["schema"]["action"] == "select|stop"


def test_identity_replay_is_noop_and_gate_excludes_a0_a1(monkeypatch) -> None:
    fake_prediction = np.array([[0.1, 0.2, 0.3]], dtype=float)
    fake_metrics = {
        "composite_mean_train_std_normalized_RMSE": 0.9,
        "physical_MAE_macro": 1.0,
        "worst_group_RMSE": {"PHIF": 0.1, "KLOGH": 0.2, "SW": 0.3},
    }

    class FakeModel:
        pass

    monkeypatch.setattr(p29, "load_a0_checkpoint", lambda n_features: FakeModel())
    monkeypatch.setattr(p29, "infer", lambda model, features, stats: fake_prediction.copy())
    monkeypatch.setattr(p29, "evaluate_predictions", lambda actual, predicted, stats: dict(fake_metrics))
    monkeypatch.setattr(p29, "prediction_hash", lambda array: "hash-a0")

    replay = p29.identity_replay(
        train_features=np.zeros((1, 2)),
        selection_features=np.zeros((1, 2)),
        selection_targets=np.zeros((1, 3)),
        promotion_features=np.zeros((1, 2)),
        promotion_targets=np.zeros((1, 3)),
        stats={},
        a0_selection_hash="hash-a0",
        a0_promotion_hash="hash-a0",
        a0_config_hash="cfg-a0",
    )
    assert replay["selection_hash_matches_a0"] is True
    assert replay["promotion_hash_matches_a0"] is True
    assert replay["config_hash"] == "cfg-a0"
    assert p29.candidate_route_names() == ("A2L", "A2D", "A3")


def test_action_effects_mark_noop_and_candidate_visibility() -> None:
    summary = {
        "a0": {"selection_prediction_hash": "sel-a0", "config_hash": "cfg-a0"},
        "a1": {"selection_hash_matches_a0": True, "selection_prediction_hash": "sel-a0", "config_hash": "cfg-a0"},
        "strategies": {
            "A2L": {
                "chosen_route_id": "tiny_mlp_default",
                "config_hash": "cfg-1",
                "selection_prediction_hash": "sel-1",
                "selection_primary_delta_rel": -0.01,
                "promotion_primary_delta_rel": -0.02,
                "semantics": "tiny_mlp: A0-compatible tiny MLP with frozen default regularization",
                "lane": "tabular-cpu",
            },
            "A2D": {
                "chosen_route_id": "tiny_mlp_l2",
                "config_hash": "cfg-2",
                "selection_prediction_hash": "sel-2",
                "selection_primary_delta_rel": 0.03,
                "promotion_primary_delta_rel": 0.04,
                "semantics": "tiny_mlp: Same architecture with stronger L2",
                "lane": "tabular-cpu",
            },
            "A3": {
                "chosen_route_id": "reservoir_ridge",
                "config_hash": "cfg-3",
                "selection_prediction_hash": "sel-3",
                "selection_primary_delta_rel": 0.05,
                "promotion_primary_delta_rel": 0.06,
                "semantics": "reservoir_ridge: SGD ridge baseline",
                "lane": "tabular-cpu",
            },
        },
        "oracle_ceiling": {
            "route_id": "tiny_mlp_default",
            "config_hash": "cfg-1",
            "prediction_hash": "sel-1",
            "selection_primary_delta_rel": -0.01,
            "promotion_primary_delta_rel": -0.02,
        },
    }
    action_effects = p29.build_action_effects(summary)
    assert action_effects["gate_excludes"] == ["A0", "A1"]
    a1_row = next(row for row in action_effects["rows"] if row["kind"] == "A1")
    assert a1_row["no_op"] is True
    candidate_rows = [row for row in action_effects["rows"] if row["kind"] in {"A2L", "A2D", "A3"}]
    assert candidate_rows and all(row["visible_to_llm"] is True for row in candidate_rows)
