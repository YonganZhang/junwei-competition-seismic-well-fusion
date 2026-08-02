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
    if not p29.TRAIN_H5.is_file():
        split = json.loads((p29.OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))["split"]
        assert set(split["train"]["families"]) == {"15/9-19", "15/9-F-1", "15/9-F-11"}
        assert set(split["selection_dev"]["families"]) == {"15/9-F-12"}
        assert set(split["promotion_dev"]["families"]) == {"15/9-F-12"}
        assert not (set(split["train"]["families"]) & set(split["selection_dev"]["families"]))
        return
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

    monkeypatch.setattr(p29, "load_checkpointed_tiny_mlp", lambda checkpoint_path, n_features: FakeModel())
    monkeypatch.setattr(p29, "infer", lambda model, features, stats: fake_prediction.copy())
    monkeypatch.setattr(p29, "evaluate_predictions", lambda actual, predicted, stats: dict(fake_metrics))
    monkeypatch.setattr(p29, "prediction_hash", lambda array: "hash-a0")

    replay = p29.identity_replay(
        selection_features=np.zeros((1, 2)),
        selection_targets=np.zeros((1, 3)),
        promotion_features=np.zeros((1, 2)),
        promotion_targets=np.zeros((1, 3)),
        stats={},
        a0_selection_hash="hash-a0",
        a0_promotion_hash="hash-a0",
        a0_config_hash="cfg-a0",
        checkpoint_path=Path("matched_a0.npz"),
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


def test_matched_a0_uses_same_seeds_and_keeps_historical_checkpoint_noncausal(monkeypatch, tmp_path) -> None:
    seen: list[tuple[int, int]] = []

    class FakeModel:
        def save_checkpoint(self, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{path.name}".encode("utf-8"))

    monkeypatch.setattr(p29, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(p29, "train_model", lambda route, features, target_norm, seed, budget_steps: seen.append((int(seed), int(budget_steps))) or FakeModel())
    monkeypatch.setattr(p29, "infer", lambda model, features, stats: np.full((len(features), 3), 0.5, dtype=float))
    monkeypatch.setattr(
        p29,
        "evaluate_predictions",
        lambda actual, predicted, stats: {
            "composite_mean_train_std_normalized_RMSE": 1.0,
            "physical_MAE_macro": 2.0,
            "worst_group_RMSE": {"PHIF": 1.0, "KLOGH": 1.0, "SW": 1.0},
        },
    )
    result = p29.train_matched_a0_trials(
        train_features=np.zeros((4, 2)),
        train_target_norm=np.zeros((4, 3)),
        selection_features=np.zeros((2, 2)),
        selection_targets=np.zeros((2, 3)),
        promotion_features=np.zeros((2, 2)),
        promotion_targets=np.zeros((2, 3)),
        stats={},
        budget_steps=7,
    )
    assert seen == [(2693, 7), (2694, 7), (2695, 7)]
    assert result["seed_pool"] == [2693, 2694, 2695]
    assert result["budget_steps"] == 7
    assert result["historical_reference"]["checkpoint_path"] == "_outputs/checkpoints/best.ckpt"
    assert result["historical_reference"]["checkpoint_path"] != result["replay_trial"]["checkpoint_path"]
    assert result["replay_trial"]["checkpoint_path"].startswith(str((tmp_path / "checkpoints").as_posix()))


def test_identity_replay_matches_saved_a0_trial_hash(monkeypatch, tmp_path) -> None:
    expected = np.array([[0.2, 0.3, 0.4]], dtype=float)

    class FakeModel:
        pass

    monkeypatch.setattr(p29, "load_checkpointed_tiny_mlp", lambda checkpoint_path, n_features: FakeModel())
    monkeypatch.setattr(p29, "infer", lambda model, features, stats: expected.copy())
    monkeypatch.setattr(p29, "prediction_hash", lambda array: "same-hash")
    monkeypatch.setattr(
        p29,
        "evaluate_predictions",
        lambda actual, predicted, stats: {
            "composite_mean_train_std_normalized_RMSE": 1.0,
            "physical_MAE_macro": 2.0,
            "worst_group_RMSE": {"PHIF": 1.0, "KLOGH": 1.0, "SW": 1.0},
        },
    )
    replay = p29.identity_replay(
        selection_features=np.zeros((1, 2)),
        selection_targets=np.zeros((1, 3)),
        promotion_features=np.zeros((1, 2)),
        promotion_targets=np.zeros((1, 3)),
        stats={},
        a0_selection_hash="same-hash",
        a0_promotion_hash="same-hash",
        a0_config_hash="cfg-a0",
        checkpoint_path=tmp_path / "a0.npz",
    )
    assert replay["selection_hash_matches_a0"] is True
    assert replay["promotion_hash_matches_a0"] is True
    assert replay["config_hash"] == "cfg-a0"
    assert replay["checkpoint_path"].endswith("a0.npz")


def test_per_strategy_gate_is_independent_of_a3_failure() -> None:
    a0_selection = {
        "composite_mean_train_std_normalized_RMSE": 1.0,
        "worst_group_RMSE": {"PHIF": 1.0, "KLOGH": 1.0, "SW": 1.0},
    }
    a0_promotion = {
        "composite_mean_train_std_normalized_RMSE": 1.0,
        "worst_group_RMSE": {"PHIF": 1.0, "KLOGH": 1.0, "SW": 1.0},
    }
    a2l = {
        "selection_median": {
            "composite_mean_train_std_normalized_RMSE": 0.8,
            "worst_group_RMSE": {"PHIF": 0.8, "KLOGH": 0.8, "SW": 0.8},
        },
        "promotion_median": {
            "composite_mean_train_std_normalized_RMSE": 0.8,
            "worst_group_RMSE": {"PHIF": 0.8, "KLOGH": 0.8, "SW": 0.8},
        },
        "status": "ok",
    }
    gates = p29.evaluate_strategy_gate(
        selection_median=a2l["selection_median"],
        promotion_median=a2l["promotion_median"],
        a0_selection_metrics=a0_selection,
        a0_promotion_metrics=a0_promotion,
        declared_comparators=[
            {
                "status": "ok",
                "selection_median": {
                    "composite_mean_train_std_normalized_RMSE": 0.9,
                },
            },
            {
                "status": "blocked",
                "selection_median": {
                    "composite_mean_train_std_normalized_RMSE": 999.0,
                },
            },
        ],
    )
    assert gates["selection_ok"] is True
    assert gates["promotion_ok"] is True
    assert gates["declared_comparison_ok"] is True
    assert gates["retained"] is True
