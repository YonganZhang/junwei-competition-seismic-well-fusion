from __future__ import annotations

import sys
from pathlib import Path


RESERVOIR_DIR = Path(__file__).resolve().parents[1]
if str(RESERVOIR_DIR) not in sys.path:
    sys.path.insert(0, str(RESERVOIR_DIR))


import p18_cigbench_property as p18  # type: ignore


def test_build_evidence_records_blocked_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        p18,
        "try_property_predictor_smoke",
        lambda **kwargs: {
            "status": "BLOCKED_DATA_OR_API",
            "reason": "property weight download / predictor init failed",
            "error": "HTTPError('[E3020] [404]')",
        },
    )

    report = p18.build_evidence()
    assert report["verdict"] == "BLOCKED_DATA_OR_API"
    assert report["smoke"]["status"] == "BLOCKED_DATA_OR_API"
    assert any("CIG-Bench-Property.pth" in reason for reason in report["blocker_reasons"])
    assert report["development_inputs"]["train"]["seismic_shape"] == (3, 3, 9)
    assert report["development_inputs"]["train"]["well_log_seq_shape"] == (9, 8)
    assert report["baseline"]["run_manifest"]["model"] == "tiny_mlp"


def test_render_evidence_mentions_blocker_and_no_holdout() -> None:
    text = p18.render_evidence(
        {
            "verdict": "BLOCKED_DATA_OR_API",
            "api": {
                "cig_bench_version": "0.2.0",
                "modelscope_version": "1.39.0",
                "torch_version": "2.13.0",
                "property_predictor_signature": "(restore_path=None, device='cpu')",
                "property_registry": {"property": ("douyimin/CIG-Bench", "CIG-Bench-Property.pth")},
            },
            "development_inputs": {
                "train": {
                    "path": Path("train.h5"),
                    "sha256": "abc",
                    "sample_count": 1135,
                    "sample_key": "sample_0000000",
                    "seismic_shape": (3, 3, 9),
                    "well_log_seq_shape": (9, 8),
                    "label_shape": (3,),
                    "meta": {"target_names": ["PHIF", "log1p(KLOGH)", "SW"]},
                },
                "guard": {
                    "path": Path("guard.npz"),
                    "sha256": "def",
                    "sample_count": 81,
                    "seismic_shape": (3, 3, 9),
                    "well_log_seq_shape": (9, 8),
                    "label_shape": (3,),
                },
            },
            "baseline": {
                "run_manifest": {"model": "tiny_mlp", "framework": "NumPy"},
                "split_manifest": {"family_partition": {"a": "train"}},
                "metrics": {
                    "per_target": {
                        "PHIF": {"RMSE": 0.1},
                        "log1p(KLOGH)": {"RMSE": 1.0},
                        "SW": {"RMSE": 0.2},
                    }
                },
            },
            "smoke": {"status": "BLOCKED_DATA_OR_API", "reason": "blocked", "error": "boom"},
            "blocker_reasons": ["reason one"],
            "commands": ["cmd one"],
            "candidate_metrics": None,
        }
    )
    assert "BLOCKED_DATA_OR_API" in text
    assert "No frozen holdout / test.h5 was opened." in text
    assert "tiny_mlp" in text
    assert "reason one" in text
