from __future__ import annotations

import json
from pathlib import Path


def test_all_six_tracks_have_a_gated_foundation_route() -> None:
    path = Path(__file__).parents[1] / "track_routes.v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload["tracks"]) == {
        "fault",
        "facies",
        "property",
        "lithofacies",
        "sweetspot",
        "reconstruction",
    }
    for route in payload["tracks"].values():
        assert route["foundation_family"]
        assert route["adapter"]
        assert route["gaia_integration"]
        assert route["current_gate"]
