from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from data_pipeline import HuginSurfaces  # noqa: E402


def test_horizon_loader_ignores_openworks_metadata(tmp_path: Path) -> None:
    path = tmp_path / "horizon.dat"
    path.write_text(
        "# comment\nST10010\nHugin_Fm_Top\nSTAT\nTIME\n,\n"
        "9976,2469,432186.7,6477029.1,2776.2\n"
        "9977,2469,432189.7,6477041.2,2779.6\n"
    )
    xy, twt = HuginSurfaces._load(path)
    assert xy.shape == (2, 2)
    assert twt.tolist() == [2776.2, 2779.6]
