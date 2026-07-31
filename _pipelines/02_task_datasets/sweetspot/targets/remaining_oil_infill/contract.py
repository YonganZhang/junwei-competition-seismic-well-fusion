"""Fail-closed feasibility gate for the simulation-only target 5."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from ..common import _source_root


TASK_ID = "sweetspot.remaining_oil_infill.simulation_case_v1"
STATUS = "not_feasible"
ARCHIVE = Path("_sandbox/volve_data/Volve_Reservoir_Model-Eclipse_model.zip")
REQUIRED_DYNAMIC_SUFFIXES = (".UNRST", ".GRID")


def not_feasible_evidence(source_root: Path | None = None) -> dict[str, Any]:
    root = _source_root(source_root)
    archive_path = root / ARCHIVE
    if not archive_path.exists():
        return {
            "task_id": TASK_ID, "status": STATUS,
            "reason": "Eclipse reservoir-model archive is missing",
            "missing": [ARCHIVE.as_posix()],
        }
    with ZipFile(archive_path) as archive:
        members = archive.namelist()
        dynamic = [name for name in members if name.upper().endswith(REQUIRED_DYNAMIC_SUFFIXES)]
        static = [
            name for name in members
            if any(token in name.upper() for token in ("PHIF", "KLOGH", "SWIRR"))
        ]
        dynamic_evidence = [
            {"member": name, "size_bytes": archive.getinfo(name).file_size, "crc32": f"{archive.getinfo(name).CRC:08x}"}
            for name in sorted(dynamic)
        ]
    parser_candidates = ("resdata", "ecl", "opm", "xtgeo")
    installed = {name: importlib.util.find_spec(name) is not None for name in parser_candidates}
    blockers = []
    if not dynamic:
        blockers.append("dynamic Eclipse UNRST/GRID members are absent")
    if not any(installed.values()):
        blockers.append("no tested Eclipse cell-state parser is installed")
    blockers.extend([
        "prediction time and evaluation time are not domain-approved",
        "candidate infill locations and economic/spacing constraints are not frozen",
    ])
    return {
        "task_id": TASK_ID,
        "status": STATUS,
        "truth_scope": "simulation case only; must never be presented as field truth",
        "archive": ARCHIVE.as_posix(),
        "archive_size_bytes": archive_path.stat().st_size,
        "dynamic_members": dynamic_evidence,
        "static_property_member_count": len(static),
        "parser_availability": installed,
        "blockers": blockers,
        "unblock_acceptance": [
            "install and validate one Eclipse cell-state parser against known cells/timesteps",
            "freeze realization, prediction date, evaluation date and visible history",
            "freeze candidate cells, spacing/economic screen, target transform and frozen test realization/time",
            "generate split manifest before training and keep test time/realization outside HPO",
        ],
        "baseline": None,
        "metrics": None,
        "figures": None,
        "no_synthetic_fallback": True,
    }
