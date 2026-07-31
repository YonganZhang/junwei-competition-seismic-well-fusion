"""Materialize the private sweetspot P6 Gaia/DAGT evidence package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _models.sweetspot.p6_gaia_dagt import P6_OUTPUT_DIR, materialize_private_package


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=P6_OUTPUT_DIR)
    args = parser.parse_args(argv)
    return materialize_private_package(args.output_dir)


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
