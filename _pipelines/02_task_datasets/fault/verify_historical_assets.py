#!/usr/bin/env python3
"""Explicit integration gate for the optional historical baseline bundle."""
from __future__ import annotations

import json

from audit_utils import verify_historical_artifacts


def main() -> None:
    verified = verify_historical_artifacts()
    print(
        json.dumps(
            {
                "status": "verified",
                "artifact_count": len(verified),
                "sha256": verified,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
