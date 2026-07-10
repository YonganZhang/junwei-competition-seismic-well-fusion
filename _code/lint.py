#!/usr/bin/env python3
"""Deprecated placeholder for old function-registry lint."""
import sys

print(
    "ERROR: _code/lint.py is not enabled in default v4 projects.\n"
    "Use the active TOP lint instead:\n\n"
    "  python3 ~/.codex/skills/share-top/scripts/top-lint.py .\n",
    file=sys.stderr,
)
sys.exit(2)
