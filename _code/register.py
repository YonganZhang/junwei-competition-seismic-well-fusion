#!/usr/bin/env python3
"""Deprecated placeholder.

Default v4 uses `_codemap.md` + `_meta/_registry.yml` as a coarse code-domain /
entrypoint registry. Function-level scanning is opt-in and must be installed by
the project with its own schema and tests.
"""
import sys

print(
    "ERROR: _code/register.py is not enabled in default v4 projects.\n"
    "Update `_codemap.md` and `_meta/_registry.yml` manually, then run:\n\n"
    "  python3 ~/.codex/skills/share-top/scripts/top-lint.py .\n\n"
    "If this project really needs function-level registration, replace this\n"
    "placeholder with a project-specific helper and document the contract.",
    file=sys.stderr,
)
sys.exit(2)
