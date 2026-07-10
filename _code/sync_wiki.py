#!/usr/bin/env python3
"""Deprecated placeholder for old registry -> wiki projection."""
import sys

print(
    "ERROR: _code/sync_wiki.py is not enabled in default v4 projects.\n"
    "Current state belongs in `_wiki-methodology/_top/_task_plan.md`; code\n"
    "domains belong in `_codemap.md` and `_meta/_registry.yml`; tests belong in\n"
    "`_wiki-methodology/_tests/`. Do not assume automatic wiki projection.\n",
    file=sys.stderr,
)
sys.exit(2)
