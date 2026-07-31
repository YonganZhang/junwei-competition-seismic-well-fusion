"""Compatibility package for canonical reconstruction models."""
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def compatibility_task_spec():
    from p4_reconstruction import task_spec
    return task_spec("strict")
