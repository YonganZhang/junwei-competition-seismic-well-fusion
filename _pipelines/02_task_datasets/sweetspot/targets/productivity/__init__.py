"""Target 3: causal 30-day productivity forecast."""

from .contract import build_dataset_and_manifest, task_spec

__all__ = ["build_dataset_and_manifest", "task_spec"]
