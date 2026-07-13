"""Stable role-based seeds and deterministic runtime reporting."""
from __future__ import annotations

import hashlib
import os
import platform
import random
from dataclasses import asdict, dataclass, field
from typing import Any


DEFAULT_ROOT_SEED = 2693
SEED_ROLES = (
    "split", "cv", "model", "loader", "sampler", "augmentation", "hpo_sampler", "diagnostic"
)


def derive_seed(root_seed: int, role: str, *parts: object) -> int:
    if not isinstance(root_seed, int) or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer")
    if not role:
        raise ValueError("role must not be empty")
    encoded = "\x1f".join([str(root_seed), role, *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(encoded, digest_size=4).digest(), "big")


@dataclass(frozen=True)
class SeedTree:
    root_seed: int = DEFAULT_ROOT_SEED
    roles: tuple[str, ...] = SEED_ROLES

    def __post_init__(self) -> None:
        if self.root_seed < 0:
            raise ValueError("root_seed must be non-negative")
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("seed roles must be unique")

    def seed(self, role: str, *parts: object) -> int:
        if role not in self.roles:
            raise KeyError(f"unknown seed role {role!r}; declared roles={self.roles}")
        return derive_seed(self.root_seed, role, *parts)

    def to_dict(self) -> dict[str, Any]:
        return {"root_seed": self.root_seed, "derived": {role: self.seed(role) for role in self.roles}}


@dataclass
class DeterminismReport:
    root_seed: int
    seed_tree: dict[str, Any]
    strict_requested: bool
    python_seeded: bool = False
    numpy_seeded: bool = False
    torch_seeded: bool = False
    torch_deterministic: bool = False
    warnings: list[str] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seed_everything(
    root_seed: int = DEFAULT_ROOT_SEED,
    *,
    strict: bool = True,
    include_torch: bool = True,
) -> DeterminismReport:
    tree = SeedTree(root_seed)
    report = DeterminismReport(
        root_seed=root_seed,
        seed_tree=tree.to_dict(),
        strict_requested=strict,
        environment={"python": platform.python_version(), "platform": platform.platform()},
    )
    os.environ["PYTHONHASHSEED"] = str(tree.seed("model"))
    random.seed(tree.seed("model"))
    report.python_seeded = True
    try:
        import numpy as np

        np.random.seed(tree.seed("model"))
        report.numpy_seeded = True
        report.environment["numpy"] = np.__version__
    except ImportError:
        report.warnings.append("NumPy is not installed; NumPy RNG was not seeded")

    if include_torch:
        try:
            import torch

            torch.manual_seed(tree.seed("model"))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(tree.seed("model"))
            torch.use_deterministic_algorithms(strict, warn_only=not strict)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = strict
            report.torch_seeded = True
            report.torch_deterministic = strict
            report.environment["torch"] = torch.__version__
            report.environment["cuda_available"] = bool(torch.cuda.is_available())
        except ImportError:
            report.warnings.append("PyTorch is not installed; Torch RNG was not seeded")
        except RuntimeError as exc:
            if strict:
                raise
            report.warnings.append(f"Torch deterministic mode degraded: {exc}")
    return report


def dataloader_worker_seed(root_seed: int, worker_id: int) -> int:
    seed = derive_seed(root_seed, "loader_worker", worker_id)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    return seed
