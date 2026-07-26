from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


def _stable_int(seed: int, index: int, item: object, mode: str) -> int:
    payload = f"{seed}|{index}|{mode}|{repr(item)}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16)


def _permute(items: Sequence[T], seed: int, mode: str) -> tuple[T, ...]:
    enumerated = sorted(
        (( _stable_int(seed, index, item, mode), index, item) for index, item in enumerate(items)),
        key=lambda row: (row[0], row[1]),
    )
    return tuple(item for _, _, item in enumerated)


def real_control(items: Iterable[T]) -> tuple[T, ...]:
    return tuple(items)


def shuffle_control(items: Iterable[T], seed: int = 0) -> tuple[T, ...]:
    sequence = tuple(items)
    return _permute(sequence, seed, "shuffle")


def random_control(items: Iterable[T], seed: int = 0) -> tuple[T, ...]:
    sequence = list(items)
    rng = random.Random(seed)
    rng.shuffle(sequence)
    return tuple(sequence)


def counterfactual_control(items: Iterable[T], seed: int = 0) -> tuple[T, ...]:
    sequence = tuple(items)
    permuted = _permute(sequence, seed, "counterfactual")
    return tuple(reversed(permuted))


def apply_control(items: Iterable[T], mode: str, seed: int = 0) -> tuple[T, ...]:
    if mode == "real":
        return real_control(items)
    if mode == "shuffle":
        return shuffle_control(items, seed=seed)
    if mode == "random":
        return random_control(items, seed=seed)
    if mode == "counterfactual":
        return counterfactual_control(items, seed=seed)
    raise ValueError(f"Unsupported control mode: {mode}")
