"""Development-only HPO records; fixed baselines work without Optuna."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifacts import atomic_write_json, hash_payload
from .seeding import derive_seed


@dataclass(frozen=True)
class HPOPlan:
    sanity_trials: int = 8
    pilot_trials: int = 20
    top_configs: int = 3
    confirm_seeds: int = 3
    sampler: str = "random_then_tpe"
    pruner: str = "nop"

    def __post_init__(self) -> None:
        if not 8 <= self.sanity_trials <= 12:
            raise ValueError("sanity_trials must be in [8, 12]")
        if not 20 <= self.pilot_trials <= 30:
            raise ValueError("pilot_trials must be in [20, 30]")
        if self.top_configs < 3 or self.confirm_seeds < 3:
            raise ValueError("at least top 3 configs and 3 confirmation seeds are required")


@dataclass
class TrialResult:
    trial_id: int
    params: Mapping[str, Any]
    fold_scores: tuple[float, ...]
    guardrails: Mapping[str, float] = field(default_factory=dict)
    cost: Mapping[str, float] = field(default_factory=dict)
    seed: int = 0
    state: str = "complete"
    failure_reason: str | None = None

    @property
    def mean(self) -> float:
        if not self.fold_scores:
            raise RuntimeError("completed trial has no fold scores")
        return sum(self.fold_scores) / len(self.fold_scores)

    @property
    def std(self) -> float:
        mean = self.mean
        return math.sqrt(sum((score - mean) ** 2 for score in self.fold_scores) / len(self.fold_scores))

    @property
    def worst(self) -> float:
        return min(self.fold_scores)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.state == "complete":
            payload.update({"fold_mean": self.mean, "fold_std": self.std, "worst_fold": self.worst})
        payload["params_hash"] = hash_payload(self.params)
        return payload


DevelopmentObjective = Callable[[Mapping[str, Any], int], Mapping[str, Any]]


def run_fixed_trials(
    configs: Sequence[Mapping[str, Any]],
    objective: DevelopmentObjective,
    *,
    root_seed: int,
    output_dir: Path,
) -> list[TrialResult]:
    """Run deterministic development-only configurations.

    The signature deliberately has no test loader/data argument.  ``objective``
    must return ``fold_scores`` and may return guardrails/cost.
    """
    if not configs:
        raise ValueError("at least one fixed baseline/HPO config is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[TrialResult] = []
    for trial_id, params in enumerate(configs):
        seed = derive_seed(root_seed, "hpo_sampler", trial_id)
        started = time.monotonic()
        try:
            outcome = dict(objective(dict(params), seed))
            fold_scores = tuple(float(value) for value in outcome.pop("fold_scores"))
            if not fold_scores or not all(math.isfinite(value) for value in fold_scores):
                raise ValueError("objective returned empty or non-finite fold scores")
            cost = dict(outcome.pop("cost", {}))
            cost.setdefault("wall_seconds", time.monotonic() - started)
            result = TrialResult(
                trial_id=trial_id,
                params=dict(params),
                fold_scores=fold_scores,
                guardrails=dict(outcome.pop("guardrails", {})),
                cost=cost,
                seed=seed,
            )
        except Exception as exc:
            result = TrialResult(
                trial_id=trial_id,
                params=dict(params),
                fold_scores=(),
                cost={"wall_seconds": time.monotonic() - started},
                seed=seed,
                state="failed",
                failure_reason=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)
        atomic_write_json(output_dir / "trials.json", [item.to_dict() for item in results])
    if not any(item.state == "complete" for item in results):
        raise RuntimeError("all HPO/fixed trials failed")
    return results


def rank_trials(results: Sequence[TrialResult]) -> list[TrialResult]:
    complete = [result for result in results if result.state == "complete"]
    return sorted(complete, key=lambda item: (item.mean, item.worst, -item.std), reverse=True)


def run_optuna_study(
    *,
    suggest_params: Callable[[Any], Mapping[str, Any]],
    objective: DevelopmentObjective,
    root_seed: int,
    output_dir: Path,
    plan: HPOPlan | None = None,
) -> list[TrialResult]:
    """Optional sequential Optuna backend using development folds only.

    Import is deferred so fixed baselines and the rest of the framework never
    require Optuna.  TPE receives the sanity phase as startup trials and the
    default pruner is deliberately ``NopPruner``.
    """
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna backend requested but optuna is not installed; fixed baselines still work") from exc

    active_plan = plan or HPOPlan()
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler_seed = derive_seed(root_seed, "hpo_sampler")
    sampler = optuna.samplers.TPESampler(seed=sampler_seed, n_startup_trials=active_plan.sanity_trials)
    if active_plan.pruner == "nop":
        pruner = optuna.pruners.NopPruner()
    elif active_plan.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=active_plan.sanity_trials)
    else:
        raise ValueError(f"unsupported pruner={active_plan.pruner!r}")
    storage = f"sqlite:///{(output_dir / 'study.sqlite3').resolve()}"
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name="p4-development",
        load_if_exists=True,
    )
    records: list[TrialResult] = []

    def wrapped(trial: Any) -> float:
        params = dict(suggest_params(trial))
        seed = derive_seed(root_seed, "hpo_sampler", trial.number)
        started = time.monotonic()
        try:
            outcome = dict(objective(params, seed))
            scores = tuple(float(value) for value in outcome.pop("fold_scores"))
            if not scores or not all(math.isfinite(value) for value in scores):
                raise ValueError("objective returned empty or non-finite fold scores")
            result = TrialResult(
                trial_id=trial.number,
                params=params,
                fold_scores=scores,
                guardrails=dict(outcome.pop("guardrails", {})),
                cost={**dict(outcome.pop("cost", {})), "wall_seconds": time.monotonic() - started},
                seed=seed,
            )
            records.append(result)
            for key, value in result.to_dict().items():
                if key not in {"params", "fold_scores", "guardrails", "cost"}:
                    trial.set_user_attr(key, value)
            atomic_write_json(output_dir / "trials.json", [item.to_dict() for item in records])
            return result.mean
        except Exception as exc:
            records.append(
                TrialResult(
                    trial_id=trial.number,
                    params=params,
                    fold_scores=(),
                    cost={"wall_seconds": time.monotonic() - started},
                    seed=seed,
                    state="failed",
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
            )
            atomic_write_json(output_dir / "trials.json", [item.to_dict() for item in records])
            raise

    study.optimize(wrapped, n_trials=active_plan.sanity_trials + active_plan.pilot_trials, n_jobs=1)
    atomic_write_json(
        output_dir / "study_summary.json",
        {
            "sampler": type(sampler).__name__,
            "pruner": type(pruner).__name__,
            "sampler_seed": sampler_seed,
            "best_params": study.best_params,
            "best_value": study.best_value,
            "study_db": "study.sqlite3",
        },
    )
    return records
