#!/usr/bin/env python3
"""P4 facies lifecycle CLI: prepare, smoke, CV, freeze, refit, test, visualize."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from _code.ml_framework.artifacts import (  # noqa: E402
    ArtifactManifest,
    atomic_write_json,
    hash_file,
    hash_payload,
)
from _code.ml_framework.checkpoint import load_checkpoint  # noqa: E402
from _code.ml_framework.cv import run_development_cv  # noqa: E402
from _code.ml_framework.lifecycle import (  # noqa: E402
    ExperimentLifecycle,
    ExperimentState,
)
from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from _code.ml_framework.run_layout import create_run_layout  # noqa: E402
from _code.ml_framework.seeding import seed_everything  # noqa: E402
from _code.ml_framework.splits import Fold, SplitManifest, validate_manifest  # noqa: E402

from p4_data import FaciesArchive, FoldPreprocessor, records_by_id  # noqa: E402
from p4_metrics import fit_temperature_scaling  # noqa: E402
from p4_spatial import build_facies_spatial_manifest  # noqa: E402
from p4_tasks import (  # noqa: E402
    TASK_IDS,
    fixed_baseline_config,
    get_task_spec,
    hpo_contract,
)
from p4_training import (  # noqa: E402
    archive_prediction_maps,
    environment_record,
    predict_consumed_frozen_test,
    resolve_device,
    run_development_fold,
    run_real_data_smoke,
    train_development_model,
)
from p4_visualize import render_archived_diagnostics  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _manifest_from_dict(payload: Mapping[str, Any]) -> SplitManifest:
    folds = tuple(
        Fold(
            fold_id=int(item["fold_id"]),
            train_groups=tuple(item["train_groups"]),
            validation_groups=tuple(item["validation_groups"]),
            train_sample_ids=tuple(item["train_sample_ids"]),
            validation_sample_ids=tuple(item["validation_sample_ids"]),
            purge=dict(item["purge"]),
            support=dict(item["support"]),
        )
        for item in payload["folds"]
    )
    manifest = SplitManifest(
        manifest_version=str(payload["manifest_version"]),
        group_key=str(payload["group_key"]),
        requested_n_splits=int(payload["requested_n_splits"]),
        effective_n_splits=int(payload["effective_n_splits"]),
        downgrade_reason=payload.get("downgrade_reason"),
        test_groups=tuple(payload["test_groups"]),
        test_sample_ids=tuple(payload["test_sample_ids"]),
        development_groups=tuple(payload["development_groups"]),
        development_sample_ids=tuple(payload["development_sample_ids"]),
        folds=folds,
        metadata=dict(payload["metadata"]),
    )
    validate_manifest(manifest)
    return manifest


def _load_manifest(run_root: Path) -> SplitManifest:
    return _manifest_from_dict(_read_json(run_root / "split_manifest.json"))


def _load_lifecycle(run_root: Path) -> ExperimentLifecycle:
    payload = _read_json(run_root / "lifecycle.json")
    lifecycle = ExperimentLifecycle(str(payload["experiment_id"]))
    lifecycle.state = ExperimentState(payload["state"])
    lifecycle.evidence = {
        str(key): dict(value) for key, value in dict(payload.get("evidence", {})).items()
    }
    lifecycle.test_consumed_at = payload.get("test_consumed_at")
    return lifecycle


def _write_lifecycle(run_root: Path, lifecycle: ExperimentLifecycle) -> None:
    atomic_write_json(run_root / "lifecycle.json", lifecycle.to_dict())


def _require_state(lifecycle: ExperimentLifecycle, state: ExperimentState) -> None:
    if lifecycle.state != state:
        raise RuntimeError(
            f"command requires lifecycle {state.value}, found {lifecycle.state.value}"
        )


def _refresh_manifest(run_root: Path, run_id: str) -> Path:
    manifest = ArtifactManifest(run_id=run_id, root=run_root)
    for path in sorted(run_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest.register(path.relative_to(run_root).as_posix(), role="p4_run_artifact")
    output = manifest.write()
    manifest.verify()
    return output


def prepare_run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty run root: {run_root}")
    create_run_layout(run_root)
    archive = FaciesArchive(args.task, args.processed_root)
    development = archive.development_index()
    frozen_test_metadata = archive.frozen_test_index(labels_consumed=False)
    manifest = build_facies_spatial_manifest(
        args.task,
        development,
        frozen_test_metadata,
        requested_n_splits=args.requested_n_splits,
        buffer_groups=args.buffer_groups,
    )
    task_spec = get_task_spec(args.task)
    config = fixed_baseline_config(args.task)
    config["run_id"] = args.run_id
    config["processed_root"] = "provided_at_runtime_not_serialized"
    seed_report = seed_everything(config["root_seed"], strict=False).to_dict()
    atomic_write_json(run_root / "task_spec.json", task_spec.to_dict())
    atomic_write_json(run_root / "run_config.json", config)
    atomic_write_json(run_root / "seed_report.json", seed_report)
    atomic_write_json(run_root / "environment.json", environment_record(torch.device("cpu")))
    atomic_write_json(run_root / "split_manifest.json", manifest.to_dict())
    lifecycle = ExperimentLifecycle(args.run_id)
    lifecycle.advance(
        ExperimentState.SPLIT_LOCKED,
        {
            "split_hash": manifest.stable_hash(),
            "task_id": args.task,
            "label_version": task_spec.label_version,
            "test_labels_read": False,
        },
    )
    _write_lifecycle(run_root, lifecycle)
    atomic_write_json(run_root / "hpo" / "plan.json", hpo_contract())
    _refresh_manifest(run_root, args.run_id)
    return {
        "state": lifecycle.state.value,
        "task_id": args.task,
        "label_version": task_spec.label_version,
        "requested_n_splits": manifest.requested_n_splits,
        "effective_n_splits": manifest.effective_n_splits,
        "downgrade_reason": manifest.downgrade_reason,
        "development_samples": len(manifest.development_sample_ids),
        "internal_buffer_samples_excluded": manifest.metadata["cv_excluded_buffer_sample_count"],
        "test_samples_frozen_without_labels": len(manifest.test_sample_ids),
        "split_hash": manifest.stable_hash(),
    }


def smoke_run(args: argparse.Namespace) -> dict[str, Any]:
    lifecycle = _load_lifecycle(args.run_root)
    _require_state(lifecycle, ExperimentState.SPLIT_LOCKED)
    manifest = _load_manifest(args.run_root)
    config = _read_json(args.run_root / "run_config.json")
    archive = FaciesArchive(config["task_id"], args.processed_root)
    evidence = run_real_data_smoke(
        task_id=config["task_id"],
        archive=archive,
        manifest=manifest,
        output_dir=args.run_root / "smoke",
        device=resolve_device(args.device),
        epochs=args.epochs,
        max_train_records=args.max_train_records,
        max_validation_records=args.max_validation_records,
    )
    evidence_hash = hash_file(args.run_root / "smoke" / "smoke_evidence.json")
    lifecycle.advance(
        ExperimentState.SMOKE_PASSED,
        {"smoke_evidence_hash": evidence_hash, "test_labels_read": False},
    )
    _write_lifecycle(args.run_root, lifecycle)
    _refresh_manifest(args.run_root, lifecycle.experiment_id)
    return {"state": lifecycle.state.value, **evidence}


def cv_run(args: argparse.Namespace) -> dict[str, Any]:
    lifecycle = _load_lifecycle(args.run_root)
    _require_state(lifecycle, ExperimentState.SMOKE_PASSED)
    manifest = _load_manifest(args.run_root)
    config = _read_json(args.run_root / "run_config.json")
    archive = FaciesArchive(config["task_id"], args.processed_root)
    development = archive.development_index()
    device = resolve_device(args.device)

    def fold_runner(fold: Fold) -> Mapping[str, Any]:
        return run_development_fold(
            fold=fold,
            archive=archive,
            development_records=development,
            run_config=config,
            manifest=manifest,
            output_dir=args.run_root / "folds" / f"fold_{fold.fold_id}",
            epochs=args.epochs,
            device=device,
        )

    summary = run_development_cv(
        manifest,
        fold_runner,
        output_dir=args.run_root,
        primary_metric="miou",
        metric_direction="maximize",
    )
    summary_hash = hash_file(args.run_root / "oof" / "summary.json")
    lifecycle.advance(
        ExperimentState.CV_COMPLETE,
        {
            "oof_hash": summary_hash,
            "primary_metric": "miou",
            "test_access": False,
        },
    )
    _write_lifecycle(args.run_root, lifecycle)
    _refresh_manifest(args.run_root, lifecycle.experiment_id)
    return {"state": lifecycle.state.value, **summary}


def freeze_run(args: argparse.Namespace) -> dict[str, Any]:
    lifecycle = _load_lifecycle(args.run_root)
    _require_state(lifecycle, ExperimentState.CV_COMPLETE)
    manifest = _load_manifest(args.run_root)
    base_config = _read_json(args.run_root / "run_config.json")
    summary = _read_json(args.run_root / "oof" / "summary.json")
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for fold in manifest.folds:
        path = args.run_root / "folds" / f"fold_{fold.fold_id}" / "calibration_logits.npz"
        with np.load(path, allow_pickle=False) as archive:
            logits.append(np.asarray(archive["logits"], dtype=np.float32))
            labels.append(np.asarray(archive["labels"], dtype=np.int64))
    calibration = fit_temperature_scaling(np.concatenate(logits), np.concatenate(labels))
    selected_temperature = calibration.temperature if calibration.nll_after <= calibration.nll_before else 1.0
    best_epochs = [int(record["best_epoch"]) + 1 for record in summary["folds"]]
    final_epochs = max(1, int(statistics.median(best_epochs)))
    frozen_config = {
        **{key: value for key, value in base_config.items() if key != "processed_root"},
        "temperature": selected_temperature,
        "calibration_method": (
            calibration.method if selected_temperature != 1.0 else "identity_selected_by_oof_nll"
        ),
        "final_epochs": final_epochs,
        "epoch_rule": "median_cv_best_epoch_one_based",
        "selection_source": "development_oof_only",
        "split_hash": manifest.stable_hash(),
    }
    frozen_config["config_hash"] = hash_payload(frozen_config)
    atomic_write_json(args.run_root / "frozen_config.json", frozen_config)
    atomic_write_json(
        args.run_root / "oof" / "calibration.json",
        {**calibration.to_dict(), "selected_temperature": selected_temperature},
    )
    lifecycle.advance(
        ExperimentState.CONFIG_FROZEN,
        {
            "config_hash": frozen_config["config_hash"],
            "oof_hash": lifecycle.evidence[ExperimentState.CV_COMPLETE.value]["oof_hash"],
            "temperature": selected_temperature,
            "final_epochs": final_epochs,
        },
    )
    _write_lifecycle(args.run_root, lifecycle)
    _refresh_manifest(args.run_root, lifecycle.experiment_id)
    return {
        "state": lifecycle.state.value,
        "config_hash": frozen_config["config_hash"],
        "temperature": selected_temperature,
        "final_epochs": final_epochs,
        "oof_nll_before": calibration.nll_before,
        "oof_nll_after": calibration.nll_after,
    }


def refit_run(args: argparse.Namespace) -> dict[str, Any]:
    lifecycle = _load_lifecycle(args.run_root)
    _require_state(lifecycle, ExperimentState.CONFIG_FROZEN)
    manifest = _load_manifest(args.run_root)
    config = _read_json(args.run_root / "frozen_config.json")
    archive = FaciesArchive(config["task_id"], args.processed_root)
    indexed = records_by_id(archive.development_index())
    development = tuple(indexed[sample_id] for sample_id in manifest.development_sample_ids)
    preprocessor = archive.fit_preprocessor(development, method=str(config["normalization"]))
    atomic_write_json(args.run_root / "refit" / "preprocess_stats.json", preprocessor.to_dict())
    # The epoch count is already frozen from CV.  The shared trainer requires a
    # validation factory, so refit replays development without using its best
    # checkpoint; checkpoint_last is the only final refit artifact consumed.
    train_development_model(
        archive=archive,
        train_records=development,
        validation_records=development,
        preprocessor=preprocessor,
        run_config=config,
        split_hash=manifest.stable_hash(),
        output_dir=args.run_root / "refit",
        epochs=int(config["final_epochs"]),
        fold_id=-1,
        device=resolve_device(args.device),
    )
    checkpoint_path = args.run_root / "refit" / "checkpoint_last.pkl"
    checkpoint_hash = hash_file(checkpoint_path)
    atomic_write_json(
        args.run_root / "refit" / "refit_evidence.json",
        {
            "checkpoint": "checkpoint_last.pkl",
            "checkpoint_hash": checkpoint_hash,
            "config_hash": config["config_hash"],
            "split_hash": manifest.stable_hash(),
            "fit_scope": "all_declared_development_core_samples",
            "epoch_selection": "frozen_before_refit",
            "validation_replay_not_used_for_selection": True,
        },
    )
    lifecycle.advance(
        ExperimentState.REFIT_COMPLETE,
        {
            "checkpoint_hash": checkpoint_hash,
            "config_hash": config["config_hash"],
            "split_hash": manifest.stable_hash(),
        },
    )
    _write_lifecycle(args.run_root, lifecycle)
    _refresh_manifest(args.run_root, lifecycle.experiment_id)
    return {
        "state": lifecycle.state.value,
        "checkpoint_hash": checkpoint_hash,
        "development_samples": len(development),
    }


def frozen_test_run(args: argparse.Namespace) -> dict[str, Any]:
    lifecycle = _load_lifecycle(args.run_root)
    _require_state(lifecycle, ExperimentState.REFIT_COMPLETE)
    manifest = _load_manifest(args.run_root)
    config = _read_json(args.run_root / "frozen_config.json")
    checkpoint_path = args.run_root / "refit" / "checkpoint_last.pkl"
    checkpoint_hash = hash_file(checkpoint_path)
    lifecycle.consume_test(
        config_hash=config["config_hash"],
        checkpoint_hash=checkpoint_hash,
        split_hash=manifest.stable_hash(),
    )
    # Persist consumption before opening labels: failures after this point are
    # fail-closed and cannot silently rerun the same campaign.
    _write_lifecycle(args.run_root, lifecycle)
    archive = FaciesArchive(config["task_id"], args.processed_root)
    frozen_test = archive.frozen_test_index(labels_consumed=True)
    if set(record.sample_id for record in frozen_test) != set(manifest.test_sample_ids):
        raise RuntimeError("frozen-test archive differs from the locked split manifest")
    preprocessor = FoldPreprocessor.from_dict(
        _read_json(args.run_root / "refit" / "preprocess_stats.json")
    )
    spec = get_task_spec(config["task_id"])
    model = discover_model("facies", str(config["model_id"])).build(
        spec,
        num_classes=int(spec.metadata["num_classes"]),
        **dict(config.get("model_config", {})),
    )
    if not isinstance(model, torch.nn.Module):
        raise TypeError("facies model registry did not return torch.nn.Module")
    device = resolve_device(args.device)
    model = model.to(device)
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    prediction = predict_consumed_frozen_test(
        lifecycle=lifecycle,
        model=model,
        archive=archive,
        records=frozen_test,
        preprocessor=preprocessor,
        batch_size=int(config["batch_size"]),
        device=device,
        temperature=float(config["temperature"]),
    )
    metrics = dict(prediction["metrics"])
    metrics["artifact_context"] = {
        "track_id": "facies",
        "task_id": config["task_id"],
        "label_version": spec.label_version,
        "run_id": lifecycle.experiment_id,
        "config_hash": config["config_hash"],
        "checkpoint_hash": checkpoint_hash,
        "split_hash": manifest.stable_hash(),
        "test_consumed_at": lifecycle.test_consumed_at,
        "temperature": config["temperature"],
    }
    prediction["metrics"] = metrics
    prediction_path, metrics_path = archive_prediction_maps(
        prediction, output_dir=args.run_root / "frozen_test"
    )
    _refresh_manifest(args.run_root, lifecycle.experiment_id)
    return {
        "state": lifecycle.state.value,
        "prediction_archive": prediction_path.relative_to(args.run_root).as_posix(),
        "metrics": metrics_path.relative_to(args.run_root).as_posix(),
        "accuracy": metrics["accuracy"],
        "miou": metrics["miou"],
        "macro_f1": metrics["macro_f1"],
        "test_consumed_at": lifecycle.test_consumed_at,
    }


def visualize_run(args: argparse.Namespace) -> dict[str, Any]:
    lifecycle = _load_lifecycle(args.run_root)
    if lifecycle.state not in {ExperimentState.TEST_CONSUMED, ExperimentState.VERIFIED}:
        raise RuntimeError("visualization requires archived frozen-test predictions")
    output = args.run_root / "visualizations" / "facies_diagnostics.png"
    sidecar = args.run_root / "visualizations" / "facies_diagnostics.json"
    render_archived_diagnostics(
        prediction_path=args.run_root / "frozen_test" / "predictions.npz",
        metrics_path=args.run_root / "frozen_test" / "metrics.json",
        output_path=output,
        sidecar_path=sidecar,
        diagnostic_seed=args.diagnostic_seed,
    )
    _refresh_manifest(args.run_root, lifecycle.experiment_id)
    return {
        "state": lifecycle.state.value,
        "visualization": output.relative_to(args.run_root).as_posix(),
        "sidecar": sidecar.relative_to(args.run_root).as_posix(),
        "read_archived_predictions_only": True,
    }


def hpo_plan_run(args: argparse.Namespace) -> dict[str, Any]:
    lifecycle = _load_lifecycle(args.run_root)
    lifecycle.require_development_access()
    plan = hpo_contract()
    atomic_write_json(args.run_root / "hpo" / "plan.json", plan)
    _refresh_manifest(args.run_root, lifecycle.experiment_id)
    return plan


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="freeze TaskSpec and spatial split; test labels stay unread")
    prepare.add_argument("--task", choices=TASK_IDS, required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--processed-root", type=Path, required=True)
    prepare.add_argument("--requested-n-splits", type=int, default=5)
    prepare.add_argument("--buffer-groups", type=int)

    smoke = commands.add_parser("smoke", help="bounded real-data development smoke")
    _add_runtime(smoke)
    smoke.add_argument("--epochs", type=int, default=1)
    smoke.add_argument("--max-train-records", type=int, default=32)
    smoke.add_argument("--max-validation-records", type=int, default=16)

    cv = commands.add_parser("cv", help="development-only spatial CV; cannot accept test")
    _add_runtime(cv)
    cv.add_argument("--epochs", type=int, default=20)

    freeze = commands.add_parser("freeze", help="freeze config/temperature/epoch from OOF")
    freeze.add_argument("--run-root", type=Path, required=True)

    refit = commands.add_parser("refit", help="fixed-epoch development refit")
    _add_runtime(refit)

    frozen_test = commands.add_parser("test", help="single-use frozen-test campaign")
    _add_runtime(frozen_test)

    visualize = commands.add_parser("visualize", help="render archived test predictions only")
    visualize.add_argument("--run-root", type=Path, required=True)
    visualize.add_argument("--diagnostic-seed", type=int, default=2693)

    hpo = commands.add_parser("hpo-plan", help="write the fixed development-only HPO plan")
    hpo.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handlers = {
        "prepare": prepare_run,
        "smoke": smoke_run,
        "cv": cv_run,
        "freeze": freeze_run,
        "refit": refit_run,
        "test": frozen_test_run,
        "visualize": visualize_run,
        "hpo-plan": hpo_plan_run,
    }
    result = handlers[args.command](args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
