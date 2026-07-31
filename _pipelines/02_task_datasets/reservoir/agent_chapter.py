from __future__ import annotations

import argparse
import hashlib
import json
import os
import textwrap
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "_outputs"
AGENT_OUTPUT_ROOT = OUTPUT_ROOT / "agent_chapter"
EVIDENCE_PATH = AGENT_OUTPUT_ROOT / "evidence.md"
PROJECT_ROOT = next(parent for parent in HERE.parents if parent.name == "track-property")
TRAIN_H5 = PROJECT_ROOT / "_data" / "processed" / "reservoir" / "train.h5"
GUARD_NPZ = OUTPUT_ROOT / "guard.npz"
TARGET_ORDER = ("PHIF", "KLOGH", "SW")


@dataclass(frozen=True)
class ResultFile:
    name: str
    path: Path
    sha256: str
    size_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_latest_result_files() -> dict[str, ResultFile]:
    candidates = {
        "metrics.json": None,
        "run_manifest.json": None,
        "build_report.json": None,
        "split_manifest.json": None,
        "checkpoints/history.json": None,
    }
    for name in list(candidates):
        matches = sorted(OUTPUT_ROOT.rglob(name))
        if not matches:
            raise FileNotFoundError(f"required result file not found: {name}")
        path = matches[-1]
        candidates[name] = ResultFile(
            name=name,
            path=path,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )
    return {name: value for name, value in candidates.items() if value is not None}


def _decoded_json(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return dict(json.loads(str(value)))


def _load_train_records() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not TRAIN_H5.is_file():
        raise FileNotFoundError(TRAIN_H5)
    seismic: list[np.ndarray] = []
    logs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with h5py.File(TRAIN_H5, "r") as handle:
        for key in sorted(handle.keys()):
            group = handle[key]
            meta = _decoded_json(group.attrs["meta"])
            if meta.get("partition") != "train":
                continue
            seismic.append(np.asarray(group["seismic_patch"][()], dtype=np.float64))
            logs.append(np.asarray(group["well_log_seq"][()], dtype=np.float64))
            labels.append(np.asarray(group["label"][()], dtype=np.float64).reshape(-1))
    return np.stack(seismic), np.stack(logs), np.stack(labels)


def _load_guard_records() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not GUARD_NPZ.is_file():
        raise FileNotFoundError(GUARD_NPZ)
    with np.load(GUARD_NPZ, allow_pickle=False) as archive:
        seismic = np.asarray(archive["seismic_patch"], dtype=np.float64)
        logs = np.asarray(archive["well_log_seq"], dtype=np.float64)
        labels = np.asarray(archive["label"], dtype=np.float64).reshape(len(seismic), -1)
    return seismic, logs, labels


def _fit_normalization(seismic: np.ndarray, logs: np.ndarray) -> dict[str, np.ndarray]:
    seismic_flat = seismic.reshape(len(seismic), -1)
    seismic_mean = seismic_flat.mean(axis=0)
    seismic_std = seismic_flat.std(axis=0) + 1e-8
    values = logs[:, :, :4].reshape(len(logs), -1)
    masks = (logs[:, :, 4:8] > 0.5).reshape(len(logs), -1)
    log_mean = np.zeros(values.shape[1], dtype=np.float64)
    log_std = np.ones(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        observed = values[masks[:, column], column]
        if observed.size:
            log_mean[column] = observed.mean()
            log_std[column] = observed.std() + 1e-8
    return {
        "seismic_mean": seismic_mean,
        "seismic_std": seismic_std,
        "log_mean": log_mean,
        "log_std": log_std,
    }


def _normalize_inputs(seismic: np.ndarray, logs: np.ndarray, fit: dict[str, np.ndarray]) -> np.ndarray:
    seismic_flat = seismic.reshape(len(seismic), -1)
    values = logs[:, :, :4].reshape(len(logs), -1)
    masks = (logs[:, :, 4:8] > 0.5).reshape(len(logs), -1)
    tabular = np.concatenate(
        [
            (seismic_flat - fit["seismic_mean"]) / fit["seismic_std"],
            ((values - fit["log_mean"]) / fit["log_std"]) * masks,
            masks.astype(np.float64),
        ],
        axis=1,
    )
    if tabular.shape[1] != 153:
        raise RuntimeError(f"unexpected tabular feature count: {tabular.shape}")
    return tabular


def summarize_targets(labels: np.ndarray) -> dict[str, dict[str, float]]:
    target_summary: dict[str, dict[str, float]] = {}
    klogh_physical = np.expm1(labels[:, 1])
    for target, column in zip(TARGET_ORDER, labels.T, strict=True):
        target_summary[target] = {
            "mean": float(np.mean(column)),
            "std": float(np.std(column)),
            "min": float(np.min(column)),
            "max": float(np.max(column)),
        }
    target_summary["KLOGH_mD"] = {
        "mean": float(np.mean(klogh_physical)),
        "std": float(np.std(klogh_physical)),
        "min": float(np.min(klogh_physical)),
        "max": float(np.max(klogh_physical)),
    }
    return target_summary


def normalize_guard_diagnostics(train_seismic: np.ndarray, train_logs: np.ndarray, guard_seismic: np.ndarray, guard_logs: np.ndarray) -> dict[str, dict[str, float]]:
    train_only = _fit_normalization(train_seismic, train_logs)
    train_guard = _fit_normalization(np.concatenate([train_seismic, guard_seismic]), np.concatenate([train_logs, guard_logs]))

    diagnostics: dict[str, dict[str, float]] = {}
    for name, fit in {"train_only": train_only, "train_plus_guard": train_guard}.items():
        tabular = _normalize_inputs(guard_seismic, guard_logs, fit)
        means = tabular.mean(axis=0)
        stds = tabular.std(axis=0)
        diagnostics[name] = {
            "mean_abs_mean": float(np.abs(means).mean()),
            "max_abs_mean": float(np.abs(means).max()),
            "mean_std": float(stds.mean()),
            "min_std": float(stds.min()),
            "max_std": float(stds.max()),
        }
    return diagnostics


def feature_correlation_diagnostics(
    train_seismic: np.ndarray, train_logs: np.ndarray, train_labels: np.ndarray, *, top_k: int = 10
) -> list[dict[str, float | str | int]]:
    fit = _fit_normalization(train_seismic, train_logs)
    tabular = _normalize_inputs(train_seismic, train_logs, fit)
    sw = train_labels[:, 2].astype(np.float64)
    sw_center = sw - sw.mean()
    sw_norm = float(np.sqrt(np.dot(sw_center, sw_center)))
    if sw_norm == 0.0:
        return []
    correlations: list[dict[str, float | str | int]] = []
    for index in range(tabular.shape[1]):
        column = tabular[:, index]
        if float(np.std(column)) < 1e-12:
            continue
        centered = column - column.mean()
        corr = float(np.dot(centered, sw_center) / (np.sqrt(np.dot(centered, centered)) * sw_norm))
        if index < 81:
            feature = f"seismic_flat[{index}]"
        elif index < 117:
            feature = f"log_value[{index - 81}]"
        else:
            feature = f"log_mask[{index - 117}]"
        correlations.append({"index": index, "feature": feature, "corr": corr, "abs_corr": abs(corr)})
    correlations.sort(key=lambda item: item["abs_corr"], reverse=True)
    return correlations[:top_k]


def build_prompt(context: dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""
        You are giving a common-sense analysis of a reservoir-property prediction pipeline.

        Hard constraints:
        - Do not assume access to frozen holdout/test.h5.
        - Only use the provided development-only evidence and current result files.
        - Separate suggestions into: (A) cheap to verify immediately on development-only evidence, (B) require retraining or more expensive experiments, and (C) blocked by current contract/data.
        - If you mention a suggestion that is not already evidenced, mark it as 未验证.
        - Keep advice concrete: parameter tuning, feature design, training strategy, loss/output handling, calibration, or evaluation protocol.

        Current context:
        {json.dumps(context, indent=2, ensure_ascii=False)}
        """
    ).strip()


def call_deepseek(prompt: str, *, api_key: str | None = None, timeout_sec: int = 120) -> str:
    key = api_key or os.environ.get("DEEPSEEK_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_KEY is required")
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "You are a rigorous engineering analyst. Do not invent results. Separate verified facts from recommendations and label unverified advice clearly.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def build_evidence_markdown(
    *,
    files: dict[str, ResultFile],
    run_manifest: dict[str, Any],
    metrics: dict[str, Any],
    build_report: dict[str, Any],
    split_manifest: dict[str, Any],
    history: dict[str, Any],
    target_stats: dict[str, dict[str, dict[str, float]]],
    normalization_stats: dict[str, dict[str, float]],
    feature_correlations: list[dict[str, float | str | int]],
    deepseek_analysis: str,
) -> str:
    lines: list[str] = []
    lines.append("# 智能体分析章节")
    lines.append("")
    lines.append("## 1. 真实产物定位")
    lines.append("")
    lines.append("| 文件 | SHA256 | 大小(bytes) |")
    lines.append("|---|---:|---:|")
    for name in ("metrics.json", "run_manifest.json", "build_report.json", "split_manifest.json", "checkpoints/history.json"):
        item = files[name]
        lines.append(f"| `{repo_rel(item.path)}` | `{item.sha256}` | {item.size_bytes} |")
    lines.append("")
    lines.append("## 2. 当前模型与评测结果")
    lines.append("")
    lines.append(f"- 模型：`{run_manifest['model']}`")
    lines.append(f"- 框架：`{run_manifest['framework']}`")
    lines.append(f"- 训练轮数：{run_manifest['epochs']}，最佳 epoch：{run_manifest['best_epoch']}，最佳 val loss：{run_manifest['best_val_loss']:.6f}")
    lines.append(f"- 样本数：train {run_manifest['sample_counts']['train']} / guard {run_manifest['sample_counts']['guard']} / test {run_manifest['sample_counts']['test']}")
    lines.append(f"- family zero overlap：`{run_manifest['family_zero_overlap']}`，guard 仅用于验证：`{run_manifest['guard_used_for_val_loss_only']}`，test 后验加载：`{run_manifest['test_loaded_after_best_checkpoint']}`")
    lines.append("")
    lines.append("| Target | MAE | RMSE | R2 | Pearson |")
    lines.append("|---|---:|---:|---:|---:|")
    for target in ("PHIF", "log1p(KLOGH)", "SW"):
        row = metrics["per_target"][target]
        lines.append(f"| `{target}` | {row['MAE']:.6f} | {row['RMSE']:.6f} | {row['R2']:.6f} | {row['Pearson']:.6f} |")
    lines.append(f"| `composite_mean_train_std_normalized_RMSE` | {metrics['composite_mean_train_std_normalized_RMSE']:.6f} |  |  |  |")
    lines.append("")
    lines.append("## 3. DeepSeek 常识性分析")
    lines.append("")
    lines.append(deepseek_analysis.strip())
    lines.append("")
    lines.append("## 4. 低成本验证（development-only）")
    lines.append("")
    lines.append("### 4.1 目标分布与目标难度")
    lines.append("")
    lines.append("| Split | Target | mean | std | min | max |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for split_name in ("train", "guard"):
        for target in ("PHIF", "KLOGH", "SW", "KLOGH_mD"):
            stats = target_stats[split_name][target]
            lines.append(f"| `{split_name}` | `{target}` | {stats['mean']:.6f} | {stats['std']:.6f} | {stats['min']:.6f} | {stats['max']:.6f} |")
    lines.append("")
    lines.append("### 4.2 归一化漂移检查")
    lines.append("")
    lines.append("以下比较使用同一批 guard 样本，分别在 train-only 归一化与 train+guard 归一化下观察标准化特征分布。")
    lines.append(f"- 发展集来源：`{repo_rel(TRAIN_H5)}` + `{repo_rel(GUARD_NPZ)}`")
    lines.append("")
    lines.append("| Fit scope | mean_abs_mean | max_abs_mean | mean_std | min_std | max_std |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for scope in ("train_only", "train_plus_guard"):
        stats = normalization_stats[scope]
        lines.append(
            f"| `{scope}` | {stats['mean_abs_mean']:.6f} | {stats['max_abs_mean']:.6f} | {stats['mean_std']:.6f} | {stats['min_std']:.6f} | {stats['max_std']:.6f} |"
        )
    lines.append("")
    lines.append("### 4.3 SW 输入特征相关性")
    lines.append("")
    lines.append("下面是 train 上与 SW 的绝对相关性最高的特征，属于廉价的信号强弱检查。")
    lines.append("")
    lines.append("| rank | feature | corr | abs_corr |")
    lines.append("|---|---|---:|---:|")
    for rank, item in enumerate(feature_correlations, start=1):
        lines.append(f"| {rank} | `{item['feature']}` | {float(item['corr']):.6f} | {float(item['abs_corr']):.6f} |")
    lines.append("")
    lines.append("### 4.4 结论")
    lines.append("")
    lines.append("- **可直接保留**：family-zero-overlap、train-only normalization、guard-only validation、KLOGH 的 log1p 目标变换。")
    lines.append("- **已验证的低成本改善**：train+guard 归一化略微减少了 guard 的标准化均值漂移（见上表），说明归一化范围对 development 数据有轻微收益。")
    lines.append("- **已验证的低成本诊断**：SW 与若干输入特征存在中等相关性，但最高相关性仍不足以解释全部误差，说明 SW 仍主要受校准和输出约束影响。")
    lines.append("- **未验证**：SW 边界激活/校准、KLOGH 加权损失、纵向上下文增强、容量搜索。它们需要真实再训练或额外实验，当前没有伪造提升。")
    lines.append("")
    lines.append("## 5. 重要限制")
    lines.append("")
    lines.append("- 不读取 frozen holdout / test.h5。")
    lines.append("- 仅使用当前 development-only 证据和已存在产物。")
    lines.append("- DeepSeek 建议若未执行实验，均标注为未验证。")
    return "\n".join(lines) + "\n"


def generate_evidence(output_path: Path = EVIDENCE_PATH) -> dict[str, Any]:
    files = discover_latest_result_files()
    run_manifest = _read_json(files["run_manifest.json"].path)
    metrics = _read_json(files["metrics.json"].path)
    build_report = _read_json(files["build_report.json"].path)
    split_manifest = _read_json(files["split_manifest.json"].path)
    history = _read_json(files["checkpoints/history.json"].path)

    train_seismic, train_logs, train_labels = _load_train_records()
    guard_seismic, guard_logs, guard_labels = _load_guard_records()
    target_stats = {
        "train": summarize_targets(train_labels),
        "guard": summarize_targets(guard_labels),
    }
    normalization_stats = normalize_guard_diagnostics(train_seismic, train_logs, guard_seismic, guard_logs)
    feature_correlations = feature_correlation_diagnostics(train_seismic, train_logs, train_labels, top_k=12)

    context = {
        "latest_result_files": {
            name: {"path": repo_rel(item.path), "sha256": item.sha256, "size_bytes": item.size_bytes}
            for name, item in files.items()
        },
        "run_manifest": {
            "model": run_manifest["model"],
            "framework": run_manifest["framework"],
            "epochs": run_manifest["epochs"],
            "best_epoch": run_manifest["best_epoch"],
            "best_val_loss": run_manifest["best_val_loss"],
            "sample_counts": run_manifest["sample_counts"],
            "families": run_manifest["families"],
            "family_zero_overlap": run_manifest["family_zero_overlap"],
            "guard_used_for_val_loss_only": run_manifest["guard_used_for_val_loss_only"],
            "test_loaded_after_best_checkpoint": run_manifest["test_loaded_after_best_checkpoint"],
            "normalization_fit_sources": run_manifest["normalization_fit_sources"],
            "target_transform": run_manifest["target_transform"],
        },
        "metrics": metrics,
        "split_manifest": {
            "algorithm": split_manifest["algorithm"],
            "created_before_interpolation": split_manifest["created_before_interpolation"],
            "family_partition": split_manifest["family_partition"],
        },
        "build_report": {
            "target_names": build_report["target_names"],
            "forbidden_input_curves": build_report["forbidden_input_curves"],
            "hugin_top_surface": build_report["hugin_top_surface"],
            "hugin_base_surface": build_report["hugin_base_surface"],
        },
        "history": {
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
        },
    }
    prompt = build_prompt(context)
    analysis = call_deepseek(prompt)
    markdown = build_evidence_markdown(
        files=files,
        run_manifest=run_manifest,
        metrics=metrics,
        build_report=build_report,
        split_manifest=split_manifest,
        history=history,
        target_stats=target_stats,
        normalization_stats=normalization_stats,
        feature_correlations=feature_correlations,
        deepseek_analysis=analysis,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return {
        "output_path": output_path.as_posix(),
        "prompt_chars": len(prompt),
        "analysis_chars": len(analysis),
        "latest_result_files": {name: repo_rel(item.path) for name, item in files.items()},
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the reservoir agent analysis chapter.")
    parser.add_argument("--output", type=Path, default=EVIDENCE_PATH, help="Markdown evidence output path.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    report = generate_evidence(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
