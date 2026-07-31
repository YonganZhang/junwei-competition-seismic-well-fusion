#!/usr/bin/env python3
"""Development-only DeepSeek analysis chapter for the lithofacies pipeline.

The chapter has three explicit stages:

``consult``
    Build a structured prompt from committed P11 evidence and the immutable
    development LOGO4 batch, then call the OpenAI-compatible DeepSeek API.
    The API key is read only from ``DEEPSEEK_KEY`` and is never serialized.
``run``
    Evaluate the three low-cost suggestions selected from the consultation on
    the same four LOGO folds and three frozen seeds as the XGBoost baseline.
``verify``
    Recompute portable summaries, verify artifact hashes and prove that all
    earlier P11 evidence remains byte-identical.

No command accepts or opens frozen-holdout/test inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from _models.lithofacies.p5_adapter_common import (  # noqa: E402
    multimodal_numpy_features,
)
from lithofacies_p5_stage3 import (  # noqa: E402
    FOLD_IDS,
    REPEAT_SEEDS,
    _fold_arrays,
    load_stage3_batch,
)
from p4_contract import classification_metrics_from_logits  # noqa: E402


SCHEMA_VERSION = "lithofacies-agent-chapter/v1"
CONSULTATION_SCHEMA = "lithofacies-agent-consultation/v1"
EXPECTED_SPLIT_HASH = (
    "a06375429f9e9cf380fb5cdebd7d0cb7b25d7a13d29522b8e2420f4dae1b4555"
)
NUM_CLASSES = 9
BASELINE_MEAN = 0.19493770207563763
CURRENT_BEST_MEAN = 0.20218697566969132
MINIMUM_PROMOTION_DELTA = 0.005
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
DEEPSEEK_REQUEST_MODEL = "deepseek-chat"
DEFAULT_OUTPUT_DIR = TRACK_DIR / "_outputs" / "agent_chapter"
VARIANTS = (
    "baseline_archived",
    "baseline_reproduced",
    "weight_alpha_075",
    "weight_alpha_100",
    "well_and_mask_only_858",
    "depth3_eta01_rounds60",
    "depth3_eta01_rounds60_prior025",
)
SUGGESTION_MAP = {
    "baseline_archived": "reference",
    "baseline_reproduced": "evaluation_alignment",
    "weight_alpha_075": "DeepSeek-S1",
    "weight_alpha_100": "DeepSeek-S1/E1",
    "well_and_mask_only_858": "DeepSeek-S2/E2",
    "depth3_eta01_rounds60": "DeepSeek-S3/E3",
    "depth3_eta01_rounds60_prior025": "adaptive-P11-prior-followup",
}
PRESERVED_OUTPUTS = {
    "_outputs/p11_cross_attention_fusion/artifact_manifest.json": (
        "2182207d9c7e0e1da5e190e64f9b96490dd1537cd02c81b27e0ecc8862808d30"
    ),
    "_outputs/p11_cross_attention_fusion/evidence.md": (
        "d7892d210451459ce25af42ccab1b98ec6a9d2f5c13360e3a36375bb7ca5e5d0"
    ),
    "_outputs/p11_cross_attention_fusion/primary_metric.png": (
        "5b4e95cb7a7a4091e403f738621a0b7472641ac0a711aa2d9d43c529b762ae05"
    ),
    "_outputs/p11_cross_attention_fusion/results.jsonl": (
        "44e5dd5210c75ab04e8fa9b286d4c9d890a2a32c57ac2ddf21b9f9337d6c343c"
    ),
    "_outputs/p11_cross_attention_fusion/summary.json": (
        "fa4049788c9bb60bd7ca91c4f3b08d9b97f0d1ec6e6130577192a0f4fde1beeb"
    ),
    "_outputs/p11_clean_well_native33/artifact_manifest.json": (
        "5e1d5b024a78313acd2953edb5b78ccc87c0969d1b706dc7f34b6b1df6c7ca06"
    ),
    "_outputs/p11_clean_well_native33/evidence.md": (
        "420f778568a156b2d0f7c45cafc97d64ae75afeb4217cecb93becb57a1d24e5d"
    ),
    "_outputs/p11_clean_well_native33/primary_metric.png": (
        "5710fad16b433d8a1e58974cb3d7adae361a648bab0b7a7907b25b70941519e1"
    ),
    "_outputs/p11_clean_well_native33/results.jsonl": (
        "b239ed0841a1e09365d1c77deabe912ce8d55bb93904b9e1f27c75fb619b6a59"
    ),
    "_outputs/p11_clean_well_native33/summary.json": (
        "d6058f791cf40f7597e0b820c5379df69e7f1e581f686b71cd497d547cce71ab"
    ),
    "_outputs/p11_residual_fusion/artifact_manifest.json": (
        "d81c3dc0647020186004c3630825d732e8ad8d18c975131c20ec14be0e97be14"
    ),
    "_outputs/p11_residual_fusion/evidence.md": (
        "eac5c4e31c86dcd080993fc82fee5958b41402edc7dfee666f97b6c80bc2d1f6"
    ),
    "_outputs/p11_residual_fusion/primary_metric.png": (
        "c3a82df5905027027078c9545dc6afe0c74c9dd6174775a59cbc72bc32c739db"
    ),
    "_outputs/p11_residual_fusion/results.jsonl": (
        "9c73c5ead55a4ee3e472368dbfdadc811ce92da5a2ffa321601cc23b550e7e3c"
    ),
    "_outputs/p11_residual_fusion/summary.json": (
        "86d1cda65a9e67e92dff1278fe6bed423fd5d4a8de918547d20eac1b60a75466"
    ),
}
FORBIDDEN_PATH_MARKERS = ("test", "holdout", "frozen")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ensure_development_only_paths(paths: Iterable[Path]) -> None:
    """Reject holdout-like path components before any input is opened."""
    for raw_path in paths:
        lowered = [
            part.lower().replace("-", "_") for part in Path(raw_path).parts
        ]
        if any(
            marker in part
            for part in lowered
            for marker in FORBIDDEN_PATH_MARKERS
        ):
            raise ValueError(f"forbidden holdout path: {raw_path}")


def verify_preserved_outputs() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in PRESERVED_OUTPUTS.items():
        path = TRACK_DIR / relative
        if not path.is_file():
            raise FileNotFoundError(f"preserved P11 artifact is missing: {path}")
        digest = _sha256(path)
        if digest != expected:
            raise RuntimeError(f"preserved P11 artifact changed: {path}")
        observed[relative] = digest
    return observed


def _load_context(
    development_batch: Path,
    p11_summary: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any], list[int]]:
    ensure_development_only_paths((development_batch, p11_summary))
    arrays, manifest = load_stage3_batch(development_batch)
    if (
        manifest.get("split_hash") != EXPECTED_SPLIT_HASH
        or manifest.get("frozen_test_accessed") is not False
        or manifest.get("test_metrics_used") is not False
    ):
        raise RuntimeError("development batch violates the immutable LOGO4 contract")
    summary = json.loads(p11_summary.read_text(encoding="utf-8"))
    if (
        summary.get("schema_version")
        != "lithofacies-p11-cross-attention-fusion/v1"
        or summary.get("evaluation", {}).get("frozen_test_accessed") is not False
        or summary.get("evaluation", {}).get("known_holdout_accessed") is not False
    ):
        raise RuntimeError("P11 summary violates the development-only contract")
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for fold_id in FOLD_IDS:
        labels = np.asarray(
            arrays[f"f{fold_id}_validation_labels"], dtype=np.int64
        )
        class_counts += np.bincount(labels, minlength=NUM_CLASSES)
    return arrays, manifest, summary, class_counts.tolist()


def build_structured_prompt(
    *,
    p11_summary: Mapping[str, Any],
    class_counts: Sequence[int],
    sample_count: int,
) -> tuple[str, str]:
    """Return the system and user prompts using only auditable facts."""
    baseline = p11_summary["variants"]["baseline"]["metrics"]
    calibrated = p11_summary["variants"]["prior_calibrated"]["metrics"]
    fusion = p11_summary["variants"]["cross_attention"]["metrics"]
    native_summary = json.loads(
        (TRACK_DIR / "_outputs/p11_clean_well_native33/summary.json").read_text(
            encoding="utf-8"
        )
    )
    pretrained_minus_random = native_summary["representation_diagnostic"][
        "pretrained_minus_random_fixed_schema_macro_f1"
    ]
    nonzero = [int(value) for value in class_counts if int(value) > 0]
    imbalance_ratio = max(nonzero) / min(nonzero)
    system = (
        "你是一名谨慎的地学机器学习顾问。只能基于用户提供的development结果做常识性分析；"
        "不得虚构提升数字，不得把相关性或系统整体增益归因于某个组件。"
        "请优先给出可在严格group-CV下低成本证伪的建议。"
    )
    user = f"""# 任务描述
GM09岩相预测：根据同一深度中心附近的测井序列和3x3地震空间patch，预测固定schema的9类成因岩相。
主指标为fixed_schema_macro_f1（9类始终进入macro平均，某fold缺类也不删类）。
评测是development-only严格LOGO4：按4个母井家族leave-one-group-out，4折×3个固定seed；禁止读取frozen holdout/test.h5。

# 数据特点
development唯一覆盖共{sample_count}样本，9类支持度依次为{list(class_counts)}，最大/最小非零为{imbalance_ratio:g}:1。
各fold训练样本315或320，验证样本87到132；部分验证fold缺类，fold 2训练集第9类支持度为0。
输入为13条真实测井曲线×33深度点、13条二值缺失mask×33点、3x3×33地震patch。
当前XGBoost把26×33测井（含mask）和3×3×33地震全部展平为1155维。

# 当前模型与真实指标
强baseline：XGBoost multi:softprob，40轮、max_depth=2、eta=0.2、hist、全特征，fold-train内按1/sqrt(class_count)做样本权重。
严格LOGO4×3seed mean fixed_schema_macro_f1={baseline['fixed_schema_macro_f1']['mean']:.10f}，std={baseline['fixed_schema_macro_f1']['std']:.10f}。
fold-train class-prior校准（0.25×centered log count）={calibrated['fixed_schema_macro_f1']['mean']:.10f}，相对baseline {p11_summary['comparison']['prior_calibrated_minus_baseline']:+.10f}。
完整系统（上述校准+以XGBoost叶索引为query、native-33 MOMENT token为key/value的gated residual cross-attention）={fusion['fixed_schema_macro_f1']['mean']:.10f}，相对baseline {p11_summary['comparison']['cross_attention_minus_baseline']:+.10f}，但相对calibration-only {p11_summary['comparison']['cross_attention_minus_prior_calibrated']:+.10f}；不能归因于MOMENT。
早期native-33 pretrained-vs-random差仅{pretrained_minus_random:+.10f}，未达到0.005阈值。
深度logit smoothing曾测试多组，单独最好约+0.00382，但与prior calibration组合不稳定，已拒绝。

# 已知限制
小样本、极端类不均衡、按井/家族domain shift、部分fold训练缺类、当前特征是高维展平。
LOGO development已用于多轮探索，任何新数字只能作为探索性证据，不能当无偏holdout估计。
本轮需要低成本验证，不训练更大的MOMENT，不访问holdout，也不要求提升必须来自大模型。

# 请输出
请用中文按以下固定结构给出：
A. 诊断（3-6条，指出最可能瓶颈）
B. 建议清单（编号S1...；每条含类别=参数调整/特征设计/训练策略/评测诊断，具体做法，预期方向但不要编数字，成本=低/中/高，主要风险）
C. 低成本优先实验（最多3个；给出明确候选参数/特征和同口径LOGO4比较方式）
D. 暂不建议（说明原因）
E. 归因边界（明确不能从当前结果断言MOMENT贡献）
"""
    return system, user


def call_deepseek(
    *,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    timeout_seconds: float = 75.0,
) -> dict[str, Any]:
    """Call DeepSeek without returning or persisting the credential."""
    if not api_key.strip():
        raise RuntimeError("DEEPSEEK_KEY is empty")
    body = {
        "model": DEEPSEEK_REQUEST_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "stream": False,
    }
    request = urllib.request.Request(
        DEEPSEEK_ENDPOINT,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.load(response)
    content = str(payload["choices"][0]["message"]["content"])
    if not all(marker in content for marker in ("A.", "B.", "C.", "D.", "E.")):
        raise RuntimeError("DeepSeek response does not follow the requested structure")
    return {
        "schema_version": CONSULTATION_SCHEMA,
        "endpoint": DEEPSEEK_ENDPOINT,
        "request_model": DEEPSEEK_REQUEST_MODEL,
        "response_model": payload.get("model"),
        "response_id": payload.get("id"),
        "usage": payload.get("usage"),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "prompt_sha256": _stable_hash(
            {"system": system_prompt, "user": user_prompt}
        ),
        "content": content,
        "credential_persisted": False,
    }


def consult(
    *,
    development_batch: Path,
    p11_summary: Path,
    output: Path,
    api_key: str,
) -> dict[str, Any]:
    ensure_development_only_paths((development_batch, p11_summary, output))
    _, manifest, summary, class_counts = _load_context(
        development_batch, p11_summary
    )
    system, user = build_structured_prompt(
        p11_summary=summary,
        class_counts=class_counts,
        sample_count=int(manifest["development_sample_count"]),
    )
    payload = call_deepseek(
        system_prompt=system,
        user_prompt=user,
        api_key=api_key,
    )
    _write_json(output, payload)
    return {
        "response_id": payload["response_id"],
        "response_model": payload["response_model"],
        "prompt_sha256": payload["prompt_sha256"],
        "credential_persisted": False,
    }


def well_and_mask_only_features(
    well: np.ndarray,
    seismic: np.ndarray,
) -> np.ndarray:
    """DeepSeek E2: retain the 13 real logs and 13 masks, drop seismic."""
    logs = np.asarray(well, dtype=np.float32)
    seismic_values = np.asarray(seismic, dtype=np.float32)
    if (
        logs.ndim != 3
        or tuple(logs.shape[1:]) != (26, 33)
        or seismic_values.ndim != 4
        or tuple(seismic_values.shape[1:]) != (3, 3, 33)
        or len(logs) != len(seismic_values)
    ):
        raise ValueError("well-only inputs must match the P11 window schema")
    matrix = logs.reshape(len(logs), -1)
    if matrix.shape != (len(logs), 858) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid well-only feature matrix: {matrix.shape}")
    return np.ascontiguousarray(matrix)


def _train_xgboost(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    class_counts: np.ndarray,
    seed: int,
    weight_alpha: float,
    max_depth: int = 2,
    eta: float = 0.2,
    rounds: int = 40,
) -> np.ndarray:
    import xgboost

    counts = np.asarray(class_counts, dtype=np.float64)
    target = np.asarray(train_labels, dtype=np.int64)
    class_weights = np.zeros(NUM_CLASSES, dtype=np.float64)
    supported = counts > 0
    class_weights[supported] = counts[supported] ** (-float(weight_alpha))
    train_matrix = xgboost.DMatrix(
        train_features,
        label=target,
        weight=class_weights[target],
    )
    booster = xgboost.train(
        {
            "objective": "multi:softprob",
            "num_class": NUM_CLASSES,
            "max_depth": int(max_depth),
            "eta": float(eta),
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
            "seed": int(seed),
            "nthread": 1,
            "verbosity": 0,
        },
        train_matrix,
        num_boost_round=int(rounds),
    )
    probabilities = np.asarray(
        booster.predict(xgboost.DMatrix(validation_features)),
        dtype=np.float64,
    )
    if probabilities.shape != (len(validation_features), NUM_CLASSES):
        raise ValueError("XGBoost did not emit fixed-nine probabilities")
    return np.log(np.clip(probabilities, 1e-12, 1.0)).astype(np.float32)


def prior_calibrate(
    logits: np.ndarray,
    class_counts: np.ndarray,
    *,
    shrinkage: float = 0.25,
) -> np.ndarray:
    """Apply the already documented fold-train-only P11 prior correction."""
    values = np.asarray(logits, dtype=np.float32)
    counts = np.asarray(class_counts, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != NUM_CLASSES:
        raise ValueError("prior calibration requires fixed-nine logits")
    if counts.shape != (NUM_CLASSES,) or np.any(counts < 0):
        raise ValueError("prior calibration requires non-negative class counts")
    log_counts = np.log(np.maximum(counts, 1.0))
    bias = float(shrinkage) * (log_counts - log_counts.mean())
    return (values.astype(np.float64) + bias[None, :]).astype(np.float32)


def _metric_payload(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    metrics = classification_metrics_from_logits(labels, logits)
    return {
        "fixed_schema_macro_f1": float(metrics["fixed_schema_macro_f1"]),
        "supported_class_macro_f1": float(
            metrics["supported_class_macro_f1"]
        ),
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "per_class_f1": [
            float(item["f1"]) for item in metrics["per_class"]
        ],
    }


def _row(
    *,
    fold_id: int,
    repeat_id: int,
    variant: str,
    labels: np.ndarray,
    logits: np.ndarray,
    training: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "track_id": "lithofacies",
        "task_id": "gm09_genetic_facies_9class",
        "fold_id": int(fold_id),
        "repeat_id": int(repeat_id),
        "seed": int(REPEAT_SEEDS[repeat_id]),
        "variant": variant,
        "suggestion": SUGGESTION_MAP[variant],
        "metrics": _metric_payload(labels, logits),
        "training": dict(training),
        "split_hash": EXPECTED_SPLIT_HASH,
        "development_only": True,
        "known_holdout_accessed": False,
        "frozen_test_accessed": False,
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)
    if set(by_variant) != set(VARIANTS):
        raise ValueError("agent chapter variant roster changed")
    variants: dict[str, Any] = {}
    baseline_by_cell = {
        (int(row["fold_id"]), int(row["repeat_id"])): float(
            row["metrics"]["fixed_schema_macro_f1"]
        )
        for row in by_variant["baseline_archived"]
    }
    for variant in VARIANTS:
        selected = by_variant[variant]
        if len(selected) != len(FOLD_IDS) * len(REPEAT_SEEDS):
            raise ValueError(f"agent chapter has incomplete cells for {variant}")
        primary = np.asarray(
            [row["metrics"]["fixed_schema_macro_f1"] for row in selected],
            dtype=np.float64,
        )
        per_class = np.asarray(
            [row["metrics"]["per_class_f1"] for row in selected],
            dtype=np.float64,
        )
        mean = float(primary.mean())
        variants[variant] = {
            "cells": int(len(selected)),
            "fixed_schema_macro_f1_mean": mean,
            "fixed_schema_macro_f1_std": float(primary.std(ddof=1)),
            "delta_from_archived_baseline": mean - BASELINE_MEAN,
            "delta_from_current_best_prior_calibration": mean
            - CURRENT_BEST_MEAN,
            "wins_over_archived_baseline": int(
                sum(
                    float(row["metrics"]["fixed_schema_macro_f1"])
                    > baseline_by_cell[
                        (int(row["fold_id"]), int(row["repeat_id"]))
                    ]
                    for row in selected
                )
            ),
            "per_class_f1_mean": per_class.mean(axis=0).tolist(),
        }
    candidates = [
        variant
        for variant in VARIANTS
        if variant not in {"baseline_archived", "baseline_reproduced"}
    ]
    best = max(
        candidates,
        key=lambda name: variants[name]["fixed_schema_macro_f1_mean"],
    )
    promotion_passed = (
        variants[best]["delta_from_archived_baseline"]
        >= MINIMUM_PROMOTION_DELTA
    )
    return {
        "variants": variants,
        "decision": {
            "best_low_cost_candidate": best,
            "best_delta_from_archived_baseline": variants[best][
                "delta_from_archived_baseline"
            ],
            "minimum_promotion_delta": MINIMUM_PROMOTION_DELTA,
            "promotion_passed": promotion_passed,
            "default_enabled": False,
            "state": (
                "DEVELOPMENT_CANDIDATE_KEEP_FOR_CONFIRMATION"
                if promotion_passed
                else "NO_LOW_COST_SUGGESTION_PROMOTED"
            ),
        },
    }


def _make_evidence(
    *,
    summary: Mapping[str, Any],
    consultation: Mapping[str, Any],
) -> str:
    variants = summary["variants"]
    best_variant = summary["decision"]["best_low_cost_candidate"]
    best_item = variants[best_variant]
    prompt = str(consultation["user_prompt"])
    response = "\n".join(
        line.rstrip() for line in str(consultation["content"]).splitlines()
    )
    if summary["decision"]["promotion_passed"]:
        decision_note = (
            f"This exceeds the existing `{MINIMUM_PROMOTION_DELTA:.3f}` "
            "development materiality line and the existing prior-calibrated "
            f"P11 result `{CURRENT_BEST_MEAN:.6f}`. It is retained only as a "
            "development candidate; default enablement and holdout claims "
            "remain forbidden."
        )
    else:
        decision_note = (
            f"This is below the existing `{MINIMUM_PROMOTION_DELTA:.3f}` "
            "development materiality line. No new suggestion is retained."
        )
    lines = [
        "# Lithofacies agent analysis chapter evidence",
        "",
        "## Outcome first",
        "",
        (
            f"The best newly tested low-cost suggestion was `{best_variant}`. "
            "Mean fixed-nine Macro-F1 changed from "
            f"`{BASELINE_MEAN:.6f}` to "
            f"`{best_item['fixed_schema_macro_f1_mean']:.6f}` "
            f"(`{best_item['delta_from_archived_baseline']:+.6f}`)."
        ),
        "",
        decision_note,
        "",
        (
            "These are adaptive exploratory development results, not an "
            "unbiased holdout estimate. The estimator settings are "
            "seed-independent here (`subsample=1`, `colsample_bytree=1`), so "
            "12 cells contain four distinct held-out well-family outcomes, "
            "each repeated across three nominal seeds."
        ),
        "",
        "## Real LOGO4 × 3-seed comparisons",
        "",
        "| variant | mean fixed-9 Macro-F1 | std | delta vs baseline | wins/12 | verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for variant in VARIANTS:
        item = variants[variant]
        if variant == "baseline_archived":
            verdict = "reference"
        elif variant == "baseline_reproduced":
            verdict = "alignment check"
        elif item["delta_from_archived_baseline"] >= MINIMUM_PROMOTION_DELTA:
            verdict = "development candidate"
        else:
            verdict = "reject / diagnostic only"
        lines.append(
            f"| {variant} | {item['fixed_schema_macro_f1_mean']:.6f} | "
            f"{item['fixed_schema_macro_f1_std']:.6f} | "
            f"{item['delta_from_archived_baseline']:+.6f} | "
            f"{item['wins_over_archived_baseline']}/12 | {verdict} |"
        )
    lines.extend(
        [
            "",
            "The reproduced baseline has identical validation argmax decisions "
            "to the archived baseline; its maximum absolute logit difference "
            f"is `{summary['evaluation_alignment']['max_abs_logit_error']:.9g}`.",
            "",
            "## Per-class diagnostic requested by DeepSeek S4",
            "",
            "| class id | development support | baseline F1 | best candidate F1 |",
            "|---:|---:|---:|---:|",
        ]
    )
    baseline_f1 = variants["baseline_archived"]["per_class_f1_mean"]
    best_f1 = variants[best_variant]["per_class_f1_mean"]
    for class_id, support in enumerate(summary["data"]["class_counts"]):
        lines.append(
            f"| {class_id} | {support} | {baseline_f1[class_id]:.6f} | "
            f"{best_f1[class_id]:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Suggestion disposition",
            "",
            "- **Verified:** S1/E1 weight exponents `0.75` and `1.0` under the "
            "otherwise unchanged baseline.",
            "- **Verified:** S2/E2 used the prompt's explicit `well + mask` "
            "interpretation (26 × 33 = 858 features) and removed seismic.",
            "- **Verified:** S3/E3 used `max_depth=3`, `eta=0.1`, and 60 rounds.",
            "- **Adaptive verified follow-up:** the already documented P11 "
            "fold-train prior correction was applied to the best S3/E3 logits "
            "after the uncalibrated result was observed.",
            "- **Verified diagnostic:** S4 per-fold/per-class F1 is retained in "
            "the result rows and summarized above.",
            "- **未验证:** S5 SMOTE/ADASYN and S6 three-point input smoothing "
            "were not tested in this bounded chapter.",
            "- **未验证:** larger MOMENT, additional seismic attributes, focal "
            "loss, class-balanced loss, and any frozen-holdout effect remain "
            "untested.",
            "",
            "## Leakage and attribution boundary",
            "",
            "- Only the immutable development LOGO4 batch was opened. Every "
            "reported row records `known_holdout_accessed=false` and "
            "`frozen_test_accessed=false`.",
            "- Feature transforms and class weights use fold-train arrays only. "
            "Validation labels are used only for evaluation.",
            "- The DeepSeek consultation is common-sense advice, not empirical "
            "evidence. Only the explicitly listed experiments above were run.",
            "- No result in this chapter measures a MOMENT contribution. "
            "大模型贡献占比待下一轮消融确认。",
            "",
            "## DeepSeek call metadata",
            "",
            f"- Requested model: `{consultation['request_model']}`.",
            f"- Provider response model: `{consultation['response_model']}`.",
            f"- Response id: `{consultation['response_id']}`.",
            f"- Prompt SHA-256: `{consultation['prompt_sha256']}`.",
            "- The API credential was process-local and was not persisted.",
            "",
            "## Structured user prompt sent to DeepSeek",
            "",
            "```text",
            prompt.rstrip(),
            "```",
            "",
            "## DeepSeek original response",
            "",
            response.rstrip(),
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    development_batch: Path,
    baseline_bundle: Path,
    p11_summary: Path,
    consultation_file: Path,
    output_dir: Path,
) -> dict[str, Any]:
    ensure_development_only_paths(
        (
            development_batch,
            baseline_bundle,
            p11_summary,
            consultation_file,
            output_dir,
        )
    )
    preserved = verify_preserved_outputs()
    arrays, manifest, p11_payload, class_counts = _load_context(
        development_batch, p11_summary
    )
    consultation = json.loads(consultation_file.read_text(encoding="utf-8"))
    if (
        consultation.get("schema_version") != CONSULTATION_SCHEMA
        or consultation.get("request_model") != DEEPSEEK_REQUEST_MODEL
        or consultation.get("credential_persisted") is not False
        or not str(consultation.get("content", "")).strip()
    ):
        raise RuntimeError("DeepSeek consultation record is invalid")
    with np.load(baseline_bundle, allow_pickle=False) as loaded:
        baseline_arrays = {
            key: loaded[key].copy() for key in loaded.files if key != "manifest"
        }
        baseline_manifest = json.loads(str(loaded["manifest"].item()))
    if (
        baseline_manifest.get("split_hash") != EXPECTED_SPLIT_HASH
        or baseline_manifest.get("development_batch_sha256")
        != _sha256(development_batch)
        or baseline_manifest.get("known_holdout_accessed") is not False
        or baseline_manifest.get("frozen_test_accessed") is not False
    ):
        raise RuntimeError("archived baseline bundle violates the contract")
    rows: list[dict[str, Any]] = []
    max_logit_error = 0.0
    argmax_mismatches = 0
    for fold_id in FOLD_IDS:
        fold = _fold_arrays(arrays, fold_id)
        train_well = np.asarray(fold["p_train_well"], dtype=np.float32)
        train_seismic = np.asarray(fold["p_train_seismic"], dtype=np.float32)
        train_labels = np.asarray(fold["p_train_labels"], dtype=np.int64)
        validation_well = np.asarray(
            fold["p_validation_well"], dtype=np.float32
        )
        validation_seismic = np.asarray(
            fold["p_validation_seismic"], dtype=np.float32
        )
        validation_labels = np.asarray(
            fold["p_validation_labels"], dtype=np.int64
        )
        counts = np.asarray(fold["class_counts"], dtype=np.int64)
        flat_train = multimodal_numpy_features(train_well, train_seismic)
        flat_validation = multimodal_numpy_features(
            validation_well, validation_seismic
        )
        well_only_train = well_and_mask_only_features(train_well, train_seismic)
        well_only_validation = well_and_mask_only_features(
            validation_well, validation_seismic
        )
        for repeat_id, seed in enumerate(REPEAT_SEEDS):
            prefix = f"f{fold_id}_r{repeat_id}"
            archived = np.asarray(
                baseline_arrays[f"{prefix}_validation_logits"],
                dtype=np.float32,
            )
            depth3_logits = _train_xgboost(
                train_features=flat_train,
                train_labels=train_labels,
                validation_features=flat_validation,
                class_counts=counts,
                seed=int(seed),
                weight_alpha=0.5,
                max_depth=3,
                eta=0.1,
                rounds=60,
            )
            variants = {
                "baseline_archived": (
                    archived,
                    {
                        "source": "verified_archived_stage3_logits",
                        "feature_count": 1155,
                        "weight_alpha": 0.5,
                        "train_rows": int(len(train_labels)),
                    },
                ),
                "baseline_reproduced": (
                    _train_xgboost(
                        train_features=flat_train,
                        train_labels=train_labels,
                        validation_features=flat_validation,
                        class_counts=counts,
                        seed=int(seed),
                        weight_alpha=0.5,
                    ),
                    {
                        "source": "retrained_alignment_control",
                        "feature_count": 1155,
                        "weight_alpha": 0.5,
                        "train_rows": int(len(train_labels)),
                    },
                ),
                "weight_alpha_075": (
                    _train_xgboost(
                        train_features=flat_train,
                        train_labels=train_labels,
                        validation_features=flat_validation,
                        class_counts=counts,
                        seed=int(seed),
                        weight_alpha=0.75,
                    ),
                    {
                        "source": "DeepSeek-S1/E1",
                        "feature_count": 1155,
                        "weight_alpha": 0.75,
                        "train_rows": int(len(train_labels)),
                    },
                ),
                "weight_alpha_100": (
                    _train_xgboost(
                        train_features=flat_train,
                        train_labels=train_labels,
                        validation_features=flat_validation,
                        class_counts=counts,
                        seed=int(seed),
                        weight_alpha=1.0,
                    ),
                    {
                        "source": "DeepSeek-S1/E1",
                        "feature_count": 1155,
                        "weight_alpha": 1.0,
                        "train_rows": int(len(train_labels)),
                    },
                ),
                "well_and_mask_only_858": (
                    _train_xgboost(
                        train_features=well_only_train,
                        train_labels=train_labels,
                        validation_features=well_only_validation,
                        class_counts=counts,
                        seed=int(seed),
                        weight_alpha=0.5,
                    ),
                    {
                        "source": "DeepSeek-S2/E2",
                        "feature_count": 858,
                        "weight_alpha": 0.5,
                        "train_rows": int(len(train_labels)),
                        "seismic_features_used": 0,
                    },
                ),
                "depth3_eta01_rounds60": (
                    depth3_logits,
                    {
                        "source": "DeepSeek-S3/E3",
                        "feature_count": 1155,
                        "weight_alpha": 0.5,
                        "train_rows": int(len(train_labels)),
                        "max_depth": 3,
                        "eta": 0.1,
                        "rounds": 60,
                    },
                ),
                "depth3_eta01_rounds60_prior025": (
                    prior_calibrate(depth3_logits, counts, shrinkage=0.25),
                    {
                        "source": "adaptive-P11-prior-followup",
                        "feature_count": 1155,
                        "weight_alpha": 0.5,
                        "train_rows": int(len(train_labels)),
                        "max_depth": 3,
                        "eta": 0.1,
                        "rounds": 60,
                        "prior_shrinkage": 0.25,
                        "prior_fit_scope": "fold_train_class_counts_only",
                    },
                ),
            }
            reproduced = variants["baseline_reproduced"][0]
            max_logit_error = max(
                max_logit_error,
                float(np.max(np.abs(archived - reproduced))),
            )
            argmax_mismatches += int(
                np.count_nonzero(
                    archived.argmax(axis=1) != reproduced.argmax(axis=1)
                )
            )
            for variant in VARIANTS:
                logits, training = variants[variant]
                rows.append(
                    _row(
                        fold_id=fold_id,
                        repeat_id=repeat_id,
                        variant=variant,
                        labels=validation_labels,
                        logits=logits,
                        training=training,
                    )
                )
    aggregation = summarize_rows(rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "data": {
            "protocol": "strict_LOGO4_three_seed_development_only",
            "sample_count": int(manifest["development_sample_count"]),
            "class_counts": class_counts,
            "fold_ids": list(FOLD_IDS),
            "seeds": [int(value) for value in REPEAT_SEEDS],
            "effective_distinct_validation_groups": len(FOLD_IDS),
            "estimator_seed_effective": False,
            "estimator_seed_reason": (
                "subsample=1.0 and colsample_bytree=1.0 make these CPU fits "
                "deterministic across the three nominal seeds"
            ),
            "split_hash": EXPECTED_SPLIT_HASH,
            "development_batch_sha256": _sha256(development_batch),
            "frozen_test_accessed": False,
            "known_holdout_accessed": False,
        },
        "current_p11": {
            "baseline_fixed_schema_macro_f1": BASELINE_MEAN,
            "prior_calibrated_fixed_schema_macro_f1": CURRENT_BEST_MEAN,
            "cross_attention_fixed_schema_macro_f1": p11_payload["variants"][
                "cross_attention"
            ]["metrics"]["fixed_schema_macro_f1"]["mean"],
            "large_model_contribution_share": (
                "pending_next_pretrained_vs_random_encoder_ablation"
            ),
        },
        "consultation": {
            key: consultation.get(key)
            for key in (
                "request_model",
                "response_model",
                "response_id",
                "prompt_sha256",
                "credential_persisted",
            )
        },
        "evaluation_alignment": {
            "archived_baseline_mean": BASELINE_MEAN,
            "reproduced_baseline_mean": aggregation["variants"][
                "baseline_reproduced"
            ]["fixed_schema_macro_f1_mean"],
            "max_abs_logit_error": max_logit_error,
            "argmax_mismatches": argmax_mismatches,
        },
        **aggregation,
        "suggestions": {
            "verified": [
                "S1/E1",
                "S2/E2",
                "S3/E3",
                "S4",
                "adaptive-P11-prior-followup",
            ],
            "unverified": [
                "S5-SMOTE-or-ADASYN",
                "S6-three-point-input-smoothing",
                "larger-MOMENT",
                "focal-or-class-balanced-loss",
                "frozen-holdout-effect",
            ],
            "leakage_safeguard": "no_cross_fold_or_cross_family_resampling",
        },
        "preserved_outputs": preserved,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "results.jsonl", rows)
    _write_json(output_dir / "summary.json", summary)
    evidence = _make_evidence(summary=summary, consultation=consultation)
    (output_dir / "evidence.md").write_text(evidence, encoding="utf-8")
    artifacts = []
    for name in ("results.jsonl", "summary.json", "evidence.md"):
        path = output_dir / name
        artifacts.append(
            {"path": name, "sha256": _sha256(path), "bytes": path.stat().st_size}
        )
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifacts": artifacts,
        "preserved_outputs": preserved,
        "frozen_test_accessed": False,
        "known_holdout_accessed": False,
    }
    _write_json(output_dir / "artifact_manifest.json", manifest_payload)
    return summary


def verify_artifacts(output_dir: Path) -> dict[str, Any]:
    ensure_development_only_paths((output_dir,))
    preserved = verify_preserved_outputs()
    manifest = json.loads(
        (output_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("frozen_test_accessed") is not False
        or manifest.get("known_holdout_accessed") is not False
        or manifest.get("preserved_outputs") != preserved
    ):
        raise RuntimeError("agent chapter manifest violates the contract")
    for artifact in manifest.get("artifacts", ()):
        path = output_dir / str(artifact["path"])
        if _sha256(path) != artifact["sha256"] or path.stat().st_size != int(
            artifact["bytes"]
        ):
            raise RuntimeError(f"agent chapter artifact changed: {path}")
    rows = _read_jsonl(output_dir / "results.jsonl")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    if len(rows) != len(VARIANTS) * len(FOLD_IDS) * len(REPEAT_SEEDS):
        raise RuntimeError("agent chapter row count changed")
    if Counter(row["variant"] for row in rows) != Counter(
        {variant: 12 for variant in VARIANTS}
    ):
        raise RuntimeError("agent chapter variant counts changed")
    if any(
        row.get("known_holdout_accessed") is not False
        or row.get("frozen_test_accessed") is not False
        or row.get("development_only") is not True
        for row in rows
    ):
        raise RuntimeError("agent chapter contains an illegal evaluation row")
    recomputed = summarize_rows(rows)
    if (
        recomputed["variants"] != summary.get("variants")
        or recomputed["decision"] != summary.get("decision")
        or summary.get("preserved_outputs") != preserved
        or summary.get("evaluation_alignment", {}).get("argmax_mismatches") != 0
    ):
        raise RuntimeError("agent chapter summary is not reproducible")
    evidence = (output_dir / "evidence.md").read_text(encoding="utf-8")
    required = (
        "DeepSeek original response",
        "未验证",
        "Leakage and attribution boundary",
        "known_holdout_accessed=false",
        "大模型贡献占比待下一轮消融确认",
    )
    if not all(marker in evidence for marker in required):
        raise RuntimeError("agent chapter evidence is incomplete")
    return {
        "status": "verified",
        "artifacts": len(manifest["artifacts"]),
        "rows": len(rows),
        "preserved_artifacts": len(preserved),
        "decision": summary["decision"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    consult_parser = subparsers.add_parser("consult")
    consult_parser.add_argument("--development-batch", type=Path, required=True)
    consult_parser.add_argument("--p11-summary", type=Path, required=True)
    consult_parser.add_argument("--output", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--development-batch", type=Path, required=True)
    run_parser.add_argument("--baseline-bundle", type=Path, required=True)
    run_parser.add_argument("--p11-summary", type=Path, required=True)
    run_parser.add_argument("--consultation", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "consult":
        payload = consult(
            development_batch=args.development_batch,
            p11_summary=args.p11_summary,
            output=args.output,
            api_key=os.environ.get("DEEPSEEK_KEY", ""),
        )
    elif args.command == "run":
        payload = run(
            development_batch=args.development_batch,
            baseline_bundle=args.baseline_bundle,
            p11_summary=args.p11_summary,
            consultation_file=args.consultation,
            output_dir=args.output_dir,
        )
    else:
        payload = verify_artifacts(args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
