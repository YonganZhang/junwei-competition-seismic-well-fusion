#!/usr/bin/env python3
"""DeepSeek analysis chapter and one low-cost facies experiment.

The tail step has two explicit phases:

1. ``analyze`` builds a source-backed prompt from the locked development
   manifests and P13 evidence, then calls DeepSeek through an environment-only
   credential.
2. ``run`` tests the selected low-cost suggestion on the same folds, samples,
   seeds, metric implementation, and P13 training harness.

No command accepts a holdout archive or test.h5 path.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import p11_residual_fusion as p11  # noqa: E402
import p12_repair_v1 as p12  # noqa: E402
import p13_cross_attention as p13  # noqa: E402


OUTPUT_ROOT = HERE / "_outputs" / "agent_chapter"
SCHEMA_VERSION = "facies-agent-analysis-chapter/v1"
ANALYSIS_SCHEMA_VERSION = "facies-deepseek-analysis/v1"
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_ENV_NAME = "DEEPSEEK_KEY"
SELECTED_EXPERIMENT = "fusion_scale_initialization_0.5"
P13_FUSION_SCALE_INITIAL = 0.2
REPAIRED_FUSION_SCALE_INITIAL = 0.5
DEEPSEEK_FAILURE_DELTA = 0.005
P13_SUMMARY_PATH = (
    p13.OUTPUT_ROOT / "p13_cross_attention_summary.json"
)
P13_RESULTS_PATH = (
    p13.OUTPUT_ROOT / "p13_cross_attention_results.jsonl"
)

SYSTEM_PROMPT = (
    "你是一名独立的地震解释与小样本语义分割顾问。"
    "只能基于给出的development证据做常识性诊断；区分已观察事实、技术假设与待验证建议；"
    "禁止声称SAM2造成了提升，因为随机权重消融尚未执行。"
    "建议须尽量给出可执行参数，并标注LOW_COST/MEDIUM/HIGH成本。"
)


def _validate_output_root(output_root: Path) -> Path:
    resolved = Path(output_root).resolve()
    try:
        resolved.relative_to(HERE)
    except ValueError as exc:
        raise ValueError(
            f"agent chapter output must stay inside facies: {resolved}"
        ) from exc
    protected = {
        p11.OUTPUT_ROOT.resolve(),
        p12.OUTPUT_ROOT.resolve(),
        p13.OUTPUT_ROOT.resolve(),
    }
    if resolved in protected:
        raise ValueError(
            "agent chapter refuses to overwrite P11/P12/P13 evidence"
        )
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _load_p13_rows() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in P13_RESULTS_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if len(rows) != 12:
        raise ValueError("P13 context must contain its exact 12-cell grid")
    return rows


def _fold_context(
    manifest_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_fold = {
        int(fold["fold_id"]): fold
        for fold in manifest_payload["folds"]
    }
    rows: list[dict[str, Any]] = []
    for fold_id in p13.FOLDS:
        fold = by_fold[fold_id]
        support = [
            int(value)
            for value in fold["support"]["train_per_class_pixels"]
        ]
        total = sum(support)
        positive = [value for value in support if value > 0]
        rows.append(
            {
                "fold_id": fold_id,
                "full_train_samples": len(fold["train_sample_ids"]),
                "full_validation_samples": len(
                    fold["validation_sample_ids"]
                ),
                "train_pixel_support": support,
                "train_pixel_percent": [
                    100.0 * value / total for value in support
                ],
                "max_to_min_positive_class_ratio": (
                    max(positive) / min(positive)
                ),
            }
        )
    return rows


def collect_context(
    *,
    f3_manifest: Path,
    penobscot_manifest: Path,
    processed_root: Path,
) -> dict[str, Any]:
    manifests = p11.validate_development_inputs(
        f3_manifest=f3_manifest,
        penobscot_manifest=penobscot_manifest,
        processed_root=processed_root,
    )
    p13_verified = p13.verify(p13.OUTPUT_ROOT)
    p13_summary = _load_json(P13_SUMMARY_PATH)
    tasks: dict[str, Any] = {}
    for task_id, display_name in (
        ("facies_f3", "F3"),
        ("facies_penobscot", "Penobscot"),
    ):
        manifest_path = manifests[task_id]
        manifest_payload = _load_json(manifest_path)
        tasks[display_name] = {
            "task_id": task_id,
            "num_classes": (
                10 if task_id == "facies_f3" else 8
            ),
            "development_samples": len(
                manifest_payload["development_sample_ids"]
            ),
            "development_groups": len(
                manifest_payload["development_groups"]
            ),
            "manifest_path": str(manifest_path),
            "manifest_stable_hash": p11.TASK_MANIFEST_HASHES[
                task_id
            ],
            "folds": _fold_context(manifest_payload),
            "p13": p13_summary["tasks"][display_name],
        }
    return {
        "task": "single-channel 128x128 seismic facies segmentation",
        "tasks": tasks,
        "evaluation": p13_summary["evaluation"],
        "experiment": p13_summary["experiment"],
        "overall": p13_summary["overall"],
        "p13_verified": p13_verified,
        "sources": {
            "p13_summary_path": str(
                P13_SUMMARY_PATH.relative_to(PROJECT_ROOT)
            ),
            "p13_summary_sha256": p11._sha256(P13_SUMMARY_PATH),
            "p13_results_path": str(
                P13_RESULTS_PATH.relative_to(PROJECT_ROOT)
            ),
            "p13_results_sha256": p11._sha256(P13_RESULTS_PATH),
        },
        "data_boundary": {
            "folds": list(p13.FOLDS),
            "train_samples_per_fold": 32,
            "validation_samples_per_fold": 16,
            "frozen_test_accessed": False,
            "holdout_paths_accepted": False,
        },
    }


def _percent_pairs(task: Mapping[str, Any]) -> str:
    first, second = task["folds"]
    pairs = [
        f"{left:.2f}/{right:.2f}"
        for left, right in zip(
            first["train_pixel_percent"],
            second["train_pixel_percent"],
        )
    ]
    return "[" + ", ".join(pairs) + "]%"


def build_user_prompt(context: Mapping[str, Any]) -> str:
    f3 = context["tasks"]["F3"]
    pen = context["tasks"]["Penobscot"]
    f3_p13 = f3["p13"]
    pen_p13 = pen["p13"]
    f3_cross = f3_p13["variant_means"]["cross_attention_fusion"]
    pen_cross = pen_p13["variant_means"]["cross_attention_fusion"]
    f3_ratios = "/".join(
        f"{fold['max_to_min_positive_class_ratio']:.2f}"
        for fold in f3["folds"]
    )
    pen_ratios = "/".join(
        f"{fold['max_to_min_positive_class_ratio']:.2f}"
        for fold in pen["folds"]
    )
    return f"""# 任务
②地震相分类赛道：输入单通道128×128地震振幅patch，逐像素预测facies。F3为10类，Penobscot为8类。主指标是固定实现的mIoU，另看macro-F1/accuracy/NLL；类别权重只由各fold训练集拟合。

# 数据与评测边界
- F3 development：{f3['development_samples']}个patch、{f3['development_groups']}个inline组；Penobscot development：{pen['development_samples']}个patch、{pen['development_groups']}个inline组。
- 锁定空间隔离5折中的fold 0和4；实际固定开发预算每折仅32个训练patch、16个验证patch，batch=2。不得使用frozen holdout或test.h5。
- F3 full fold-train像素占比在fold 0/4中约为：{_percent_pairs(f3)}，最大类/最小正类约{f3_ratios}倍。
- Penobscot full fold-train像素占比约：{_percent_pairs(pen)}，最大类/最小类约{pen_ratios}倍。

# 当前模型（P13）
- strong_small_baseline：40 updates的小CNN。
- continued_cnn_control：从同一baseline继续160 updates，只训练CNN decoder/head；AdamW，CNN lr=5e-5，weight_decay=1e-4，cosine，CE+0.25 Dice，水平翻转p=0.5、强度缩放和Gaussian noise std=0.03。
- cross_attention_fusion：CNN最深层特征作query，SAM2 native-128特征作key/value，4 heads、dim=128，feature residual写回CNN decoder；SAM2最后Hiera blocks 22/23解冻。lr：CNN 5e-5、fusion 2e-4、SAM2 1e-5；其余训练口径同continued control。当前用pretrained权重，但贡献占比待random-weight消融。

# 固定development结果（fold 0/4等权均值）
- F3：baseline mIoU {f3_p13['variant_means']['strong_small_baseline']['miou']:.6f}，continued CNN {f3_p13['variant_means']['continued_cnn_control']['miou']:.6f}，cross-attention {f3_cross['miou']:.6f}；cross比continued {f3_p13['comparison']['cross_attention_minus_continued_cnn']:+.6f}。cross macro-F1 {f3_cross['macro_f1']:.6f}。
- Penobscot：baseline {pen_p13['variant_means']['strong_small_baseline']['miou']:.6f}，continued CNN {pen_p13['variant_means']['continued_cnn_control']['miou']:.6f}，cross-attention {pen_cross['miou']:.6f}；cross比continued {pen_p13['comparison']['cross_attention_minus_continued_cnn']:+.6f}。cross macro-F1 {pen_cross['macro_f1']:.6f}。
- 两任务等权：baseline {context['overall']['baseline_mean_miou']:.6f} → cross {context['overall']['cross_attention_mean_miou']:.6f}（整体{context['overall']['cross_attention_minus_baseline']:+.6f}），但不能归因于SAM2。

# 已知限制/诊断
- fusion_scale=sigmoid(logit)，初始化0.2；训练后F3均值{f3_cross['fusion_scale']:.6f}、Penobscot {pen_cross['fusion_scale']:.6f}，几乎钳在初始化附近，门控学习可能不充分。
- 归一化attention entropy偏高：F3 {f3_cross['attention_entropy']:.6f}、Penobscot {pen_cross['attention_entropy']:.6f}，尤其Penobscot注意力接近均匀。
- 训练样本极少、类别极不均衡；P13只跑两个固定fold，尚无随机SAM2消融，不能做来源归因。

# 请分析并回答
1. 先列3–6条最可能的技术瓶颈，逐条说明依据，不要把相关性写成因果。
2. 给出按优先级排序的参数调整、特征设计、训练策略建议；每条标注[LOW_COST]/[MEDIUM]/[HIGH]、精确改动值或范围、预期观察信号与失败判据。
3. 至少给出一个可在上述相同development口径下立刻验证、只改学习率/正则/损失权重之一的低成本实验，最好针对fusion_scale不动或高attention entropy；一次只改一个主要因素。
4. 明确列出“未验证建议”，并提醒大模型贡献需后续pretrained/random对照确认。
5. 不得给出虚构提升数字；可以给假设方向，但不要承诺提升。"""


def _call_deepseek(
    *,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    if not api_key.strip():
        raise ValueError("DeepSeek credential is empty")
    request_payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        DEEPSEEK_ENDPOINT,
        data=json.dumps(
            request_payload,
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            response_payload = json.loads(
                response.read().decode("utf-8")
            )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"DeepSeek HTTP {exc.code}: {detail[:500]}"
        ) from exc
    choices = response_payload.get("choices", [])
    if not choices:
        raise RuntimeError("DeepSeek response omitted choices")
    raw_text = choices[0].get("message", {}).get("content", "")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise RuntimeError("DeepSeek response omitted assistant content")
    return {
        "model_requested": DEEPSEEK_MODEL,
        "model_returned": response_payload.get("model", ""),
        "usage": response_payload.get("usage", {}),
        "raw_text": raw_text,
    }


def analyze(
    *,
    f3_manifest: Path,
    penobscot_manifest: Path,
    processed_root: Path,
    output_root: Path = OUTPUT_ROOT,
    timeout_seconds: float = 90.0,
) -> Path:
    output_root = _validate_output_root(output_root)
    context = collect_context(
        f3_manifest=f3_manifest,
        penobscot_manifest=penobscot_manifest,
        processed_root=processed_root,
    )
    api_key = os.environ.get(DEEPSEEK_ENV_NAME, "")
    response = _call_deepseek(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(context),
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    payload = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "provider": "DeepSeek",
        "endpoint": DEEPSEEK_ENDPOINT,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": build_user_prompt(context),
        "context": context,
        "response": response,
        "credential_transport": "ephemeral_process_environment",
        "api_key_persisted": False,
    }
    return p11._write_json(
        output_root / "deepseek_analysis.json",
        payload,
    )


def _set_fusion_scale_initialization(
    fusion: p13.CrossAttentionFusion,
    probability: float,
) -> None:
    if not 0.0 < probability < 1.0:
        raise ValueError("fusion probability must be inside (0, 1)")
    logit = math.log(probability / (1.0 - probability))
    with torch.no_grad():
        fusion.fusion_scale_logit.fill_(logit)


def _request_deterministic_execution() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _p13_cross_rows() -> dict[tuple[str, int], dict[str, Any]]:
    rows = {
        (str(row["task_id"]), int(row["fold_id"])): row
        for row in _load_p13_rows()
        if row["variant"] == "cross_attention_fusion"
    }
    expected = {
        (task_id, fold_id)
        for task_id in ("facies_f3", "facies_penobscot")
        for fold_id in p13.FOLDS
    }
    if set(rows) != expected:
        raise ValueError("P13 cross-attention context is incomplete")
    return rows


def _run_task(
    task_id: str,
    *,
    manifest_path: Path,
    processed_root: Path,
    device: str,
    run_command: str,
    output_root: Path,
    before_rows: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    budget = p11.stage3.Stage3Budget()
    for fold_id in p13.FOLDS:
        started = time.perf_counter()
        prepared = p11.stage3.prepare_fold(
            task_id=task_id,
            fold_id=fold_id,
            manifest_path=manifest_path,
            processed_root=processed_root,
            budget=budget,
        )
        if (
            prepared.manifest_stable_hash
            != p11.TASK_MANIFEST_HASHES[task_id]
        ):
            raise RuntimeError(f"{task_id} manifest identity drifted")
        seed = p13.ROOT_SEED + fold_id
        baseline_model = p11._build_seeded_small_model(
            task_id,
            prepared.num_classes,
            device,
            seed=seed,
        )
        baseline_result = p11._train_ce_model(
            baseline_model,
            prepared,
            device=device,
            seed=seed,
            label=f"{task_id}/agent-baseline",
        )
        trained_baseline = baseline_result.pop("model")
        trained_baseline.cpu()
        before = before_rows[(task_id, fold_id)]
        reproduced_baseline = float(
            baseline_result["metrics"]["miou"]
        )
        baseline_reproduction_delta = (
            reproduced_baseline - float(before["baseline_metric"])
        )
        def train_gate_variant(
            gate_initialization: float,
        ) -> dict[str, Any]:
            p11._seed_all(seed)
            small_model = copy.deepcopy(trained_baseline)
            sam2_encoder = p13.build_sam2_encoder(
                task_id,
                prepared.num_classes,
                device,
                weight_mode="pretrained",
                seed=seed,
            )
            model = p13.CrossAttentionSegmentationModel(
                small_model,
                sam2_encoder,
            )
            _set_fusion_scale_initialization(
                model.fusion,
                gate_initialization,
            )
            result = p13._train_cross_attention(
                model,
                prepared,
                device=device,
                seed=seed,
            )
            del model, small_model, sam2_encoder
            gc.collect()
            torch.cuda.empty_cache()
            return result

        control_result = train_gate_variant(
            P13_FUSION_SCALE_INITIAL
        )
        result = train_gate_variant(
            REPAIRED_FUSION_SCALE_INITIAL
        )
        control_metrics = control_result["metrics"]
        metrics = result["metrics"]
        row = {
            "schema_version": SCHEMA_VERSION,
            "track": "facies",
            "task_id": task_id,
            "dataset": (
                "F3" if task_id == "facies_f3" else "Penobscot"
            ),
            "fold_id": fold_id,
            "seed": seed,
            "variant": SELECTED_EXPERIMENT,
            "before_variant": "fresh_gate_0.2_control",
            "p13_reference_miou": float(before["metric_value"]),
            "fresh_baseline_miou": reproduced_baseline,
            "baseline_reproduction_delta_vs_p13": (
                baseline_reproduction_delta
            ),
            "before_miou": float(control_metrics["miou"]),
            "after_miou": float(metrics["miou"]),
            "delta_miou": float(
                metrics["miou"] - control_metrics["miou"]
            ),
            "p13_reference_delta_from_fresh_control": float(
                control_metrics["miou"] - before["metric_value"]
            ),
            "before_macro_f1": float(control_metrics["macro_f1"]),
            "after_macro_f1": float(metrics["macro_f1"]),
            "before_accuracy": float(control_metrics["accuracy"]),
            "after_accuracy": float(metrics["accuracy"]),
            "before_nll": float(control_metrics["nll"]),
            "after_nll": float(metrics["nll"]),
            "before_attention_entropy": float(
                control_result["attention_entropy"]
            ),
            "after_attention_entropy": float(
                result["attention_entropy"]
            ),
            "before_fusion_scale_initial": (
                P13_FUSION_SCALE_INITIAL
            ),
            "before_fusion_scale_final": float(
                control_result["fusion_scale"]
            ),
            "after_fusion_scale_initial": (
                REPAIRED_FUSION_SCALE_INITIAL
            ),
            "after_fusion_scale_final": float(
                result["fusion_scale"]
            ),
            "existing_grad_clip_max_norm": p13.GRAD_CLIP,
            "candidate_updates": p13.CANDIDATE_UPDATES,
            "train_samples": len(prepared.train_images),
            "validation_samples": len(prepared.validation_images),
            "manifest_stable_hash": (
                prepared.manifest_stable_hash
            ),
            "fold_split_hash": prepared.fold_split_hash,
            "sam2_weight_mode": "pretrained",
            "sam2_trainable_blocks": [22, 23],
            "before_sam2_update_l2": float(
                control_result["sam2_update_l2"]
            ),
            "sam2_update_l2": float(result["sam2_update_l2"]),
            "before_last_grad_norm": float(
                control_result["last_grad_norm"]
            ),
            "last_grad_norm": float(result["last_grad_norm"]),
            "train_loss_last": float(result["train_loss_last"]),
            "train_loss_mean": float(result["train_loss_mean"]),
            "duration_seconds": time.perf_counter() - started,
            "runner_sha256": p11._sha256(Path(__file__)),
            "git_head_at_run": p11._git_head(),
            "p13_results_sha256": p11._sha256(P13_RESULTS_PATH),
            "command": run_command,
            "deterministic_algorithms_requested": True,
            "evidence_path": str(
                (output_root / "evidence.md").relative_to(
                    PROJECT_ROOT
                )
            ),
            "frozen_test_accessed": False,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "task_id": task_id,
                    "fold_id": fold_id,
                    "before_miou": row["before_miou"],
                    "after_miou": row["after_miou"],
                    "delta_miou": row["delta_miou"],
                    "fusion_scale_final": row[
                        "after_fusion_scale_final"
                    ],
                    "p13_reference_miou": row[
                        "p13_reference_miou"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del trained_baseline, baseline_model
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def _task_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    before = float(np.mean([row["before_miou"] for row in rows]))
    after = float(np.mean([row["after_miou"] for row in rows]))
    return {
        "before_miou": before,
        "after_miou": after,
        "delta_miou": after - before,
        "before_macro_f1": float(
            np.mean([row["before_macro_f1"] for row in rows])
        ),
        "after_macro_f1": float(
            np.mean([row["after_macro_f1"] for row in rows])
        ),
        "before_attention_entropy": float(
            np.mean(
                [row["before_attention_entropy"] for row in rows]
            )
        ),
        "after_attention_entropy": float(
            np.mean(
                [row["after_attention_entropy"] for row in rows]
            )
        ),
        "before_fusion_scale_final": float(
            np.mean(
                [row["before_fusion_scale_final"] for row in rows]
            )
        ),
        "after_fusion_scale_final": float(
            np.mean(
                [row["after_fusion_scale_final"] for row in rows]
            )
        ),
    }


def _verdict(
    overall_delta: float,
    task_summaries: Mapping[str, Mapping[str, float]],
) -> str:
    task_deltas = [
        float(values["delta_miou"])
        for values in task_summaries.values()
    ]
    if min(task_deltas) < 0.0 < max(task_deltas):
        return "MIXED_TASK_RESULT"
    if overall_delta > DEEPSEEK_FAILURE_DELTA:
        return "KEEP_FOR_FURTHER_VALIDATION"
    if overall_delta < -DEEPSEEK_FAILURE_DELTA:
        return "REJECT"
    return "NO_MATERIAL_GAIN"


def _write_evidence(
    *,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
    output_root: Path,
) -> Path:
    lines = [
        "# ②地震相分类：智能体分析章节",
        "",
        "## 证据上下文",
        "",
        "本章以已提交的 P13 cross-attention 固定 development "
        "结果为模型上下文，同时保留 small-CNN baseline 和 "
        "continued-CNN control，避免把额外训练收益误写成大模型贡献。",
        "",
        "| 数据集 | development patch | inline组 | 类别数 | "
        "fold 0/4 最大类÷最小正类 | 固定实验样本/折 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    context = analysis["context"]
    for task_name in ("F3", "Penobscot"):
        task = context["tasks"][task_name]
        ratios = "/".join(
            f"{fold['max_to_min_positive_class_ratio']:.2f}"
            for fold in task["folds"]
        )
        lines.append(
            f"| {task_name} | {task['development_samples']} | "
            f"{task['development_groups']} | {task['num_classes']} | "
            f"{ratios} | 32 train / 16 validation |"
        )
    lines.extend(
        [
            "",
            "P13 mIoU：F3 `0.136263 baseline → 0.206862 "
            "continued CNN → 0.292153 cross-attention`；Penobscot "
            "`0.129101 → 0.189564 → 0.205270`。P13 的 "
            "fusion scale 从 0.2 初始化后仅到 F3 0.200461、"
            "Penobscot 0.200573，attention entropy 分别为 "
            "0.891912、0.951056。",
            "",
            "## 发给 DeepSeek 的结构化 prompt",
            "",
            f"请求模型：`{analysis['response']['model_requested']}`；"
            f"服务端返回模型：`{analysis['response']['model_returned']}`。",
            "",
            "### System",
            "",
            "~~~text",
            analysis["system_prompt"],
            "~~~",
            "",
            "### User",
            "",
            "~~~text",
            analysis["user_prompt"],
            "~~~",
            "",
            "## DeepSeek 原始分析文本",
            "",
            analysis["response"]["raw_text"],
            "",
            "## 低成本 development 验证",
            "",
            "DeepSeek 提议把 `fusion_scale` 初值由 0.2 改为 0.5。"
            "P13 既有的全参数 `max_norm=1.0` 梯度裁剪保持不变，"
            "因此本次新增的唯一主要变量是门控初值 `0.2 → 0.5`；"
            "学习率、权重衰减、"
            "损失、增强、更新数、SAM2 解冻层、样本、fold 和 seed "
            "均保持不变。考虑到 CUDA 复跑有非严格确定性，每个 cell "
            "都从同一个本次重训 baseline 分叉并重新训练 gate=0.2 "
            "control 与 gate=0.5 repair；P13 同 cell 数字只作历史参照。",
            "",
            "| 数据集 | fold | P13参照 | 新鲜0.2 control | 0.5 repair | Δ | "
            "后 fusion scale | 后 attention entropy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['fold_id']} | "
            f"{row['p13_reference_miou']:.6f} | "
            f"{row['before_miou']:.6f} | "
            f"{row['after_miou']:.6f} | "
            f"{row['delta_miou']:+.6f} | "
            f"{row['after_fusion_scale_final']:.6f} | "
            f"{row['after_attention_entropy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| 数据集 | 前均值 mIoU | 后均值 mIoU | Δ | "
            "前/后 macro-F1 | 前/后 attention entropy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for task_name in ("F3", "Penobscot"):
        task = summary["tasks"][task_name]
        lines.append(
            f"| {task_name} | {task['before_miou']:.6f} | "
            f"{task['after_miou']:.6f} | "
            f"{task['delta_miou']:+.6f} | "
            f"{task['before_macro_f1']:.6f}/"
            f"{task['after_macro_f1']:.6f} | "
            f"{task['before_attention_entropy']:.6f}/"
            f"{task['after_attention_entropy']:.6f} |"
        )
    overall = summary["overall"]
    lines.extend(
        [
            "",
            f"两任务等权 mean mIoU 从 "
            f"`{overall['before_mean_miou']:.6f}` 变为 "
            f"`{overall['after_mean_miou']:.6f}` "
            f"(`{overall['delta_mean_miou']:+.6f}`)。按本 runner "
            f"预注册的 `> {DEEPSEEK_FAILURE_DELTA:.3f}` materiality "
            "阈值并同时检查任务方向一致性，"
            f"本次判定为 `{summary['verdict']}`。该判定只覆盖这两个"
            "固定 development folds，不外推到 frozen holdout。",
            f"F3 为 `{summary['tasks']['F3']['delta_miou']:+.6f}`，"
            f"Penobscot 为 "
            f"`{summary['tasks']['Penobscot']['delta_miou']:+.6f}`，"
            "方向不一致；两任务最终 fusion scale 也仍紧贴新的 0.5 "
            "初值。因此实验只证明固定融合强度变化会改变指标，未证明"
            "门控已经学会自适应。",
            "",
            "复跑 baseline 相对 P13 历史 baseline 的单 cell 漂移已记录"
            "在 `low_cost_results.jsonl`；正式 Δ 始终由同次新鲜 "
            "gate=0.2 control 与 gate=0.5 repair 相减，不混用历史运行。",
            "历史 P13 harness 只固定随机种子、未启用严格确定性 "
            "CUDA 算法，因此不同进程的绝对分数会漂移；本次成对运行"
            "额外请求 deterministic algorithms 并关闭 cuDNN benchmark，"
            "两支设置相同。本章不把历史 P13 与本次运行的差异解释成"
            "超参数效果。",
            "",
            "## 未验证建议",
            "",
            "以下 DeepSeek 建议本轮未验证，原文已完整保留在上文：",
            "",
            "- key/value LayerNorm 与可学习 attention temperature：未验证。",
            "- warmup、缩短 cosine 周期或 Penobscot 单独调 fusion LR：未验证。",
            "- Dice 权重 0.1/0.5、Focal loss：未验证。",
            "- 改 attention dim/heads、融合层位或多尺度特征：未验证。",
            "- 更大规模预训练或更多 development 样本：未验证。",
            "",
            "**大模型贡献占比仍需下一轮 pretrained/random 权重"
            "对照确认。** 本章只报告整体数值变化，不作 SAM2 因果归因。",
            "",
            "## 数据与密钥边界",
            "",
            "- DeepSeek 密钥由调用进程的临时环境注入，未写入任何产物。",
            "- 只读取锁定 split manifest、development `train.h5` "
            "和既有 P13 证据；未读取 frozen holdout/`test.h5`。",
            "- mIoU 仍调用 P11/P13 的同一 probability evaluator，"
            "fold 0/4 和每折 32/16 样本不变。",
            "- P11、P12、P13 产物在运行前后以 manifest SHA-256 "
            "保护，未删除或覆盖。",
        ]
    )
    path = output_root / "evidence.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build(
    *,
    f3_manifest: Path,
    penobscot_manifest: Path,
    processed_root: Path,
    device: str = "cuda:0",
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Path]:
    started = time.perf_counter()
    _request_deterministic_execution()
    p11._validate_cuda_device(device)
    manifests = p11.validate_development_inputs(
        f3_manifest=f3_manifest,
        penobscot_manifest=penobscot_manifest,
        processed_root=processed_root,
    )
    output_root = _validate_output_root(output_root)
    analysis_path = output_root / "deepseek_analysis.json"
    if not analysis_path.is_file():
        raise FileNotFoundError(
            "run analyze first so the tested suggestion has a raw source"
        )
    analysis = _load_json(analysis_path)
    if analysis.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise ValueError("unsupported DeepSeek analysis schema")
    if analysis.get("api_key_persisted") is not False:
        raise ValueError("analysis did not prove ephemeral credential use")
    p11.verify(p11.OUTPUT_ROOT)
    p12.verify(p12.OUTPUT_ROOT)
    p13.verify(p13.OUTPUT_ROOT)
    protected_hashes = {
        "p11_manifest": p11._sha256(
            p11.OUTPUT_ROOT / "artifact_manifest.csv"
        ),
        "p12_manifest": p11._sha256(
            p12.OUTPUT_ROOT / "artifact_manifest.csv"
        ),
        "p13_manifest": p11._sha256(
            p13.OUTPUT_ROOT / "artifact_manifest.csv"
        ),
    }
    dependency_site = p11._prepare_sam2_dependency_path()
    source_root = p11.verify_git_source(
        p11.SAM2_SOURCE_ROOT,
        p11.SAM2_SOURCE_REVISION,
    )
    checkpoint = p11.verify_checkpoint(
        "facies",
        p11.SAM2_CHECKPOINT,
    )
    p11.insert_import_root(source_root, "sam2")
    output_root.mkdir(parents=True, exist_ok=True)
    run_command = " ".join(
        [
            sys.executable,
            str(Path(__file__).relative_to(PROJECT_ROOT)),
            "run",
            "--f3-manifest",
            str(manifests["facies_f3"]),
            "--penobscot-manifest",
            str(manifests["facies_penobscot"]),
            "--processed-root",
            str(Path(processed_root).resolve()),
            "--device",
            device,
        ]
    )
    before_rows = _p13_cross_rows()
    rows: list[dict[str, Any]] = []
    for task_id in ("facies_f3", "facies_penobscot"):
        rows.extend(
            _run_task(
                task_id,
                manifest_path=manifests[task_id],
                processed_root=Path(processed_root).resolve(),
                device=device,
                run_command=run_command,
                output_root=output_root,
                before_rows=before_rows,
            )
        )
    task_summaries = {
        display_name: _task_summary(
            [row for row in rows if row["task_id"] == task_id]
        )
        for task_id, display_name in (
            ("facies_f3", "F3"),
            ("facies_penobscot", "Penobscot"),
        )
    }
    before_mean = float(
        np.mean(
            [task_summaries[name]["before_miou"] for name in task_summaries]
        )
    )
    after_mean = float(
        np.mean(
            [task_summaries[name]["after_miou"] for name in task_summaries]
        )
    )
    current_hashes = {
        "p11_manifest": p11._sha256(
            p11.OUTPUT_ROOT / "artifact_manifest.csv"
        ),
        "p12_manifest": p11._sha256(
            p12.OUTPUT_ROOT / "artifact_manifest.csv"
        ),
        "p13_manifest": p11._sha256(
            p13.OUTPUT_ROOT / "artifact_manifest.csv"
        ),
    }
    if current_hashes != protected_hashes:
        raise RuntimeError("existing P11/P12/P13 evidence changed")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "selected_experiment": {
            "name": SELECTED_EXPERIMENT,
            "source": "DeepSeek raw analysis",
            "single_changed_factor": "fusion_scale_initialization",
            "before": P13_FUSION_SCALE_INITIAL,
            "after": REPAIRED_FUSION_SCALE_INITIAL,
            "existing_grad_clip_max_norm": p13.GRAD_CLIP,
            "unchanged": {
                "cnn_lr": p13.CNN_LR,
                "fusion_lr": p13.FUSION_LR,
                "sam2_lr": p13.SAM2_LR,
                "weight_decay": p13.WEIGHT_DECAY,
                "dice_weight": p13.DICE_WEIGHT,
                "candidate_updates": p13.CANDIDATE_UPDATES,
                "sam2_weight_mode": "pretrained",
                "sam2_trainable_blocks": [22, 23],
            },
        },
        "tasks": task_summaries,
        "overall": {
            "aggregation": "equal_mean_of_F3_and_Penobscot_mIoU",
            "before_mean_miou": before_mean,
            "after_mean_miou": after_mean,
            "delta_mean_miou": after_mean - before_mean,
        },
        "verdict": _verdict(
            after_mean - before_mean,
            task_summaries,
        ),
        "diagnostic": {
            "task_direction_consistent": all(
                values["delta_miou"] > 0.0
                for values in task_summaries.values()
            ),
            "gate_moved_materially_from_new_initialization": any(
                abs(
                    values["after_fusion_scale_final"]
                    - REPAIRED_FUSION_SCALE_INITIAL
                )
                > 0.01
                for values in task_summaries.values()
            ),
        },
        "deepseek": {
            "analysis_sha256": p11._sha256(analysis_path),
            "model_requested": analysis["response"]["model_requested"],
            "model_returned": analysis["response"]["model_returned"],
            "unverified_suggestions_retained_in_evidence": True,
        },
        "evaluation": {
            "folds": list(p13.FOLDS),
            "root_seed": p13.ROOT_SEED,
            "train_samples_per_fold": 32,
            "validation_samples_per_fold": 16,
            "metric_implementation": (
                "_pipelines.02_task_datasets.facies."
                "p4_metrics.evaluate_probabilities"
            ),
            "frozen_test_accessed": False,
            "holdout_paths_accepted": False,
            "deterministic_algorithms_requested": True,
            "deterministic_algorithms_warn_only": True,
            "cudnn_benchmark": False,
            "comparison_design": (
                "same_invocation_fresh_gate_0.2_control_vs_gate_0.5_repair"
            ),
        },
        "protected_evidence": protected_hashes,
        "runtime": {
            "command": run_command,
            "device": device,
            "duration_seconds": time.perf_counter() - started,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "runner_sha256": p11._sha256(Path(__file__)),
            "sam2_source_revision": p11.SAM2_SOURCE_REVISION,
            "sam2_checkpoint_sha256": p11._sha256(checkpoint),
            "sam2_dependency_site": str(dependency_site),
        },
    }
    results_path = p11._write_jsonl(
        output_root / "low_cost_results.jsonl",
        rows,
    )
    summary_path = p11._write_json(
        output_root / "summary.json",
        summary,
    )
    evidence_path = _write_evidence(
        summary=summary,
        rows=rows,
        analysis=analysis,
        output_root=output_root,
    )
    artifact_rows = []
    for kind, path in (
        ("json", analysis_path),
        ("jsonl", results_path),
        ("json", summary_path),
        ("md", evidence_path),
    ):
        artifact_rows.append(
            {
                "kind": kind,
                "name": path.name,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": p11._sha256(path),
            }
        )
    manifest_path = p11._write_csv(
        output_root / "artifact_manifest.csv",
        artifact_rows,
        ["kind", "name", "path", "sha256"],
    )
    return {
        "analysis": analysis_path,
        "results": results_path,
        "summary": summary_path,
        "evidence": evidence_path,
        "artifact_manifest": manifest_path,
    }


def verify(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = _validate_output_root(output_root)
    expected_paths = {
        "deepseek_analysis.json",
        "low_cost_results.jsonl",
        "summary.json",
        "evidence.md",
    }
    manifest_path = output_root / "artifact_manifest.csv"
    with manifest_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        manifest_rows = list(csv.DictReader(handle))
    if {row["name"] for row in manifest_rows} != expected_paths:
        raise ValueError("agent chapter artifact grid is incomplete")
    for row in manifest_rows:
        path = PROJECT_ROOT / row["path"]
        path.resolve().relative_to(HERE)
        if not path.is_file() or p11._sha256(path) != row["sha256"]:
            raise ValueError(f"agent artifact hash mismatch: {path}")
    analysis = _load_json(output_root / "deepseek_analysis.json")
    summary = _load_json(output_root / "summary.json")
    if analysis.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise ValueError("unsupported DeepSeek analysis schema")
    if analysis.get("api_key_persisted") is not False:
        raise ValueError("DeepSeek key persistence is not disproved")
    if any(
        name in analysis
        for name in ("authorization", "api_key", "credential_value")
    ):
        raise ValueError("analysis persisted a credential-bearing field")
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported agent chapter schema")
    if (
        summary["evaluation"]["frozen_test_accessed"]
        or summary["evaluation"]["holdout_paths_accepted"]
        or tuple(summary["evaluation"]["folds"]) != p13.FOLDS
    ):
        raise ValueError("agent chapter violated development-only scope")
    rows = [
        json.loads(line)
        for line in (output_root / "low_cost_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    expected_cells = {
        (task_id, fold_id)
        for task_id in ("facies_f3", "facies_penobscot")
        for fold_id in p13.FOLDS
    }
    if {
        (row["task_id"], int(row["fold_id"])) for row in rows
    } != expected_cells:
        raise ValueError("agent experiment lacks its exact four cells")
    runner_sha = p11._sha256(Path(__file__))
    for row in rows:
        if (
            row["runner_sha256"] != runner_sha
            or row["frozen_test_accessed"]
            or row["variant"] != SELECTED_EXPERIMENT
            or not math.isclose(
                row["after_fusion_scale_initial"],
                REPAIRED_FUSION_SCALE_INITIAL,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("agent experiment row contract drifted")
    current_protected = {
        "p11_manifest": p11._sha256(
            p11.OUTPUT_ROOT / "artifact_manifest.csv"
        ),
        "p12_manifest": p11._sha256(
            p12.OUTPUT_ROOT / "artifact_manifest.csv"
        ),
        "p13_manifest": p11._sha256(
            p13.OUTPUT_ROOT / "artifact_manifest.csv"
        ),
    }
    if summary["protected_evidence"] != current_protected:
        raise ValueError("protected P11/P12/P13 evidence changed")
    evidence = (output_root / "evidence.md").read_text(
        encoding="utf-8"
    )
    if analysis["response"]["raw_text"] not in evidence:
        raise ValueError("DeepSeek raw text was not preserved verbatim")
    for phrase in (
        "未验证建议",
        "大模型贡献占比仍需下一轮",
        "未读取 frozen holdout/`test.h5`",
    ):
        if phrase not in evidence:
            raise ValueError(f"evidence omitted required phrase: {phrase}")
    return {
        "schema_version": SCHEMA_VERSION,
        "rows": len(rows),
        "artifacts": len(manifest_rows),
        "verdict": summary["verdict"],
        "overall_delta_miou": summary["overall"][
            "delta_mean_miou"
        ],
        "frozen_test_accessed": False,
        "api_key_persisted": False,
    }


def _add_common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--f3-manifest", type=Path, required=True)
    parser.add_argument(
        "--penobscot-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analyze_parser = commands.add_parser(
        "analyze",
        help="call DeepSeek with source-backed development context",
    )
    _add_common_inputs(analyze_parser)
    analyze_parser.add_argument("--timeout-seconds", type=float, default=90.0)
    run_parser = commands.add_parser(
        "run",
        help="run the selected low-cost suggestion on development",
    )
    _add_common_inputs(run_parser)
    run_parser.add_argument("--device", default="cuda:0")
    verify_parser = commands.add_parser(
        "verify",
        help="verify the complete agent analysis chapter",
    )
    verify_parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if args.command == "analyze":
        result: Any = analyze(
            f3_manifest=args.f3_manifest,
            penobscot_manifest=args.penobscot_manifest,
            processed_root=args.processed_root,
            output_root=args.output_root,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command == "run":
        result = build(
            f3_manifest=args.f3_manifest,
            penobscot_manifest=args.penobscot_manifest,
            processed_root=args.processed_root,
            device=args.device,
            output_root=args.output_root,
        )
    else:
        result = verify(args.output_root)
    if isinstance(result, Path):
        payload = {"analysis": str(result)}
    elif isinstance(result, Mapping):
        payload = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in result.items()
        }
    else:
        payload = result
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
