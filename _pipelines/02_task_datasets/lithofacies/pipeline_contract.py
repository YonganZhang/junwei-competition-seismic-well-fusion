"""Hard data, leakage, split, and metric contracts for lithofacies track ④."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np


TARGET_SOURCE = "GM09"
TARGET_CURVE_TYPE = "GENETIC FACIES"
PIPELINE_VERSION = "gm09_multimodal_v1"

# Fixed from all 139 explicit GM09 intervals in the eleven Facies.xlsx files.
CLASS_NAMES = (
    "F-MARSH",
    "F-MOUTHBAR",
    "F-OFFSHORE",
    "F-LOWER SHOREFACE",
    "F-UPPER SHOREFACE",
    "F-TIDAL BAR",
    "F-TIDAL CHANNEL",
    "F-TIDAL FLAT MUDDY",
    "F-TIDAL FLAT SANDY",
)
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

# Concept channels are a strict whitelist of observed/basic log measurements.
# No VSH, PHIE, SAND, facies, lithology, RMS, or formation-derived curve is read.
LOG_CHANNELS = (
    "gamma_ray",
    "acoustic_slowness",
    "caliper",
    "bulk_density",
    "neutron_porosity",
    "deep_resistivity",
    "rate_of_penetration",
    "weight_on_bit",
    "rotary_speed",
    "flow_rate",
    "torque",
    "standpipe_pressure",
    "equivalent_circulating_density",
)
LOG_ALIASES = {
    "gamma_ray": (
        "GR", "LFP_GR", "GRMA", "GRMA_ECO_RT", "GRMA_BHC_RT",
        "GRMA_DH_ECO_RT", "ARC_GR_RT", "ARC_GR_UNC_RT", "GR_ARC", "GRM1",
    ),
    "acoustic_slowness": ("AC", "LFP_DT"),
    "caliper": ("CALI", "LFP_CALI", "BCAV"),
    "bulk_density": ("DEN", "LFP_RHOB", "ROBB"),
    "neutron_porosity": ("NEU", "LFP_NPHI", "BPHI"),
    "deep_resistivity": ("RDEP", "LFP_RT", "RT_ARC"),
    "rate_of_penetration": ("ROP5", "ROP5_RM"),
    "weight_on_bit": ("SWOB",),
    "rotary_speed": ("RPM",),
    "flow_rate": ("TFLO",),
    "torque": ("TQA",),
    "standpipe_pressure": ("SPPA",),
    "equivalent_circulating_density": ("ECD", "ECD_ARC", "ECD_ECO_RT", "ECD_MWD"),
}

# The assignment is explicit, deterministic, and applied before any resampling.
FAMILY_PARTITIONS = {
    "15/9-19": "train",
    "15/9-F-12": "train",
    "15/9-F-14": "train",
    "15/9-F-15": "train",
    "15/9-F-4": "guard",
    "15/9-F-5": "test",
}


@dataclass(frozen=True)
class LabelInterval:
    well_id: str
    family_id: str
    top_md_m: float
    base_md_m: float
    class_name: str
    class_id: int
    source_member: str
    source_row: int

    @property
    def partition(self) -> str:
        return FAMILY_PARTITIONS[self.family_id]


def normalize_well_id(value: str) -> str:
    """Canonicalize archive/workbook/pick variants without merging distinct tracks."""
    text = str(value).strip().upper().replace("_", "/", 1)
    text = re.sub(r"^NO\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def mother_family(well_id: str) -> str:
    """Group a mother well and every sidetrack before any sample operation."""
    well = normalize_well_id(well_id)
    if well.startswith("15/9-19"):
        return "15/9-19"
    if well.startswith("15/9-F-15"):
        return "15/9-F-15"
    return well


def partition_for_well(well_id: str) -> str:
    family = mother_family(well_id)
    try:
        return FAMILY_PARTITIONS[family]
    except KeyError as exc:
        raise ValueError(f"井 {well_id!r} 的母井族 {family!r} 没有冻结的partition") from exc


def assert_family_isolation(records: Iterable[dict]) -> dict[str, list[str]]:
    by_partition: dict[str, set[str]] = {"train": set(), "guard": set(), "test": set()}
    for record in records:
        partition = str(record["partition"])
        family = str(record["family_id"])
        if partition not in by_partition:
            raise ValueError(f"未知partition: {partition!r}")
        by_partition[partition].add(family)
        expected = FAMILY_PARTITIONS.get(family)
        if expected != partition:
            raise ValueError(f"母井族 {family} 应属于 {expected}，实际写成 {partition}")
    if not by_partition["train"] or not by_partition["guard"] or not by_partition["test"]:
        raise ValueError(f"train/guard/test必须都有可用母井族: {by_partition}")
    names = tuple(by_partition)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = by_partition[left] & by_partition[right]
            if overlap:
                raise ValueError(f"{left}/{right}母井族泄漏: {sorted(overlap)}")
    return {key: sorted(value) for key, value in by_partition.items()}


def validate_class_name(class_name: str) -> int:
    if class_name in ("UNKNOWN", "UNDEFINED"):
        raise ValueError(f"禁止把 {class_name} 当作GM09标签")
    try:
        return CLASS_TO_ID[class_name]
    except KeyError as exc:
        raise ValueError(
            f"标签 {class_name!r} 不在冻结的9类schema {list(CLASS_NAMES)} 中"
        ) from exc


def validate_label_ids(labels: np.ndarray) -> None:
    array = np.asarray(labels)
    if array.size == 0 or array.dtype.kind not in "iu":
        raise ValueError("标签必须是非空整数数组")
    invalid = set(int(x) for x in np.unique(array)).difference(range(len(CLASS_NAMES)))
    if invalid:
        raise ValueError(f"标签ID越出固定9类schema: {sorted(invalid)}")


def classification_metrics_from_confusion(confusion: np.ndarray) -> dict[str, object]:
    """Fixed-nine-class finite metrics; zero-support classes stay visible as zeros."""
    matrix = np.asarray(confusion, dtype=np.int64)
    n_classes = len(CLASS_NAMES)
    if matrix.shape != (n_classes, n_classes) or np.any(matrix < 0) or matrix.sum() <= 0:
        raise ValueError(f"混淆矩阵必须是非空非负 {n_classes}x{n_classes}，收到 {matrix.shape}")
    tp = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.int64)
    predicted = matrix.sum(axis=0).astype(np.int64)
    fp = predicted - tp
    fn = support - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    union = tp + fp + fn
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    accuracy = float(tp.sum() / matrix.sum())
    supported = support > 0
    # Standard balanced accuracy averages recalls of classes present in y_true.
    # Fixed-schema variants remain explicit so absent test classes are not hidden.
    balanced_accuracy = float(recall[supported].mean())
    macro_f1 = float(f1.mean())
    metrics: dict[str, object] = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "fixed_schema_balanced_accuracy": float(recall.mean()),
        "supported_class_macro_f1": float(f1[supported].mean()),
        "fixed_schema_mean_iou": float(iou.mean()),
        "supported_class_mean_iou": float(iou[supported].mean()),
        "per_class": [
            {
                "class_id": index,
                "class_name": CLASS_NAMES[index],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "iou": float(iou[index]),
                "support": int(support[index]),
                "predicted": int(predicted[index]),
            }
            for index in range(n_classes)
        ],
        "confusion_matrix": matrix.tolist(),
        "evaluated_samples": int(matrix.sum()),
    }
    numeric = [
        accuracy,
        balanced_accuracy,
        macro_f1,
        metrics["fixed_schema_balanced_accuracy"],
        metrics["supported_class_macro_f1"],
        metrics["fixed_schema_mean_iou"],
        metrics["supported_class_mean_iou"],
    ]
    numeric.extend(
        float(row[key])
        for row in metrics["per_class"]
        for key in ("precision", "recall", "f1", "iou")
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("评估指标出现NaN/Inf")
    return metrics
