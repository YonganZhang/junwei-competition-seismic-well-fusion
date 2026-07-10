"""统一数据接口层 — 6条赛道共用的样本存取schema。

用途:每条赛道的build_dataset.py用save_split()写出train/test，
每条赛道的建模代码用load_dataset()读，不用管底层Volve/F3/Penobscot具体怎么拼出来的。

样本schema(单个sample字典):
    seismic_patch: np.ndarray            # 地震局部体/切片，float32
    well_log_seq:  np.ndarray | None      # 测井曲线片段，部分任务可能没有(纯地震任务)
    position:      dict                  # {"inline":int,"crossline":int,"time_ms":float,"well_name":str|None}
    label:         np.ndarray | float | int
    meta:          dict                  # {"task":str,"source":str,"split":str, ...}

存储格式:HDF5(.h5)，每个split一个文件，路径由 task_dir() 统一给出，不手写路径字符串。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "_data" / "processed"

VALID_TASKS = ("fault", "facies", "reservoir", "lithofacies", "sweetspot", "reconstruction")
VALID_SPLITS = ("train", "test")


def task_dir(task: str) -> Path:
    if task not in VALID_TASKS:
        raise ValueError(f"未知任务 {task!r}，必须是 {VALID_TASKS} 之一")
    d = PROCESSED_DIR / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def split_path(task: str, split: str) -> Path:
    if split not in VALID_SPLITS:
        raise ValueError(f"split 必须是 {VALID_SPLITS} 之一，收到 {split!r}")
    return task_dir(task) / f"{split}.h5"


def save_split(task: str, split: str, samples: list[dict[str, Any]]) -> Path:
    """把一批样本写成统一schema的HDF5文件。samples为空直接报错，不允许生成假数据充数。"""
    if not samples:
        raise ValueError(f"{task}/{split} 传入空样本列表，拒绝写出空数据集")

    path = split_path(task, split)
    with h5py.File(path, "w") as f:
        f.attrs["task"] = task
        f.attrs["split"] = split
        f.attrs["n_samples"] = len(samples)
        for i, s in enumerate(samples):
            g = f.create_group(f"sample_{i:07d}")
            g.create_dataset("seismic_patch", data=np.asarray(s["seismic_patch"], dtype="float32"))
            if s.get("well_log_seq") is not None:
                g.create_dataset("well_log_seq", data=np.asarray(s["well_log_seq"], dtype="float32"))
            g.create_dataset("label", data=np.asarray(s["label"]))
            g.attrs["position"] = json.dumps(s.get("position", {}))
            g.attrs["meta"] = json.dumps({**s.get("meta", {}), "task": task, "split": split})
    return path


def load_dataset(task: str, split: str = "train") -> Iterator[dict[str, Any]]:
    """按统一schema逐样本读出，供任意模型的DataLoader包一层用。"""
    path = split_path(task, split)
    if not path.exists():
        raise FileNotFoundError(
            f"{task}/{split} 数据集不存在: {path}。先跑对应 _pipelines/02_task_datasets/{task}/build_dataset.py"
        )
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            g = f[key]
            sample = {
                "seismic_patch": g["seismic_patch"][()],
                "well_log_seq": g["well_log_seq"][()] if "well_log_seq" in g else None,
                "label": g["label"][()],
                "position": json.loads(g.attrs["position"]),
                "meta": json.loads(g.attrs["meta"]),
            }
            yield sample


def dataset_stats(task: str, split: str = "train") -> dict[str, Any]:
    """基本统计检查:样本数、label分布，用于跑完build_dataset后立刻核实，不凭感觉猜。"""
    path = split_path(task, split)
    if not path.exists():
        return {"task": task, "split": split, "exists": False}
    with h5py.File(path, "r") as f:
        n = f.attrs.get("n_samples", len(f.keys()))
        labels = [f[k]["label"][()] for k in f.keys()]
    labels_arr = np.asarray(labels)
    stats: dict[str, Any] = {"task": task, "split": split, "exists": True, "n_samples": int(n)}
    if labels_arr.dtype.kind in "iu" and labels_arr.ndim == 1:
        vals, counts = np.unique(labels_arr, return_counts=True)
        stats["label_distribution"] = dict(zip(vals.tolist(), counts.tolist()))
    else:
        stats["label_shape"] = labels_arr.shape
        stats["label_dtype"] = str(labels_arr.dtype)
    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3 or sys.argv[1] not in ("stats",):
        print("用法: python dataset_io.py stats <task>[/<split>]")
        sys.exit(1)
    arg = sys.argv[2]
    task, _, split = arg.partition("/")
    print(json.dumps(dataset_stats(task, split or "train"), ensure_ascii=False, indent=2))
