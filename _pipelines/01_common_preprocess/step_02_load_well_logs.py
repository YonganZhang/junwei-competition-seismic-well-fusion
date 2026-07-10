"""Step 02 — 读取Volve测井曲线(LAS，用lasio)，做缺失值标记/重采样/归一化。

数据来源(真实解压自 Volve_Well_logs.zip，只解出 06.LFP/ 目录，109MB):
    Well_logs/06.LFP/15_9-19 A/159-19A_LFP.las
    Well_logs/06.LFP/15_9-19 BT2/159-19BT2_LFP.las
    Well_logs/06.LFP/15_9-19 SR/159-19SR_LFP.las
这3口是同一井槽(15/9-19)的不同侧钻/分支，是Volve测井里少数导出成LAS格式、
且带完整"LFP(Log Formation Properties)"计算曲线套餐的井。

⚠️ 格式跟赛题描述不完全一致，如实记录:
  - Volve测井原始数据里绝大多数井是 DLIS/LIS 格式(见 04.COMPOSITE 目录)，
    不是LAS；只有06.LFP这3口 + 04.COMPOSITE下2口(15_9-19 SR / 15_9-F-7)导出过LAS。
  - 每口LAS里有171条曲线(远超赛题说的"常规九线")，绝大多数是同一物理量的
    不同处理版本(如 LFP_GR / LFP_GRMAX / LFP_GRMIN，LFP_AI / LFP_AI_B / LFP_AI_G ...)。
  - 实测没有独立的"SW/VSH"传统九线命名，但有对应替代:
        LFP_GR(伽马) LFP_RHOB(密度) LFP_NPHI(中子孔隙度) LFP_DT(纵波时差)
        LFP_DTS(横波时差) LFP_RT(电阻率) LFP_CALI(井径) LFP_VSH(泥质含量)
        LFP_PHIE(有效孔隙度)
    这9条作为本pipeline选定的"核心九线"，实际可用曲线远不止这些(见下方打印的完整
    curve列表)，需要更多曲线时可直接从清洗后的h5里按LFP_*曲线名读取。

清洗规则:
    1. 缺失值标记: LAS头NULL值(-999.25)替换为NaN；同时用1%/99%分位裁剪明显野值
       (如GR偶发跳到>1500 API这种测井仪artefact)。
    2. 深度重采样: 原始采样步长约0.1524m(0.5ft)，重采样到规则0.5m网格
       (线性插值，超出原深度范围的位置标NaN，不外推)。
    3. 归一化:
       - GR用泥岩基线归一化(与readme一致): IGR=(GR-GR_min)/(GR_max-GR_min)，
         GR_min/GR_max直接用LAS里逐深度点算好的LFP_GRMIN/LFP_GRMAX曲线(不是
         全井统一常数，是该LFP处理流程自带的局部基线)。
       - 其余曲线用全井z-score标准化(忽略NaN)。
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import lasio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WELL_LOGS_DIR = PROJECT_ROOT / "_sandbox/volve_data/_extracted_welllogs/Well_logs/06.LFP"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_H5 = OUT_DIR / "well_logs_clean.h5"

CORE_NINE = [
    "LFP_GR", "LFP_RHOB", "LFP_NPHI", "LFP_DT", "LFP_DTS",
    "LFP_RT", "LFP_CALI", "LFP_VSH", "LFP_PHIE",
]
RESAMPLE_STEP_M = 0.5


def _find_las_files() -> list[Path]:
    return sorted(WELL_LOGS_DIR.glob("*/*.las")) + sorted(WELL_LOGS_DIR.glob("*/*.LAS"))


def _clip_outliers(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return x
    lo, hi = np.nanpercentile(finite, [1, 99])
    return np.clip(x, lo, hi)


def load_and_clean_well(las_path: Path) -> dict:
    las = lasio.read(str(las_path))
    null_value = las.well.NULL.value if "NULL" in las.well else -999.25
    depth = np.asarray(las["DEPTH"], dtype=np.float64)

    available_curves = [c.mnemonic for c in las.curves if c.mnemonic != "DEPTH"]
    present_core = [c for c in CORE_NINE if c in available_curves]
    missing_core = [c for c in CORE_NINE if c not in available_curves]

    # 目标重采样深度网格
    d0, d1 = np.nanmin(depth), np.nanmax(depth)
    grid = np.arange(np.ceil(d0 / RESAMPLE_STEP_M) * RESAMPLE_STEP_M, d1, RESAMPLE_STEP_M)

    raw = {}
    cleaned = {}
    for name in present_core:
        v = np.asarray(las[name], dtype=np.float64)
        v = np.where(np.isclose(v, null_value), np.nan, v)
        v = _clip_outliers(v)
        raw[name] = v
        # 重采样(线性插值，仅在有效深度范围内；越界给NaN)
        finite_mask = np.isfinite(v)
        if finite_mask.sum() >= 2:
            v_grid = np.interp(grid, depth[finite_mask], v[finite_mask], left=np.nan, right=np.nan)
        else:
            v_grid = np.full_like(grid, np.nan)
        cleaned[name] = v_grid

    # GR 泥岩基线归一化，用逐点 LFP_GRMIN/LFP_GRMAX（若无则退化为全井z-score）
    norm = {}
    if "LFP_GR" in cleaned and "LFP_GRMIN" in available_curves and "LFP_GRMAX" in available_curves:
        grmin = np.interp(grid, depth, np.asarray(las["LFP_GRMIN"], dtype=np.float64), left=np.nan, right=np.nan)
        grmax = np.interp(grid, depth, np.asarray(las["LFP_GRMAX"], dtype=np.float64), left=np.nan, right=np.nan)
        denom = np.where((grmax - grmin) != 0, grmax - grmin, np.nan)
        norm["LFP_GR"] = (cleaned["LFP_GR"] - grmin) / denom
    for name in present_core:
        if name == "LFP_GR" and "LFP_GR" in norm:
            continue
        v = cleaned[name]
        mu, sigma = np.nanmean(v), np.nanstd(v)
        norm[name] = (v - mu) / sigma if sigma and np.isfinite(sigma) and sigma > 0 else v * 0

    well_name = las.well.WELL.value if "WELL" in las.well else las_path.stem
    # LAS头里的WELL对三个侧钻/分支都是同一个"15/9-19"槽名，
    # 用所在目录名(如"15_9-19 A"/"15_9-19 BT2"/"15_9-19 SR")做唯一track标识，
    # 避免H5分组重名。
    track_id = las_path.parent.name
    return {
        "well_name": well_name,
        "track_id": track_id,
        "las_path": str(las_path),
        "n_available_curves_total": len(available_curves),
        "available_curves_sample": available_curves[:20],
        "core_nine_present": present_core,
        "core_nine_missing": missing_core,
        "depth_grid": grid,
        "cleaned": cleaned,
        "normalized": norm,
        "native_depth_range": [float(d0), float(d1)],
        "native_n_samples": int(depth.size),
    }


def save_all(records: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUT_H5, "w") as f:
        for rec in records:
            g = f.create_group(rec["track_id"].replace("/", "_"))
            g.attrs["well_name"] = rec["well_name"]
            g.attrs["las_path"] = rec["las_path"]
            g.attrs["core_nine_present"] = json.dumps(rec["core_nine_present"])
            g.attrs["core_nine_missing"] = json.dumps(rec["core_nine_missing"])
            g.attrs["native_depth_range"] = rec["native_depth_range"]
            g.create_dataset("depth_grid_m", data=rec["depth_grid"].astype("float32"))
            for name, arr in rec["cleaned"].items():
                g.create_dataset(f"clean/{name}", data=arr.astype("float32"))
            for name, arr in rec["normalized"].items():
                g.create_dataset(f"norm/{name}", data=arr.astype("float32"))
    return OUT_H5


def _demo():
    las_files = _find_las_files()
    print(f"=== step_02 well logs === 找到 {len(las_files)} 个LAS文件")
    records = []
    for p in las_files:
        rec = load_and_clean_well(p)
        records.append(rec)
        print(f"\n井 {rec['well_name']} / track {rec['track_id']} ({p.name})")
        print(f"  原始曲线总数: {rec['n_available_curves_total']} (前20个: {rec['available_curves_sample']})")
        print(f"  核心九线命中: {rec['core_nine_present']}")
        if rec["core_nine_missing"]:
            print(f"  核心九线缺失: {rec['core_nine_missing']}")
        print(f"  原始深度范围: {rec['native_depth_range']} m, 原始采样点数: {rec['native_n_samples']}")
        print(f"  重采样后网格点数: {rec['depth_grid'].shape[0]} (step={RESAMPLE_STEP_M}m)")
        gr = rec["cleaned"].get("LFP_GR")
        if gr is not None:
            finite = gr[np.isfinite(gr)]
            print(f"  LFP_GR 清洗后 shape={gr.shape}, 有效点={finite.size}, "
                  f"range=[{finite.min():.2f},{finite.max():.2f}]" if finite.size else "  LFP_GR 全NaN")

    out_path = save_all(records)
    print(f"\n已写出 {out_path}")


if __name__ == "__main__":
    _demo()
