"""Step 04 — 弱井震标定(weak well tie)。

⚠️ 这是弱标定，不是精确井震标定:
    真正的井震标定需要声波(DT)+密度(RHOB)合成一条反射系数序列，再跟子波褶积生成
    合成地震记录，用互相关去对齐真实地震道，此外还需要VSP/checkshot时深表校正
    速度模型误差。本级步骤仍是早期弱标定，但项目后续获取的 Volve VSP 归档
    实际包含 5 口井的 checkshot；P23 已完成独立井验证，见
    `_wiki-methodology/_wiki/_methods/explorations/011-p23-reconstruction-checkshot-target-tie.md`。
    本文件只保留旧产物的可复现性，不再代表当前最佳时深关系。
    本step做的只是:
      1. 井轨迹的(Easting,Northing) —— 用官方 Well_picks_Volve_v1.dat 里同一口井
         不同深度的地层分界点位置，线性插值/外推出连续深度上的井轨迹平面坐标
         (这本身是真实测井解释数据，不是编的，但只有稀疏的~10几个分层点，
         不是真实连续测斜数据，所以轨迹在两个分层点之间是线性近似)。
      2. 深度->双程时(TWT) —— 同样用 Well_picks 里该井每个分层点自带的TWT，
         对MD做分段线性插值/外推得到一条粗糙的时深曲线(不是速度模型算出来的，
         是直接内插同一口井的官方分层TWT，这是目前能拿到的最接近事实的曲线，
         但点数少、井斜段外插误差会明显放大，这是已知局限)。
      3. 每个深度点的(Easting,Northing) 反解到最近的(inline,crossline)网格点
         (nearest neighbor，网格间距12.5m，等于横向误差上限)。

数据依赖:
    - step_01 的 seismic_index(UTM<->il/xl仿射变换 + SeismicVolume.get_trace)
    - step_02 的 outputs/well_logs_clean.h5(重采样后的LFP_*曲线深度网格)
    - Geophysical_Interpretations/Wells/Well_picks_Volve_v1.dat(官方分层MD/TWT/UTM)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_01_load_seismic import SeismicVolume, load_index  # noqa: E402
from step_02_load_well_logs import OUT_H5 as WELL_LOGS_H5  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WELL_PICKS_PATH = (
    PROJECT_ROOT
    / "_sandbox/volve_data/_extracted_interp/Geophysical_Interpretations/Wells/Well_picks_Volve_v1.dat"
)
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def _track_id_to_pick_name(track_id: str) -> str:
    """'15_9-19 BT2' -> 'NO 15/9-19 BT2'（只替换第一个下划线为斜杠，
    并加上 Well_picks 文件里"Well name"列固定带的 "NO " 前缀）。"""
    return "NO " + track_id.replace("_", "/", 1)


def parse_well_picks(path: Path = WELL_PICKS_PATH) -> dict[str, list[dict]]:
    """按 §用header的dash行自动推导列宽 的方式解析，返回 {well_name: [pick, ...]}。"""
    lines = path.read_text(errors="replace").splitlines(keepends=True)
    spans: list[tuple[int, int]] | None = None
    wells: dict[str, list[dict]] = {}
    for line in lines:
        if re.match(r"^\s*-{5,}", line):
            spans = [(m.start(), m.end()) for m in re.finditer(r"-+", line)]
            continue
        if spans is None:
            continue
        if not line.strip() or line.startswith("Well NO") or line.lstrip().startswith("Well name"):
            continue
        cols = [line[s:e].strip() for s, e in spans]
        if len(cols) < 12:
            continue
        well_name, surface, obs, qlf, md, tvd, tvdss, twt, dip, azi, easting, northing = cols[:12]
        if not well_name or not md:
            continue
        try:
            rec = {
                "surface": surface,
                "md": float(md),
                "twt_ms": float(twt) if twt else None,
                "easting": float(easting) if easting else None,
                "northing": float(northing) if northing else None,
            }
        except ValueError:
            continue
        wells.setdefault(well_name, []).append(rec)
    return wells


def build_weak_tie_curve(picks: list[dict]) -> dict:
    """从该井的分层点里筛出同时有 MD/TWT/Easting/Northing 的点，
    按MD排序，返回可插值/外推的数组。"""
    pts = [p for p in picks if p["twt_ms"] is not None and p["easting"] is not None]
    pts.sort(key=lambda p: p["md"])
    # 同一MD多个分层同深度打点(如Seabed跟NORDLAND GP.Top常常是同一MD)会在插值时
    # 产生零间距、除零出NaN，这里按MD去重，同MD只保留第一个。
    dedup: list[dict] = []
    seen_md = set()
    for p in pts:
        if p["md"] in seen_md:
            continue
        seen_md.add(p["md"])
        dedup.append(p)
    md = np.array([p["md"] for p in dedup])
    twt = np.array([p["twt_ms"] for p in dedup])
    easting = np.array([p["easting"] for p in dedup])
    northing = np.array([p["northing"] for p in dedup])
    return {"md": md, "twt_ms": twt, "easting": easting, "northing": northing, "n_picks": len(dedup)}


def _interp_extrap(x_query: np.ndarray, x_known: np.ndarray, y_known: np.ndarray) -> np.ndarray:
    """分段线性插值；超出已知范围时用端点局部斜率线性外推(弱近似，越远越不可信)。"""
    y = np.interp(x_query, x_known, y_known)
    if len(x_known) >= 2:
        below = x_query < x_known[0]
        if below.any():
            slope = (y_known[1] - y_known[0]) / (x_known[1] - x_known[0])
            y[below] = y_known[0] + slope * (x_query[below] - x_known[0])
        above = x_query > x_known[-1]
        if above.any():
            slope = (y_known[-1] - y_known[-2]) / (x_known[-1] - x_known[-2])
            y[above] = y_known[-1] + slope * (x_query[above] - x_known[-1])
    return y


def tie_well(track_id: str, depth_grid_m: np.ndarray, picks_by_well: dict, vol: SeismicVolume) -> dict:
    pick_name = _track_id_to_pick_name(track_id)
    if pick_name not in picks_by_well:
        raise KeyError(f"Well_picks 里找不到 {pick_name!r}（track_id={track_id}）")
    curve = build_weak_tie_curve(picks_by_well[pick_name])
    if curve["n_picks"] < 2:
        raise RuntimeError(f"{pick_name} 只有 {curve['n_picks']} 个可用分层点，不足以插值")

    twt_est = _interp_extrap(depth_grid_m, curve["md"], curve["twt_ms"])
    easting_est = _interp_extrap(depth_grid_m, curve["md"], curve["easting"])
    northing_est = _interp_extrap(depth_grid_m, curve["md"], curve["northing"])

    idx = vol.index
    samples_ms = idx["samples_ms"]
    inline = np.empty(len(depth_grid_m), dtype=np.int32)
    crossline = np.empty(len(depth_grid_m), dtype=np.int32)
    time_idx = np.empty(len(depth_grid_m), dtype=np.int32)
    for i in range(len(depth_grid_m)):
        il, xl = vol.utm_to_il_xl(easting_est[i], northing_est[i])
        inline[i] = il
        crossline[i] = xl
        time_idx[i] = int(np.clip(np.searchsorted(samples_ms, twt_est[i]), 0, len(samples_ms) - 1))

    return {
        "pick_name": pick_name,
        "n_picks_used": curve["n_picks"],
        "depth_grid_m": depth_grid_m,
        "twt_est_ms": twt_est,
        "easting_est": easting_est,
        "northing_est": northing_est,
        "inline": inline,
        "crossline": crossline,
        "time_idx": time_idx,
        "samples_ms": samples_ms,
    }


def _demo():
    picks_by_well = parse_well_picks()
    print(f"=== step_04 well tie (weak) === 解析到 {len(picks_by_well)} 口井的分层点")

    idx = load_index()
    results = {}
    with SeismicVolume(index=idx) as vol, h5py.File(WELL_LOGS_H5, "r") as f:
        for track_id in f.keys():
            depth_grid = f[track_id]["depth_grid_m"][()].astype(np.float64)
            try:
                res = tie_well(track_id, depth_grid, picks_by_well, vol)
            except (KeyError, RuntimeError) as e:
                print(f"\n井 {track_id}: 跳过 -- {e}")
                continue
            results[track_id] = res
            print(f"\n井 {track_id} (匹配 Well_picks 名 {res['pick_name']!r}, "
                  f"用了 {res['n_picks_used']} 个官方分层点做插值)")
            mid = len(depth_grid) // 2
            for i in (0, mid, len(depth_grid) - 1):
                d = depth_grid[i]
                print(f"  MD={d:.1f}m -> TWT_est={res['twt_est_ms'][i]:.1f}ms, "
                      f"UTM=({res['easting_est'][i]:.1f},{res['northing_est'][i]:.1f}) "
                      f"-> inline={res['inline'][i]}, crossline={res['crossline'][i]}, "
                      f"seismic_time_idx={res['time_idx'][i]} (t={res['samples_ms'][res['time_idx'][i]]:.0f}ms)")

            # 抽一个中间深度点，实际去查那个位置的地震道，证明这个粗略坐标真的能取到数据
            il_q, xl_q, ti_q = int(res["inline"][mid]), int(res["crossline"][mid]), int(res["time_idx"][mid])
            trace = vol.get_trace(il_q, xl_q)
            window = trace[max(0, ti_q - 3): ti_q + 4]
            print(f"  -> 抽样核查: 该(il={il_q},xl={xl_q})地震道在时间样本{ti_q}附近的振幅窗口: {window}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / "well_tie_weak.npz"
        save_kwargs = {}
        for track_id, res in results.items():
            key = track_id.replace(" ", "_")
            save_kwargs[f"{key}__depth_m"] = res["depth_grid_m"].astype("float32")
            save_kwargs[f"{key}__twt_est_ms"] = res["twt_est_ms"].astype("float32")
            save_kwargs[f"{key}__inline"] = res["inline"]
            save_kwargs[f"{key}__crossline"] = res["crossline"]
            save_kwargs[f"{key}__time_idx"] = res["time_idx"]
        np.savez(out_path, **save_kwargs)
        print(f"\n已写出 {out_path}（含 {len(results)} 口井的弱标定结果）")


if __name__ == "__main__":
    _demo()
