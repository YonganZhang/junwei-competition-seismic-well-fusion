"""Step 01 — 读取Volve ST0202地震体（SEG-Y），建立inline/crossline/time网格索引。

数据来源(真实解压自 Volve_Seismic_ST0202.zip，只解出这一个文件，1.1GB，其余
Raw_data/PreMig_data/Prestack_data 里的原始炮集/叠前道集(合计1TB+)不解压):
    ST0202/Stacks/ST0202R08_PZ_PSDM_FULL_OFFSET_PP_TIME.MIG_FIN.POST_STACK.3D.JS-017534.segy
    = PZ检波器、PSDM偏移、全偏移距、PP时间域、最终叠后3D体（2008年处理版本）。

实测网格(用segyio读header拿到，不是假设):
    inline  9985..10369 (共385条，步长1)
    crossline 1932..2536 (共605条，步长1)
    trace数 232925 == 385*605，说明这是一个没有缺道的规则网格(规则网格才能用
    简单的 (il,xl) -> trace_index 线性公式，不用建KD树/字典)
    time samples: 0..4500 ms，间隔4ms，1126个采样点
    CDP_X/CDP_Y 有 scalco=-100 (即数值要除以100)，单位UTM31N米

不把整个1.1GB体积load进内存：本文件只在建索引阶段用 segyio 的 header-only
attributes()扫一遍所有道头(只读头，不读波形样本)，真正取波形数据(f.trace[i])
时才按需读那一道，segyio底层走的是文件的懒读取/内存映射，不会一次性铺到RAM。

索引产出 outputs/seismic_index.npz + outputs/seismic_index_meta.json，
供 step_03 / step_04 复用 UTM<->inline/crossline 的仿射变换，不重复解析SEG-Y。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import segyio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEGY_PATH = (
    PROJECT_ROOT
    / "_sandbox/volve_data/_extracted_seismic/ST0202/Stacks/"
    "ST0202R08_PZ_PSDM_FULL_OFFSET_PP_TIME.MIG_FIN.POST_STACK.3D.JS-017534.segy"
)
OUT_DIR = Path(__file__).resolve().parent / "outputs"
INDEX_NPZ = OUT_DIR / "seismic_index.npz"
INDEX_META = OUT_DIR / "seismic_index_meta.json"


def build_index(segy_path: Path = SEGY_PATH) -> dict:
    """扫一遍道头(header-only)，建立inline/crossline网格索引 + UTM仿射变换。"""
    with segyio.open(str(segy_path), "r", ignore_geometry=True) as f:
        n_traces = f.tracecount
        il = f.attributes(segyio.TraceField.INLINE_3D)[:].astype(np.int32)
        xl = f.attributes(segyio.TraceField.CROSSLINE_3D)[:].astype(np.int32)
        cdpx = f.attributes(segyio.TraceField.CDP_X)[:].astype(np.float64)
        cdpy = f.attributes(segyio.TraceField.CDP_Y)[:].astype(np.float64)
        scalco = int(f.header[0][segyio.TraceField.SourceGroupScalar])
        samples = np.asarray(f.samples, dtype=np.float64)  # time axis, ms

    scale = (1.0 / -scalco) if scalco < 0 else float(scalco or 1)
    utmx = cdpx * scale
    utmy = cdpy * scale

    il_min, il_max = int(il.min()), int(il.max())
    xl_min, xl_max = int(xl.min()), int(xl.max())
    n_il = il_max - il_min + 1
    n_xl = xl_max - xl_min + 1
    is_regular_full_grid = n_traces == n_il * n_xl and (
        np.array_equal(il, il_min + np.repeat(np.arange(n_il), n_xl))
        and np.array_equal(xl, xl_min + np.tile(np.arange(n_xl), n_il))
    )
    if not is_regular_full_grid:
        raise RuntimeError(
            "实测网格不是预期的无缺道规则网格(inline慢变/crossline快变步长1)，"
            "get_trace()里的线性索引公式不适用，需要另建 (il,xl)->trace_index 字典。"
        )

    # 最小二乘拟合 (il, xl) -> (UTMX, UTMY) 的仿射变换，供 step_03/04 反解UTM点
    # 落在哪个 (il, xl)。用全部23万道拟合，过定定问题，精度足够网格级定位。
    A = np.column_stack([il.astype(np.float64), xl.astype(np.float64), np.ones(n_traces)])
    coef_x, *_ = np.linalg.lstsq(A, utmx, rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, utmy, rcond=None)
    pred_x = A @ coef_x
    pred_y = A @ coef_y
    fit_rms_m = float(np.sqrt(np.mean((pred_x - utmx) ** 2 + (pred_y - utmy) ** 2)))

    index = {
        "il_min": il_min,
        "il_max": il_max,
        "xl_min": xl_min,
        "xl_max": xl_max,
        "n_il": n_il,
        "n_xl": n_xl,
        "n_traces": n_traces,
        "samples_ms": samples,
        "affine_il_xl_to_xy": np.array([coef_x, coef_y]),  # shape (2,3): [a,b,c] s.t. X=a*il+b*xl+c
        "affine_fit_rms_m": fit_rms_m,
    }
    return index


def save_index(index: dict, segy_path: Path = SEGY_PATH) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        INDEX_NPZ,
        il_min=index["il_min"],
        il_max=index["il_max"],
        xl_min=index["xl_min"],
        xl_max=index["xl_max"],
        n_il=index["n_il"],
        n_xl=index["n_xl"],
        n_traces=index["n_traces"],
        samples_ms=index["samples_ms"],
        affine_il_xl_to_xy=index["affine_il_xl_to_xy"],
    )
    meta = {
        "segy_path": str(segy_path),
        "il_range": [index["il_min"], index["il_max"]],
        "xl_range": [index["xl_min"], index["xl_max"]],
        "n_traces": index["n_traces"],
        "n_samples": len(index["samples_ms"]),
        "sample_interval_ms": float(index["samples_ms"][1] - index["samples_ms"][0]),
        "affine_fit_rms_m": index["affine_fit_rms_m"],
        "note": "affine_il_xl_to_xy 行0=X系数[a,b,c] 行1=Y系数[d,e,f]，X=a*il+b*xl+c",
    }
    INDEX_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index() -> dict:
    if not INDEX_NPZ.exists():
        index = build_index()
        save_index(index)
        return index
    z = np.load(INDEX_NPZ)
    return {k: z[k] for k in z.files}


class SeismicVolume:
    """按需读取单道数据，不整体load进内存。用segyio打开文件句柄常驻，
    每次get_trace()只读那一道的样本(segyio内部为懒读取)。"""

    def __init__(self, segy_path: Path = SEGY_PATH, index: dict | None = None):
        self.segy_path = Path(segy_path)
        self.index = index if index is not None else load_index()
        self._f = segyio.open(str(self.segy_path), "r", ignore_geometry=True)

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def trace_index(self, inline: int, crossline: int) -> int:
        idx = self.index
        if not (idx["il_min"] <= inline <= idx["il_max"] and idx["xl_min"] <= crossline <= idx["xl_max"]):
            raise ValueError(
                f"(inline={inline}, crossline={crossline}) 超出实测网格范围 "
                f"il[{idx['il_min']},{idx['il_max']}] xl[{idx['xl_min']},{idx['xl_max']}]"
            )
        return int((inline - idx["il_min"]) * idx["n_xl"] + (crossline - idx["xl_min"]))

    def get_trace(self, inline: int, crossline: int) -> np.ndarray:
        """查询任意(inline,crossline)对应的一段地震道，按需从磁盘读取该道。"""
        ti = self.trace_index(inline, crossline)
        return np.asarray(self._f.trace[ti], dtype=np.float32)

    def nearest_valid_il_xl(self, inline: float, crossline: float) -> tuple[int, int]:
        idx = self.index
        il = int(np.clip(round(inline), idx["il_min"], idx["il_max"]))
        xl = int(np.clip(round(crossline), idx["xl_min"], idx["xl_max"]))
        return il, xl

    def utm_to_il_xl(self, utmx: float, utmy: float) -> tuple[int, int]:
        """反解仿射变换 X=a*il+b*xl+c, Y=d*il+e*xl+f 为 (il,xl)，
        再clip到实测网格范围内，返回最近的有效grid node(供井位弱定位用)。"""
        a, b, c = self.index["affine_il_xl_to_xy"][0]
        d, e, f = self.index["affine_il_xl_to_xy"][1]
        M = np.array([[a, b], [d, e]], dtype=np.float64)
        rhs = np.array([utmx - c, utmy - f], dtype=np.float64)
        il_f, xl_f = np.linalg.solve(M, rhs)
        return self.nearest_valid_il_xl(il_f, xl_f)


def _demo():
    idx = load_index()
    print("=== step_01 seismic index ===")
    print(f"segy: {SEGY_PATH.name}")
    print(f"inline range: {idx['il_min']}..{idx['il_max']} (n_il={idx['n_il']})")
    print(f"crossline range: {idx['xl_min']}..{idx['xl_max']} (n_xl={idx['n_xl']})")
    print(f"n_traces: {idx['n_traces']}, n_samples: {len(idx['samples_ms'])}, "
          f"sample_interval_ms: {idx['samples_ms'][1]-idx['samples_ms'][0]}")

    with SeismicVolume(index=idx) as vol:
        il_q, xl_q = idx["il_min"] + idx["n_il"] // 2, idx["xl_min"] + idx["n_xl"] // 2
        trace = vol.get_trace(int(il_q), int(xl_q))
        print(f"\n查询 (inline={il_q}, crossline={xl_q}) 的地震道:")
        print(f"  shape={trace.shape}, dtype={trace.dtype}, "
              f"min={trace.min():.4f}, max={trace.max():.4f}, mean={trace.mean():.6f}")

        il_edge, xl_edge = idx["il_min"], idx["xl_min"]
        trace2 = vol.get_trace(int(il_edge), int(xl_edge))
        print(f"查询边界 (inline={il_edge}, crossline={xl_edge}) 的地震道: "
              f"shape={trace2.shape}, first 5 samples={trace2[:5]}")


if __name__ == "__main__":
    _demo()
