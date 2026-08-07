#!/usr/bin/env python3
"""P45 (c): does the I0 physics baseline drift when the SEG-Y source changes
time range / sampling parameters?

GitHub issue #1's own failure mode: "模型训练时使用固定的0-5000ms时间轴...
未知工区SEG-Y仅覆盖0-2500ms时,归一化位置被错误映射到新的物理时间范围,导致
预测随记录长度发生尺度漂移". This does not require new data: Volve already
has two independently processed seismic vintages over the SAME wells --
ST0202 (time range 0-4500ms, inline 9985-10369, crossline 1932-2536) and
ST10010 (time range 12-3408ms -- a materially SHORTER window, inline
9961-10361, crossline 1961-2680, different processing/migration run).

This reruns the exact same I0 pipeline (identical code, same T0_SEARCH_MS,
same wavelet-estimation-from-real-trace, same ambiguity gate) against
ST10010 instead of ST0202, for the same 3 wells, and compares predicted t0
and MAE against the ST0202 result. Because I0's design never hardcodes an
absolute time axis (t0 is searched fresh from each survey's own samples_ms,
and the wavelet frequency is estimated from each survey's own trace), a
robust design should show comparable ambiguity/ MAE behaviour on both
surveys, not a mechanical shift proportional to the record-length change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "01_common_preprocess"))

import segyio

from p45_well_tie_physics_baseline import (  # noqa: E402
    WELLS, CHECKSHOT_MEMBER, VSP_ZIP, PROJECT_ROOT, RAW_PROJECT_ROOT,
    predict_well, metrics, parse_checkshots,
)

ST10010_SEGY = (
    RAW_PROJECT_ROOT
    / "_sandbox/volve_data/_extracted_seismic/ST10010/Stacks/"
    "ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.JS-017536.segy"
)
OUTPUT_DIR = HERE / "_outputs/p45_well_tie_st10010_robustness"
I0_SUMMARY = HERE / "_outputs/p45_well_tie_physics_baseline/summary.json"


def build_index(segy_path: Path) -> dict:
    """Same affine-fit index as step_01, but tolerant of a non-full grid
    (ST10010 has missing traces) via a (il,xl)->trace_index lookup array
    instead of the linear formula step_01 requires."""
    with segyio.open(str(segy_path), "r", ignore_geometry=True) as f:
        n_traces = f.tracecount
        il = f.attributes(segyio.TraceField.INLINE_3D)[:].astype(np.int32)
        xl = f.attributes(segyio.TraceField.CROSSLINE_3D)[:].astype(np.int32)
        cdpx = f.attributes(segyio.TraceField.CDP_X)[:].astype(np.float64)
        cdpy = f.attributes(segyio.TraceField.CDP_Y)[:].astype(np.float64)
        scalco = int(f.header[0][segyio.TraceField.SourceGroupScalar])
        samples = np.asarray(f.samples, dtype=np.float64)

    scale = (1.0 / -scalco) if scalco < 0 else float(scalco or 1)
    utmx, utmy = cdpx * scale, cdpy * scale

    il_min, il_max = int(il.min()), int(il.max())
    xl_min, xl_max = int(xl.min()), int(xl.max())
    n_il, n_xl = il_max - il_min + 1, xl_max - xl_min + 1
    trace_map = np.full((n_il, n_xl), -1, dtype=np.int32)
    trace_map[il - il_min, xl - xl_min] = np.arange(n_traces, dtype=np.int32)
    missing = int(np.sum(trace_map < 0))

    A = np.column_stack([il.astype(np.float64), xl.astype(np.float64), np.ones(n_traces)])
    coef_x, *_ = np.linalg.lstsq(A, utmx, rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, utmy, rcond=None)
    fit_rms_m = float(np.sqrt(np.mean((A @ coef_x - utmx) ** 2 + (A @ coef_y - utmy) ** 2)))

    return {
        "il_min": il_min, "il_max": il_max, "xl_min": xl_min, "xl_max": xl_max,
        "n_il": n_il, "n_xl": n_xl, "n_traces": n_traces, "samples_ms": samples,
        "affine_il_xl_to_xy": np.array([coef_x, coef_y]),
        "affine_fit_rms_m": fit_rms_m, "trace_map": trace_map, "missing_traces": missing,
    }


class ST10010Volume:
    """Minimal drop-in for step_01.SeismicVolume that tolerates ST10010's
    non-full (il,xl) grid via trace_map instead of the linear formula."""

    def __init__(self, segy_path: Path, index: dict):
        self.segy_path = Path(segy_path)
        self.index = index
        self._f = segyio.open(str(self.segy_path), "r", ignore_geometry=True)

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def get_trace(self, inline: int, crossline: int) -> np.ndarray:
        idx = self.index
        if not (idx["il_min"] <= inline <= idx["il_max"] and idx["xl_min"] <= crossline <= idx["xl_max"]):
            raise ValueError(f"(il={inline},xl={crossline}) outside ST10010 grid")
        trace_index = int(idx["trace_map"][inline - idx["il_min"], crossline - idx["xl_min"]])
        if trace_index < 0:
            raise ValueError(f"(il={inline},xl={crossline}) is a missing trace in ST10010")
        return np.asarray(self._f.trace[trace_index], dtype=np.float32)

    def nearest_valid_il_xl(self, inline: float, crossline: float) -> tuple[int, int]:
        idx = self.index
        il = int(np.clip(round(inline), idx["il_min"], idx["il_max"]))
        xl = int(np.clip(round(crossline), idx["xl_min"], idx["xl_max"]))
        return il, xl

    def utm_to_il_xl(self, utmx: float, utmy: float) -> tuple[int, int]:
        a, b, c = self.index["affine_il_xl_to_xy"][0]
        d, e, f = self.index["affine_il_xl_to_xy"][1]
        M = np.array([[a, b], [d, e]], dtype=np.float64)
        rhs = np.array([utmx - c, utmy - f], dtype=np.float64)
        il_f, xl_f = np.linalg.solve(M, rhs)
        il, xl = self.nearest_valid_il_xl(il_f, xl_f)
        # if the nearest grid node is itself a missing trace, spiral out to
        # the nearest populated node rather than silently failing
        if self.index["trace_map"][il - self.index["il_min"], xl - self.index["xl_min"]] < 0:
            for radius in range(1, 10):
                found = False
                for dil in range(-radius, radius + 1):
                    for dxl in range(-radius, radius + 1):
                        cand_il, cand_xl = il + dil, xl + dxl
                        if not (self.index["il_min"] <= cand_il <= self.index["il_max"] and
                                self.index["xl_min"] <= cand_xl <= self.index["xl_max"]):
                            continue
                        if self.index["trace_map"][cand_il - self.index["il_min"], cand_xl - self.index["xl_min"]] >= 0:
                            il, xl, found = cand_il, cand_xl, True
                            break
                    if found:
                        break
                if found:
                    break
        return il, xl


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    st0202_summary = json.loads(I0_SUMMARY.read_text(encoding="utf-8"))

    st10010_index = build_index(ST10010_SEGY)
    checkshots = parse_checkshots(VSP_ZIP, CHECKSHOT_MEMBER)

    results = {}
    with ST10010Volume(segy_path=ST10010_SEGY, index=st10010_index) as vol:
        for well in WELLS:
            try:
                pred = predict_well(well, st10010_index, vol)
            except ValueError as exc:
                results[well] = {"status": "out_of_grid", "error": str(exc)}
                print(f"{well}: OUT OF GRID on ST10010 -- {exc}")
                continue

            depth_true, twt_true = checkshots[well]
            pred_at_checkshot_depth = np.interp(
                depth_true, pred["depth_tvdss"], pred["predicted_twt_ms"],
                left=np.nan, right=np.nan,
            )
            valid = np.isfinite(pred_at_checkshot_depth)
            valid &= (depth_true >= pred["depth_tvdss"].min()) & (depth_true <= pred["depth_tvdss"].max())
            m = metrics(twt_true[valid], pred_at_checkshot_depth[valid])

            st0202_r = st0202_summary["wells"][well]
            results[well] = {
                "status": "evaluated",
                "st10010_t0_ms": pred["coarse_t0_ms"],
                "st10010_xcorr": pred["coarse_xcorr"],
                "st10010_ambiguity_gap": pred["ambiguity_gap"],
                "st10010_reject_ambiguous": pred["reject_ambiguous"],
                "st10010_metrics": m,
                "st0202_t0_ms": st0202_r["coarse_t0_ms"],
                "st0202_metrics": st0202_r["metrics_vs_own_checkshot"],
                "t0_delta_ms": pred["coarse_t0_ms"] - st0202_r["coarse_t0_ms"],
                "mae_delta_ms": m["mae_ms"] - st0202_r["metrics_vs_own_checkshot"]["mae_ms"],
                "both_rejected_or_both_kept": (
                    pred["reject_ambiguous"] == st0202_r["reject_ambiguous"]
                ),
            }
            print(f"{well}: ST0202 t0={st0202_r['coarse_t0_ms']:.0f}ms MAE={st0202_r['metrics_vs_own_checkshot']['mae_ms']:.1f}ms  "
                  f"| ST10010 t0={pred['coarse_t0_ms']:.0f}ms MAE={m['mae_ms']:.1f}ms  "
                  f"| t0_delta={results[well]['t0_delta_ms']:+.0f}ms mae_delta={results[well]['mae_delta_ms']:+.1f}ms "
                  f"gate_agrees={results[well]['both_rejected_or_both_kept']}")

    n_evaluated = sum(1 for r in results.values() if r.get("status") == "evaluated")
    n_gate_agrees = sum(1 for r in results.values() if r.get("both_rejected_or_both_kept"))
    summary = {
        "st10010_survey": {
            "segy": str(ST10010_SEGY),
            "time_range_ms": [float(st10010_index["samples_ms"].min()), float(st10010_index["samples_ms"].max())],
            "sample_interval_ms": float(st10010_index["samples_ms"][1] - st10010_index["samples_ms"][0]),
            "il_range": [st10010_index["il_min"], st10010_index["il_max"]],
            "xl_range": [st10010_index["xl_min"], st10010_index["xl_max"]],
        },
        "st0202_survey_reference": {"time_range_ms": [0.0, 4500.0], "sample_interval_ms": 4.0},
        "results": results,
        "verdict": {
            "n_wells_evaluated_on_both": n_evaluated,
            "n_wells_ambiguity_gate_agrees": n_gate_agrees,
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwrote", OUTPUT_DIR / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
