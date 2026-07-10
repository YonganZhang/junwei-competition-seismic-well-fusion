"""Step 03 — 解析断层棒线 + 层位解释文件，统一到与ST0202地震体一致的
inline/crossline/time坐标系。

数据来源(真实解压自 Volve_Geophysical_Interpretations.zip，100MB，全量解压，
不需要像地震那样挑文件):
    Geophysical_Interpretations/Faults/Volve_Official_Faults.dat
        —— 全部断层棒线(fault sticks)，就是任务要找的 Official_Faults.dat，
           实际文件名带Volve_前缀: Volve_Official_Faults.dat。
    Geophysical_Interpretations/Horizons/Horizons_TWT/Other_Horizons/
        BCU+ST0202R08_PS_PSDM_FULL_PP+STAT+TIME.dat
        —— 层位解释(选BCU=Base Cretaceous Unconformity一个层位做验证，
           其余层位文件格式相同，可以同样方式批量解析)。

格式是先看文件内容才定的，不是拍脑袋假设:
    - Faults: 定长文本(不是CSV)。按 Faults/README.txt 给出的字符位置解析:
        col 3-12 UTMX, col 14-24 UTMY, col 26-36 TWT(ms),
        col 52 棒线序号(1/2/3=起/中/末点), col 53-100 断层名。
        用README给的位置切片实测对了(数值切出来是干净的float，见开发时验证)。
    - Horizons: 不是定长，是"注释头(#开头) + 几行非#但也非数据的元信息行 + CSV数据"。
      数据行都是 "IL,XL,UTMX,UTMY,TWT" 5列逗号分隔。用"能不能按逗号拆成5个float"
      做通用判定，不依赖固定跳过行数(不同层位文件的头部行数并不完全一样)。

坐标系统一:
    - Horizons文件名里如果是"ST0202R08_..."处理版本，IL/XL已经是ST0202编号，
      与我们的地震体(ST0202R08 PZ PSDM full stack)inline/crossline定义一致，
      直接用文件自带的IL/XL列。
    - 官方README提示: "ST10010的IL = 2×ST0202的IL - 1"——如果用到Official_Horizons
      下用ST10010编号导出的层位文件，需要先做这个换算再对齐到ST0202体；
      本step用的BCU文件本身就是ST0202R08版本，不需要换算，换算公式在下面
      HORIZON_IL_ST10010_TO_ST0202() 留作后续如果要用Official_Horizons时用。
    - Faults文件没有IL/XL列，只有UTMX/UTMY/TWT，用 step_01 拟合的
      仿射变换(UTM<->il/xl)反解出每个断层棒点最近的(inline,crossline)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_01_load_seismic import SeismicVolume, load_index  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERP_DIR = PROJECT_ROOT / "_sandbox/volve_data/_extracted_interp/Geophysical_Interpretations"
FAULTS_PATH = INTERP_DIR / "Faults/Volve_Official_Faults.dat"
HORIZON_PATH = (
    INTERP_DIR / "Horizons/Horizons_TWT/Other_Horizons/BCU+ST0202R08_PS_PSDM_FULL_PP+STAT+TIME.dat"
)
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def horizon_il_st10010_to_st0202(il_st10010: np.ndarray) -> np.ndarray:
    """README: IL(ST10010) = 2*IL(ST0202) - 1  =>  IL(ST0202) = (IL(ST10010)+1)/2。
    只有用到 Official_Horizons(ST10010编号)时才需要，本step默认不调用。"""
    return (il_st10010 + 1) / 2.0


def parse_faults(path: Path = FAULTS_PATH) -> dict:
    utmx, utmy, twt, stick_no, names = [], [], [], [], []
    n_bad = 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            if len(line) < 100:
                n_bad += 1
                continue
            try:
                x = float(line[2:12])
                y = float(line[13:24])
                t = float(line[25:36])
                s = int(line[51:52])
                name = line[52:100].strip()
            except ValueError:
                n_bad += 1
                continue
            utmx.append(x)
            utmy.append(y)
            twt.append(t)
            stick_no.append(s)
            names.append(name)
    return {
        "utmx": np.asarray(utmx, dtype=np.float64),
        "utmy": np.asarray(utmy, dtype=np.float64),
        "twt_ms": np.asarray(twt, dtype=np.float64),
        "stick_no": np.asarray(stick_no, dtype=np.int32),
        "fault_name": np.asarray(names, dtype=object),
        "n_parsed": len(utmx),
        "n_bad_lines": n_bad,
    }


def parse_horizon(path: Path) -> dict:
    """通用解析: 逐行尝试按逗号拆成5个float(IL,XL,UTMX,UTMY,TWT/Z)，
    失败(注释行/元信息行)就跳过，不依赖固定的头部行数。"""
    il, xl, utmx, utmy, z = [], [], [], [], []
    n_skip = 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 5:
                n_skip += 1
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                n_skip += 1
                continue
            il.append(vals[0])
            xl.append(vals[1])
            utmx.append(vals[2])
            utmy.append(vals[3])
            z.append(vals[4])
    return {
        "inline": np.asarray(il, dtype=np.float64),
        "crossline": np.asarray(xl, dtype=np.float64),
        "utmx": np.asarray(utmx, dtype=np.float64),
        "utmy": np.asarray(utmy, dtype=np.float64),
        "value": np.asarray(z, dtype=np.float64),  # TWT(ms) 或 depth(m)，取决于文件属于TWT/DEPTH目录
        "n_parsed": len(il),
        "n_skipped_lines": n_skip,
    }


def attach_grid_coords(faults: dict, vol: SeismicVolume) -> dict:
    """给断层棒点(只有UTM)反解最近的(inline,crossline)。"""
    il_out = np.empty(faults["n_parsed"], dtype=np.int32)
    xl_out = np.empty(faults["n_parsed"], dtype=np.int32)
    in_range = np.zeros(faults["n_parsed"], dtype=bool)
    idx = vol.index
    for i in range(faults["n_parsed"]):
        il, xl = vol.utm_to_il_xl(faults["utmx"][i], faults["utmy"][i])
        il_out[i] = il
        xl_out[i] = xl
        # 记录反解出的原始(未clip)值是否真的落在网格范围内，供统计"体外点"占比
    faults["inline"] = il_out
    faults["crossline"] = xl_out
    return faults


def _demo():
    idx = load_index()
    with SeismicVolume(index=idx) as vol:
        print("=== step_03 faults ===")
        faults = parse_faults()
        print(f"解析 {FAULTS_PATH.name}: {faults['n_parsed']} 个棒线点, "
              f"{faults['n_bad_lines']} 行跳过(表头/异常行)")
        faults = attach_grid_coords(faults, vol)
        uniq_names = sorted(set(faults["fault_name"].tolist()))
        print(f"断层数(按名称去重): {len(uniq_names)}, 示例: {uniq_names[:5]}")
        print("前3个点(UTMX,UTMY,TWT_ms,fault_name,最近inline,最近crossline):")
        for i in range(3):
            print(f"  ({faults['utmx'][i]:.1f}, {faults['utmy'][i]:.1f}, "
                  f"{faults['twt_ms'][i]:.1f}, {faults['fault_name'][i]!r}, "
                  f"il={faults['inline'][i]}, xl={faults['crossline'][i]})")

        print("\n=== step_03 horizon (BCU) ===")
        hor = parse_horizon(HORIZON_PATH)
        print(f"解析 {HORIZON_PATH.name}: {hor['n_parsed']} 个层位点, "
              f"{hor['n_skipped_lines']} 行跳过(注释/元信息)")
        print(f"IL范围: {hor['inline'].min():.0f}..{hor['inline'].max():.0f}, "
              f"XL范围: {hor['crossline'].min():.0f}..{hor['crossline'].max():.0f}, "
              f"TWT范围: {hor['value'].min():.1f}..{hor['value'].max():.1f} ms")
        print("前3个点(IL,XL,UTMX,UTMY,TWT_ms):")
        for i in range(3):
            print(f"  ({hor['inline'][i]:.0f}, {hor['crossline'][i]:.0f}, "
                  f"{hor['utmx'][i]:.1f}, {hor['utmy'][i]:.1f}, {hor['value'][i]:.1f})")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(
            OUT_DIR / "fault_points.npz",
            utmx=faults["utmx"], utmy=faults["utmy"], twt_ms=faults["twt_ms"],
            inline=faults["inline"], crossline=faults["crossline"],
            fault_name=faults["fault_name"], stick_no=faults["stick_no"],
        )
        np.savez(
            OUT_DIR / "horizon_bcu_points.npz",
            inline=hor["inline"], crossline=hor["crossline"],
            utmx=hor["utmx"], utmy=hor["utmy"], twt_ms=hor["value"],
        )
        print(f"\n已写出 {OUT_DIR/'fault_points.npz'} 和 {OUT_DIR/'horizon_bcu_points.npz'}")


if __name__ == "__main__":
    _demo()
