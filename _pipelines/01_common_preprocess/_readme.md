# 01_common_preprocess — Layer1 公共预处理

6条赛道共用，不属于任何单一赛道。产出供 `_pipelines/02_task_datasets/` 下各赛道复用。

## Steps

- `step_01_load_seismic.py`：读取Volve地震体（SEG-Y，用segyio），建立inline/crossline/time网格索引，导出为numpy/zarr方便后续快速切片，不必每次重新解析SEG-Y。
- `step_02_load_well_logs.py`：读取Volve测井曲线（LAS，用lasio），做去噪/缺失值插补/深度重采样/归一化（泥岩基线归一化）。
- `step_03_load_fault_horizon.py`：解析 `Official_Faults.dat` 断层棒线 + 层位解释文件，统一到与地震体一致的inline/crossline/time坐标系。
- `step_04_well_tie_weak.py`：早期弱井震标定产物，使用稀疏分层点对 MD–TWT 做近似换算。比赛输入本身不含 VSP/checkshot，但项目后续下载的 Volve VSP 归档含 5 口井的 checkshot。P23 用 3 口井拟合、2 口从未参与拟合的井独立校验，已证明 checkshot 时深关系显著优于该旧弱标定；详见 `011-p23-reconstruction-checkshot-target-tie.md`。旧 NPZ 仅为保持已有管线可复现，不应解读为最佳时深标定。

## 输出

`outputs/` 下存放中间产物（地震索引、清洗后测井曲线、标准化坐标的断层/层位点集），供Layer2直接读取，不重复解析原始zip。

## 验证方式

跑完 `step_04` 后，抽样几口井，画"井位置的测井曲线 + 对应地震道波形"对比图，人工核查标定是否合理。若用于新实验，应优先使用 P23 的 3 拟合井/2 独立校验井口径，并将"标定误差"与"下游孔隙度误差"分开报告。
