# 01_common_preprocess — Layer1 公共预处理

6条赛道共用，不属于任何单一赛道。产出供 `_pipelines/02_task_datasets/` 下各赛道复用。

## Steps

- `step_01_load_seismic.py`：读取Volve地震体（SEG-Y，用segyio），建立inline/crossline/time网格索引，导出为numpy/zarr方便后续快速切片，不必每次重新解析SEG-Y。
- `step_02_load_well_logs.py`：读取Volve测井曲线（LAS，用lasio），做去噪/缺失值插补/深度重采样/归一化（泥岩基线归一化）。
- `step_03_load_fault_horizon.py`：解析 `Official_Faults.dat` 断层棒线 + 层位解释文件，统一到与地震体一致的inline/crossline/time坐标系。
- `step_04_well_tie_weak.py`：弱井震标定——比赛不提供VSP/合成记录/时深表，只能用经验速度函数做深度→时间的近似换算，把测井定位到地震体坐标，成果需要人工抽样核查（画图对比，见验证方式）。

## 输出

`outputs/` 下存放中间产物（地震索引、清洗后测井曲线、标准化坐标的断层/层位点集），供Layer2直接读取，不重复解析原始zip。

## 验证方式

跑完 `step_04` 后，抽样几口井，画"井位置的测井曲线 + 对应地震道波形"对比图，人工核查标定是否合理（弱标定精度有限，这是已知局限，见 `_wiki-methodology/_wiki/_methods/pipeline-skeleton.md`）。
