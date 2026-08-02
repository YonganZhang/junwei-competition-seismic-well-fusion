# P32 混合智能体优化验收证据

## 验收结论

- ③物性：`RETAIN_HYBRID`；相对确定性 promotion 主指标改善 `4.269645%`，配对种子 `3/3` 获胜，目标级非劣门禁通过。
- ②地震相：`RETAIN_HYBRID`；promotion 等均 mIoU 绝对改善 `0.024698592`，F3 改善 `0.049397183`，Penobscot 不降。
- 两赛道均完成两次独立 provider 调用和真实训练复跑；最终可执行配置、端点指标、门禁结论一致。
- 两赛道完整候选池都不完全一致，因此只验收最终决策稳定性，不宣称候选清单逐项稳定。
- selection 与 promotion 隔离，frozen test 未读取，凭证未持久化。

## 机器门禁

```bash
python3 -m pytest -q _pipelines/02_task_datasets/reservoir/tests/test_p32_hybrid_agent_optimizer.py
python3 _pipelines/02_task_datasets/reservoir/p32_hybrid_agent_optimizer.py verify
python3 _pipelines/02_task_datasets/reservoir/p32_hybrid_agent_optimizer.py verify-replay

${P5_TORCH_PYTHON} -m pytest -q _pipelines/02_task_datasets/facies/tests/test_p32_hybrid_agent_optimizer.py
${P5_TORCH_PYTHON} _pipelines/02_task_datasets/facies/p32_hybrid_agent_optimizer.py verify
${P5_TORCH_PYTHON} _pipelines/02_task_datasets/facies/p32_hybrid_agent_optimizer.py verify-replay

python3 -m unittest -v _pipelines/02_task_datasets/tests/test_track_lifecycle.py
```

## 证据文件

- `_pipelines/02_task_datasets/reservoir/_outputs/p32_hybrid_agent_optimizer/summary.json`
- `_pipelines/02_task_datasets/reservoir/_outputs/p32_hybrid_agent_optimizer/independent_verification.json`
- `_pipelines/02_task_datasets/facies/_outputs/p32_hybrid_agent_optimizer/summary.json`
- `_pipelines/02_task_datasets/facies/_outputs/p32_hybrid_agent_optimizer/independent_verification.json`
