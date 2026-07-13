# 统一模型目录

规范路径是 `_models/<track>/<model_id>.py`。每个模型模块必须导出：

```python
model_id = "example"

def capabilities() -> dict:
    ...

def build_model(task_spec, **model_config):
    ...
```

可选导出 `suggest_hparams(trial, task_spec)`。模型只能实现网络/估计器本身，不得读取 test、划分数据、拟合预处理、计算正式 test 指标或画图。动态发现入口是 `_code.ml_framework.model_discovery.discover_model()`。

赛道旧 `models/` 目录只允许保留短兼容 shim；真实实现只能在这里存在一份。
