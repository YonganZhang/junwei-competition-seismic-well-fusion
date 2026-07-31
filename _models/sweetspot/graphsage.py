"""Thin PyG GraphSAGE adapter with explicit graph inputs and no cross-split edges."""
from __future__ import annotations

from typing import Any

from .p5_common import TorchModuleAdapter, output_dim, require_single_target

model_id = "graphsage"


def capabilities() -> dict[str, Any]:
    return {"task_types": ["binary", "multiclass", "regression"], "input_modalities": ["graph"], "supports_missing_mask": True, "supports_uncertainty": False, "stage1_input_key": "graph_x", "required_inputs": ["graph_x", "edge_index"]}


def build_model(task_spec, **config):
    require_single_target(task_spec, capabilities()["task_types"])
    try:
        from torch_geometric.nn import GraphSAGE
    except ImportError as exc:
        from .p5_common import dependency_skip
        raise dependency_skip("torch_geometric") from exc
    in_channels = int(config.get("in_channels", 4)); hidden = int(config.get("hidden_channels", 16)); out = output_dim(task_spec)
    device = str(config.get("device", "cpu"))
    factory = lambda: GraphSAGE(in_channels, hidden, num_layers=2, out_channels=out, dropout=0.0)
    def forward(module, inputs, torch):
        if not {"graph_x", "edge_index"} <= set(inputs):
            raise ValueError("GraphSAGE requires graph_x and split-pruned edge_index")
        x = torch.as_tensor(inputs["graph_x"], dtype=torch.float32, device=device)
        edge = torch.as_tensor(inputs["edge_index"], dtype=torch.long, device=device)
        return module(x, edge)
    return TorchModuleAdapter(task_spec, factory, input_key="graph_x", device=device, forward_fn=forward)
