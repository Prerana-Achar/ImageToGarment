from __future__ import annotations

import torch

from garment_tree_model import GarmentTreeConfig, GarmentTreeNet


def tiny_schema():
    return {
        "n_cont": 0,
        "n_const": 4,
        "n_cat": 4,
        "cont_slots": {},
        "const_slots": {
            "shirt.length": 0,
            "collar.width": 1,
            "sleeve.length": 2,
            "pants.length": 3,
        },
        "const_ranges": {
            "shirt.length": [0.5, 3.5],
            "collar.width": [-0.5, 1.0],
            "sleeve.length": [0.1, 1.2],
            "pants.length": [0.2, 0.9],
        },
        "cat_vocab": {
            "meta.upper": ["Shirt", None],
            "meta.wb": ["StraightWB", None],
            "meta.bottom": ["Pants", None],
            "sleeve.sleeveless": [True, False],
        },
    }


def test_forward_shapes_and_backward():
    config = GarmentTreeConfig(
        widths=(16, 24, 32, 48),
        depths=(1, 1, 1, 1),
        dimension=48,
        query_layers=1,
        attention_heads=4,
        dropout=0.0,
        drop_path=0.0,
    )
    model = GarmentTreeNet(tiny_schema(), config)
    image = torch.randn(2, 3, 64, 64)
    categories = torch.tensor([[0, 0, 0, 1], [0, 1, 1, 0]])
    output = model(image, categories, teacher_force=0.5)

    assert output["numeric_mean"].shape == (2, 4)
    assert output["numeric_log_scale"].shape == (2, 4)
    assert output["numeric_activity"].shape == (2, 4)
    assert output["categorical_activity"].shape == (2, 4)
    assert [item.shape for item in output["categorical_logits"]] == [
        (2, 2),
        (2, 2),
        (2, 2),
        (2, 2),
    ]
    assert output["part_features"].shape == (2, 12, 48)
    assert torch.all((0.0 <= output["numeric_mean"]) & (output["numeric_mean"] <= 1.0))

    loss = output["numeric_mean"].mean()
    loss = loss + sum(item.mean() for item in output["categorical_logits"])
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

