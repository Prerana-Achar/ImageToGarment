#!/usr/bin/env python3
"""Reconstruct one route-constrained GarmentCode design YAML from one image."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torchvision import transforms
import yaml

from garment_ensemble_model import (
    GarmentEnsembleConfig,
    GarmentTreeEnsemble,
    ROOT_PATHS,
    RouteConstraints,
)
from garment_tree_model import GarmentTreeConfig, GarmentTreeNet
from prepare_data import PadToSquare


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--yaml-out")
    parser.add_argument("--json-out")
    parser.add_argument(
        "--template",
        default=str(root / "GarmentCodeRC/assets/design_params/default_new.yaml"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def image_transform(size: int, aspect_pad: bool = False) -> transforms.Compose:
    operations = [PadToSquare()] if aspect_pad else []
    operations.extend(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    return transforms.Compose(operations)


def get_spec(design: dict[str, Any], path: str) -> dict[str, Any]:
    node: Any = design
    for part in path.split("."):
        node = node[part]
    if not isinstance(node, dict) or "v" not in node:
        raise KeyError(path)
    return node


def cast_value(value: Any, specification: dict[str, Any]) -> Any:
    kind = specification.get("type")
    if value is None:
        return None
    if kind == "int":
        return int(round(float(value)))
    if kind == "float":
        return float(value)
    if kind == "bool":
        return bool(value)
    return value


def decode(
    output: dict[str, Any], schema: dict[str, Any], template: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    design = copy.deepcopy(template["design"])
    numeric_mean = output["numeric_mean"][0].detach().cpu().float().clamp(0.0, 1.0)
    numeric_scale = output["numeric_log_scale"][0].detach().cpu().float().exp()
    numeric_activity = output["numeric_activity"][0].detach().cpu().float().sigmoid()
    route_numeric = output.get("route_numeric_mask")
    route_categorical = output.get("route_categorical_mask")
    if route_numeric is not None:
        route_numeric = route_numeric[0].detach().cpu().bool()
    if route_categorical is not None:
        route_categorical = route_categorical[0].detach().cpu().bool()
    root_selection = output.get("root_selection")
    if root_selection is not None:
        root_selection = root_selection[0].detach().cpu().long()
    details: dict[str, Any] = {
        "numeric": {},
        "categorical": {},
        "inactive_paths": [],
    }

    numeric_index = 0
    for path in [*schema.get("cont_slots", {}), *schema.get("const_slots", {})]:
        specification = get_spec(design, path)
        if path in schema.get("cont_slots", {}):
            low, high = map(float, specification["range"])
        else:
            low, high = map(float, schema["const_ranges"][path])
        normalized = float(numeric_mean[numeric_index])
        raw = low + normalized * (high - low)
        route_allowed = route_numeric is None or bool(route_numeric[numeric_index])
        predicted_active = float(numeric_activity[numeric_index]) >= 0.5
        active = route_allowed and predicted_active
        if active:
            specification["v"] = cast_value(raw, specification)
        else:
            details["inactive_paths"].append(path)
        details["numeric"][path] = {
            "active": active,
            "route_allowed": route_allowed,
            "normalized": normalized if active else None,
            "value": specification["v"],
            "uncertainty_normalized": float(numeric_scale[numeric_index]),
            "active_probability": float(numeric_activity[numeric_index]),
        }
        numeric_index += 1
    root_positions = {path: position for position, path in enumerate(ROOT_PATHS)}
    for index, (path, vocab) in enumerate(schema["cat_vocab"].items()):
        logits = output["categorical_logits"][index][0].detach().cpu().float()
        probability = logits.softmax(dim=-1)
        if root_selection is not None and path in root_positions:
            predicted_index = int(root_selection[root_positions[path]])
        else:
            predicted_index = int(probability.argmax())
        value = vocab[predicted_index]
        specification = get_spec(design, path)
        route_allowed = route_categorical is None or bool(route_categorical[index])
        active_probability = float(
            output["categorical_activity"][0, index].detach().cpu().sigmoid()
        )
        is_root = path in root_positions
        active = route_allowed and (is_root or active_probability >= 0.5)
        if active:
            specification["v"] = cast_value(value, specification)
        else:
            details["inactive_paths"].append(path)
        details["categorical"][path] = {
            "active": active,
            "route_allowed": route_allowed,
            "value": specification["v"],
            "predicted_value": cast_value(value, specification) if active else None,
            "index": predicted_index,
            "confidence": float(probability[predicted_index]),
            "probabilities": [float(item) for item in probability],
            "active_probability": active_probability,
        }
    if root_selection is not None:
        bottom_index = root_positions["meta.bottom"]
        details["selected_lower_family"] = schema["cat_vocab"]["meta.bottom"][
            int(root_selection[bottom_index])
        ]
    return {"design": design}, details


def build_model(
    checkpoint: dict[str, Any], device: torch.device
) -> tuple[Any, dict[str, Any], str]:
    format_name = checkpoint.get("format")
    schema = checkpoint["schema"]
    if format_name == "GarmentTreeEnsemble/checkpoint-v1":
        config = GarmentEnsembleConfig.from_dict(checkpoint["model_config"])
        constraints = RouteConstraints.from_dict(checkpoint["route_constraints"])
        model = GarmentTreeEnsemble(schema, constraints, config).to(device)
        model.load_state_dict(checkpoint["model"])
        return model.eval(), schema, "GarmentTreeEnsemble/prediction-v1"
    if format_name == "GarmentTreeNet/checkpoint-v1":
        config = GarmentTreeConfig.from_dict(checkpoint["model_config"])
        model = GarmentTreeNet(schema, config).to(device)
        model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
        return model.eval(), schema, "GarmentTreeNet/prediction-v1"
    raise SystemExit(f"Unsupported checkpoint format: {format_name!r}")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, schema, prediction_format = build_model(checkpoint, device)
    train_args = checkpoint.get("train_args", {})
    image_size = int(train_args.get("image_size", 288))
    image = Image.open(args.image).convert("RGB")
    tensor = image_transform(
        image_size, aspect_pad=bool(train_args.get("aspect_pad", False))
    )(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(tensor)

    template = yaml.safe_load(Path(args.template).read_text())
    design, details = decode(output, schema, template)
    image_path = Path(args.image)
    yaml_path = Path(args.yaml_out) if args.yaml_out else image_path.with_name(
        f"{image_path.stem}_garmentcode.yaml"
    )
    json_path = Path(args.json_out) if args.json_out else yaml_path.with_suffix(".json")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(design, sort_keys=False, default_flow_style=False))
    json_path.write_text(
        json.dumps(
            {
                "format": prediction_format,
                "source_image": str(image_path.resolve()),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "validation": checkpoint.get("validation"),
                "prediction": details,
            },
            indent=2,
        )
        + "\n"
    )
    roots = details["categorical"]
    root_confidence = min(
        roots[path]["confidence"] for path in ROOT_PATHS if path in roots
    )
    print(f"wrote {yaml_path}")
    print(f"wrote {json_path}")
    print(f"root topology minimum confidence: {root_confidence:.3f}")
    if "selected_lower_family" in details:
        print(f"selected lower family: {details['selected_lower_family']}")


if __name__ == "__main__":
    main()