#!/usr/bin/env python3
"""Probe root garment predictions on a few GarmentImage examples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

from PIL import Image
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from garment_ensemble_model import ROOT_PATHS
from infer_garment_tree import build_model, choose_device, decode, image_transform

DEFAULT_IMAGES = (
    "Bottoms/wide_leg_pants.jpg",
    "Bottoms/straight_leg_pants.jpg",
    "Bottoms/palazzo_pants.jpg",
    "Tops/button_up_shirt.jpg",
    "Tops/t_shirt.jpg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-root", type=Path, default=Path("/is/cluster/fast/pachar/Data/GarmentImage"))
    parser.add_argument("--image", action="append", dest="images", help="Relative path under --image-root or absolute image path. Repeatable.")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--template", type=Path, default=Path("GarmentCodeRC/assets/design_params/default_new.yaml"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_images(root: Path, values: list[str] | None) -> list[Path]:
    selected = values or list(DEFAULT_IMAGES)
    paths = []
    for value in selected:
        path = Path(value)
        paths.append(path if path.is_absolute() else root / path)
    return paths


def root_summary(details: dict[str, Any]) -> dict[str, Any]:
    categorical = details["categorical"]
    return {
        path: {
            "value": categorical[path]["value"],
            "confidence": categorical[path]["confidence"],
        }
        for path in ROOT_PATHS
    }


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, schema, prediction_format = build_model(checkpoint, device)
    train_args = checkpoint.get("train_args", {})
    image_size = int(train_args.get("image_size", 288))
    transform = image_transform(image_size, aspect_pad=bool(train_args.get("aspect_pad", False)))
    template = yaml.safe_load(args.template.read_text())
    rows = []
    for image_path in resolve_images(args.image_root, args.images):
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model(tensor)
        _, details = decode(output, schema, template)
        roots = root_summary(details)
        row = {
            "image": str(image_path),
            "prediction_format": prediction_format,
            "checkpoint": str(Path(args.checkpoint)),
            "roots": roots,
            "selected_lower_family": details.get("selected_lower_family"),
        }
        rows.append(row)
        root_text = " ".join(
            f"{path}={roots[path]['value']!r}@{roots[path]['confidence']:.3f}"
            for path in ROOT_PATHS
        )
        print(f"{image_path.name}: {root_text}", flush=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()