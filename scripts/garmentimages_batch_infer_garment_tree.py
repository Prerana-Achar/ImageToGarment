#!/usr/bin/env python3
"""Route-constrained ensemble adapter for the GarmentImage comparison pipeline."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
import torch
from torch import nn
from PIL import Image
from torchvision import transforms

import garmentimages_batch_infer as batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from garment_tree_model import GarmentTreeConfig, GarmentTreeNet
from garment_ensemble_model import (
    GarmentEnsembleConfig,
    GarmentTreeEnsemble,
    ROOT_PATHS,
    RouteConstraints,
)
from infer_garment_tree import decode as decode_ensemble_design

DEFAULT_IMAGE_ROOT = Path("/is/cluster/fast/pachar/Data/GarmentImage")
ENSEMBLE_FORMAT = "GarmentTreeEnsemble/checkpoint-v1"
SINGLE_FORMAT = "GarmentTreeNet/checkpoint-v1"


class BaselineContractAdapter(nn.Module):
    """Expose packed tensors while preserving constrained root selections."""

    def __init__(self, model: nn.Module, schema: dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        self.categorical_paths = list(schema["cat_vocab"])

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(image)
        logits = list(output["categorical_logits"])
        if "root_selection" in output:
            logits = [field.clone() for field in logits]
            for position, path in enumerate(ROOT_PATHS):
                index = self.categorical_paths.index(path)
                selected = output["root_selection"][:, position]
                logits[index].fill_(-30.0)
                logits[index].scatter_(1, selected[:, None], 30.0)
        return output["numeric_mean"], torch.cat(logits, dim=-1)


_original_checkpoint_architecture = batch.checkpoint_architecture
_original_baseline_builder = batch.build_baseline_inference_model
_original_write_model_yaml = batch.write_model_yaml
_original_torch_load = torch.load
_original_load_image = batch.load_image
_aspect_pad_inputs = False


def compatible_checkpoint_architecture(checkpoint: dict[str, Any]) -> str:
    if checkpoint.get("format") in (SINGLE_FORMAT, ENSEMBLE_FORMAT):
        return "baseline"
    return _original_checkpoint_architecture(checkpoint)


def compatible_baseline_builder(
    checkpoint: dict[str, Any], device: torch.device
) -> nn.Module:
    format_name = checkpoint.get("format")
    if format_name == ENSEMBLE_FORMAT:
        config = GarmentEnsembleConfig.from_dict(checkpoint["model_config"])
        constraints = RouteConstraints.from_dict(checkpoint["route_constraints"])
        model = GarmentTreeEnsemble(checkpoint["schema"], constraints, config).to(device)
        model.load_state_dict(checkpoint["model"])
        return BaselineContractAdapter(model, checkpoint["schema"]).to(device).eval()
    if format_name == SINGLE_FORMAT:
        config = GarmentTreeConfig.from_dict(checkpoint["model_config"])
        model = GarmentTreeNet(checkpoint["schema"], config).to(device)
        model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
        return BaselineContractAdapter(model, checkpoint["schema"]).to(device).eval()
    return _original_baseline_builder(checkpoint, device)
def compatible_load_image(
    path: str, image_size: int, device: torch.device
) -> torch.Tensor:
    if not _aspect_pad_inputs:
        return _original_load_image(path, image_size, device)
    image = Image.open(path).convert("RGB")
    side = max(image.width, image.height)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return transform(canvas).unsqueeze(0).to(device)



def compatible_write_model_yaml(
    architecture: str,
    checkpoint: dict[str, Any],
    model: nn.Module,
    adapter: Any,
    image_path: Path,
    image_size: int,
    device: torch.device,
    out_yaml: Path,
    args: Any,
) -> dict[str, Any]:
    if checkpoint.get("format") != ENSEMBLE_FORMAT:
        return _original_write_model_yaml(
            architecture,
            checkpoint,
            model,
            adapter,
            image_path,
            image_size,
            device,
            out_yaml,
            args,
        )
    raw_model = model.model if isinstance(model, BaselineContractAdapter) else model
    image = batch.load_image(str(image_path), image_size, device)
    with torch.inference_mode():
        output = raw_model(image)
    template = yaml.safe_load(Path(args.template).read_text())
    design, details = decode_ensemble_design(output, checkpoint["schema"], template)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(design, sort_keys=False, default_flow_style=False))
    return {
        "architecture": "garment_tree_ensemble",
        "checkpoint": str(args.checkpoint),
        "checkpoint_validation": checkpoint.get("validation"),
        "source": {"image": str(image_path)},
        "prediction": details,
    }

def compatible_torch_load(*args: Any, **kwargs: Any) -> Any:
    global _aspect_pad_inputs
    checkpoint = _original_torch_load(*args, **kwargs)
    if isinstance(checkpoint, dict) and checkpoint.get("format") in (
        SINGLE_FORMAT,
        ENSEMBLE_FORMAT,
    ):
        checkpoint.setdefault("args", checkpoint.get("train_args", {}))
        _aspect_pad_inputs = bool(checkpoint["args"].get("aspect_pad", False))
    return checkpoint


def main() -> None:
    batch.MODEL_DEFAULTS["garment_tree"] = {
        "checkpoint": PROJECT_ROOT / "runs" / "garment_tree_ensemble_v1" / "best.pt",
        "out_dir": (
            PROJECT_ROOT
            / "runs"
            / "garmentimage_comparisons"
            / "garment_tree_ensemble_v1"
        ),
    }
    batch.checkpoint_architecture = compatible_checkpoint_architecture
    batch.build_baseline_inference_model = compatible_baseline_builder
    batch.write_model_yaml = compatible_write_model_yaml
    batch.torch.load = compatible_torch_load
    batch.load_image = compatible_load_image
    if "--image-root" not in sys.argv:
        sys.argv.extend(("--image-root", str(DEFAULT_IMAGE_ROOT)))
    batch.main(default_model="garment_tree")


if __name__ == "__main__":
    main()