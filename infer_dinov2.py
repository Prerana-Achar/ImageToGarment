#!/usr/bin/env python
"""Run inference with a trained DINOv2 GarmentCode checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torchvision import transforms

from train_dinov2 import GarmentDinoModel

TOP_PREFIXES = ("wholebody_garment", "upperbody_garment", "lowerbody_garment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/dinov2_vits14/best.pt")
    parser.add_argument("--prepared-dir", default="prepared_v2")
    parser.add_argument("--gid", help="garment id from prepared_v2, e.g. v2_1327")
    parser.add_argument("--image", help="path to one RGB image")
    parser.add_argument("--frame", default="0", help="frame to use when --gid is given")
    parser.add_argument("--view", type=int, default=0, choices=(0, 1, 2, 3))
    parser.add_argument(
        "--out",
        help="write prediction JSON here; defaults to <checkpoint-run>/inference/infer_<gid-or-image-stem>.json",
    )
    parser.add_argument(
        "--yaml-out",
        help="write GarmentCode-style design YAML here; defaults to <checkpoint-run>/inference/infer_<gid-or-image-stem>.yaml",
    )
    parser.add_argument(
        "--template",
        default="verify_dump/demo_design_v2_1327.yaml",
        help="GarmentCode design YAML/template carrying v/range/type fields",
    )
    parser.add_argument(
        "--source-mode",
        choices=("split", "wholebody", "all"),
        default="split",
        help="which top-level prediction namespace to write into the single design tree",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def load_checkpoint(path: str, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def path_from_gid(prepared_dir: str, gid: str, frame: str, view: int) -> str:
    with open(Path(prepared_dir) / "images.json") as f:
        images = json.load(f)
    if gid not in images:
        raise SystemExit(f"gid {gid!r} not found in {prepared_dir}/images.json")

    frames = images[gid]["frames"]
    frame_key = frame if frame in frames else sorted(frames, key=int)[0]
    views = frames[frame_key]
    path = views[view] if view < len(views) else None
    if path is None:
        present = [p for p in views if p is not None]
        if not present:
            raise SystemExit(f"gid {gid!r} frame {frame_key!r} has no image paths")
        path = present[0]
    return path


def load_image(path: str, image_size: int, device: torch.device) -> torch.Tensor:
    transform = image_transform(image_size)
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def build_model(ckpt: dict[str, Any], device: torch.device) -> GarmentDinoModel:
    train_args = ckpt["args"]
    schema = ckpt["schema"]
    vocab_sizes = [len(vocab) for vocab in schema["cat_vocab"].values()]
    model = GarmentDinoModel(
        backbone_name=train_args.get("backbone", "dinov2_vits14"),
        reg_dim=int(schema["n_cont"]) + int(schema["n_const"]),
        cat_vocab_sizes=vocab_sizes,
        hidden_dim=int(train_args.get("hidden_dim", 512)),
        dropout=float(train_args.get("dropout", 0.1)),
        freeze_backbone=not bool(train_args.get("unfreeze_backbone", False)),
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def decode_prediction(
    pred_reg: torch.Tensor,
    pred_logits: torch.Tensor,
    schema: dict[str, Any],
) -> dict[str, Any]:
    pred_reg = pred_reg.squeeze(0).detach().cpu().float().clamp(0.0, 1.0)
    pred_logits = pred_logits.squeeze(0).detach().cpu().float()

    cont_paths = list(schema["cont_slots"].keys())
    const_paths = list(schema["const_slots"].keys())
    n_cont = len(cont_paths)

    cont = {
        path: float(pred_reg[idx])
        for path, idx in schema["cont_slots"].items()
    }

    const_norm = {}
    const_raw = {}
    for path, idx in schema["const_slots"].items():
        value = float(pred_reg[n_cont + idx])
        lo, hi = schema["const_ranges"][path]
        const_norm[path] = value
        const_raw[path] = float(lo + value * (hi - lo))

    cats = {}
    offset = 0
    for path, vocab in schema["cat_vocab"].items():
        logits = pred_logits[offset:offset + len(vocab)]
        probs = logits.softmax(dim=0)
        pred_idx = int(probs.argmax())
        cats[path] = {
            "value": vocab[pred_idx],
            "index": pred_idx,
            "confidence": float(probs[pred_idx]),
        }
        offset += len(vocab)

    return {
        "continuous_norm": cont,
        "constants_norm": const_norm,
        "constants_raw": const_raw,
        "categoricals": cats,
    }


def strip_top_prefix(path: str) -> str:
    parts = path.split(".")
    if parts and parts[0] in TOP_PREFIXES:
        parts = parts[1:]
    return ".".join(parts)


def top_prefix(path: str) -> str | None:
    first = path.split(".", 1)[0]
    return first if first in TOP_PREFIXES else None


def allowed_path(path: str, source_mode: str) -> bool:
    prefix = top_prefix(path)
    if source_mode == "all" or prefix is None:
        return True
    if source_mode == "wholebody":
        return prefix == "wholebody_garment"
    return prefix in {"upperbody_garment", "lowerbody_garment"}


def get_spec(design: dict[str, Any], dotted_path: str) -> dict[str, Any]:
    node = design
    for part in dotted_path.split("."):
        node = node[part]
    if not isinstance(node, dict) or "v" not in node:
        raise KeyError(dotted_path)
    return node


def cast_value(value: Any, spec: dict[str, Any]) -> Any:
    typ = spec.get("type")
    if value is None:
        return None
    if typ == "int":
        return int(round(float(value)))
    if typ == "float":
        return float(value)
    if typ == "bool":
        return bool(value)
    return value


def numeric_range(spec: dict[str, Any]) -> tuple[float, float]:
    values = spec["range"]
    lo, hi = values[0], values[-1]
    return float(lo), float(hi)


def set_value(design: dict[str, Any], path: str, value: Any) -> None:
    spec = get_spec(design, path)
    spec["v"] = cast_value(value, spec)


def prediction_to_design_yaml(
    result: dict[str, Any],
    template_path: str,
    source_mode: str,
) -> tuple[dict[str, Any], list[str]]:
    with open(template_path) as f:
        template = yaml.safe_load(f)

    design = copy.deepcopy(template["design"])
    pred = result["prediction"]
    warnings = []

    for full_path, norm_value in pred["continuous_norm"].items():
        if not allowed_path(full_path, source_mode):
            continue
        path = strip_top_prefix(full_path)
        try:
            spec = get_spec(design, path)
            lo, hi = numeric_range(spec)
            raw = lo + float(norm_value) * (hi - lo)
            set_value(design, path, raw)
        except Exception as exc:
            warnings.append(f"{full_path}: {exc}")

    for full_path, raw_value in pred["constants_raw"].items():
        if not allowed_path(full_path, source_mode):
            continue
        path = strip_top_prefix(full_path)
        try:
            set_value(design, path, raw_value)
        except Exception as exc:
            warnings.append(f"{full_path}: {exc}")

    for full_path, item in pred["categoricals"].items():
        if not allowed_path(full_path, source_mode):
            continue
        path = strip_top_prefix(full_path)
        try:
            set_value(design, path, item["value"])
        except Exception as exc:
            warnings.append(f"{full_path}: {exc}")

    return {"design": design}, warnings


def write_yaml(result: dict[str, Any], template_path: str, source_mode: str, out_path: str) -> None:
    design_yaml, warnings = prediction_to_design_yaml(result, template_path, source_mode)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(design_yaml, f, sort_keys=False, default_flow_style=False)
    print(f"wrote {out_path}")
    if warnings:
        print(f"warnings: {len(warnings)} YAML paths were not written")
        for warning in warnings[:20]:
            print(f"  {warning}")


def default_output_stem(args: argparse.Namespace) -> str:
    if args.gid:
        return f"infer_{args.gid}"
    image_stem = Path(args.image).stem
    return f"infer_{image_stem}"


def default_output_dir(args: argparse.Namespace) -> Path:
    return Path(args.checkpoint).parent / "inference"


def main() -> None:
    args = parse_args()
    if bool(args.gid) == bool(args.image):
        raise SystemExit("Provide exactly one of --gid or --image")

    root = Path(__file__).resolve().parent
    os.environ.setdefault("TORCH_HOME", str(root / ".cache" / "torch"))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt = load_checkpoint(args.checkpoint, device)
    model = build_model(ckpt, device)
    image_size = int(ckpt["args"].get("image_size", 224))

    image_path = args.image
    source = {"image": args.image}
    if args.gid:
        image_path = path_from_gid(args.prepared_dir, args.gid, args.frame, args.view)
        source = {"gid": args.gid, "frame": args.frame, "view": args.view, "image": image_path}

    image = load_image(image_path, image_size, device)
    with torch.inference_mode():
        pred_reg, pred_logits = model(image)

    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_val": ckpt.get("best_val"),
        "source": source,
        "prediction": decode_prediction(pred_reg, pred_logits, ckpt["schema"]),
    }

    out_dir = default_output_dir(args)
    out_json = args.out or str(out_dir / f"{default_output_stem(args)}.json")
    out_yaml = args.yaml_out or str(out_dir / f"{default_output_stem(args)}.yaml")

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out_json}")

    write_yaml(result, args.template, args.source_mode, out_yaml)


if __name__ == "__main__":
    main()
