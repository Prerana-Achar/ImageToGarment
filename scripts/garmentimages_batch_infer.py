#!/usr/bin/env python
"""Batch inference/render comparison for images under GarmentImages."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dinov2_pipeline import (
    DEFAULT_MODEL_SCHEMA,
    ROOT,
    build_modelpy_inference_model,
    checkpoint_architecture,
    decode_modelpy_design,
    forward_modelpy,
    modelpy_prediction_json,
    resolve_device,
)
from infer_dinov2 import (
    build_model as build_baseline_inference_model,
    decode_prediction as decode_baseline_prediction,
    load_image,
    write_yaml as write_baseline_yaml,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODEL_DEFAULTS = {
    "dinov2_vits14": {
        "checkpoint": ROOT / "runs" / "dinov2_vits14" / "best.pt",
        "out_dir": ROOT / "runs" / "garmentimage_comparisons" / "dinov2_vits14",
    },
    "baseline_all_images": {
        "checkpoint": ROOT / "runs" / "dinov2_baseline_all_images" / "best.pt",
        "out_dir": ROOT / "runs" / "garmentimage_comparisons" / "baseline_all_images",
    },
    "modelpy_all_images": {
        "checkpoint": ROOT / "runs" / "dinov2_modelpy_all_images" / "best.pt",
        "out_dir": ROOT / "runs" / "garmentimage_comparisons" / "modelpy_all_images",
    },
}


def image_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def safe_stem_for_path(image_root: Path, image_path: Path) -> str:
    relative = image_path.relative_to(image_root)
    parts = [*relative.parent.parts, image_path.stem]
    return "_".join(part for part in parts if part).replace(" ", "_")


def relative_output_path(base: Path, image_root: Path, image_path: Path, suffix: str) -> Path:
    relative = image_path.relative_to(image_root)
    return base / relative.parent / f"{image_path.stem}{suffix}"


def write_model_yaml(
    architecture: str,
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
    adapter: Any,
    image_path: Path,
    image_size: int,
    device: torch.device,
    out_yaml: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    image = load_image(str(image_path), image_size, device)
    source = {"image": str(image_path)}
    with torch.inference_mode():
        if architecture == "baseline":
            pred_reg, pred_logits = model(image)
            prediction = decode_baseline_prediction(pred_reg, pred_logits, checkpoint["schema"])
            result = {
                "architecture": architecture,
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_best_val": checkpoint.get("best_val"),
                "source": source,
                "prediction": prediction,
            }
            write_baseline_yaml(result, str(args.template), args.source_mode, str(out_yaml))
            return result

        outputs = forward_modelpy(model, image)
        prediction = modelpy_prediction_json(model, outputs, adapter)
        design = decode_modelpy_design(model, outputs, adapter)
        out_yaml.parent.mkdir(parents=True, exist_ok=True)
        with open(out_yaml, "w") as handle:
            yaml.safe_dump(design, handle, sort_keys=False, default_flow_style=False)
        return {
            "architecture": architecture,
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_best_val": checkpoint.get("best_val"),
            "source": source,
            "prediction": prediction,
        }


def render_front_preview(yaml_path: Path, render_name: str, args: argparse.Namespace) -> Path:
    # Do not resolve the interpreter symlink: resolving venv/bin/python to the
    # system executable bypasses the venv and can mix binary package versions.
    render_python = args.render_python.expanduser()
    garmentcode_dir = args.garmentcode_dir.expanduser().resolve()
    render_script = ROOT / "render_garmentcode.py"
    if not render_python.is_file():
        raise FileNotFoundError(f"GarmentCode Python interpreter not found: {render_python}")
    if not (garmentcode_dir / "assets" / "garment_programs").is_dir():
        raise FileNotFoundError(f"GarmentCode checkout not found: {garmentcode_dir}")

    with tempfile.TemporaryDirectory(prefix="gc_render_") as tmp:
        temp_out = Path(tmp)
        command = [
            str(render_python),
            str(render_script),
            "--design",
            str(yaml_path.resolve()),
            "--name",
            render_name,
            "--out-dir",
            str(temp_out),
            "--resolution-scale",
            str(args.render_resolution_scale),
            "--garmentcode-dir",
            str(garmentcode_dir),
        ]
        for field in ("upper", "wb", "bottom"):
            value = getattr(args, f"render_{field}")
            if value is not None:
                command.extend([f"--override-{field}", value])
        if args.max_sim_steps is not None:
            command.extend(["--max-sim-steps", str(args.max_sim_steps)])
        if args.max_sim_time is not None:
            command.extend(["--max-sim-time", str(args.max_sim_time)])

        subprocess.run(command, cwd=ROOT, check=True)
        preview = (
            temp_out
            / "simulation"
            / render_name
            / f"{render_name}_render_front_preview.png"
        )
        if not preview.is_file():
            raise FileNotFoundError(f"Expected front preview was not produced: {preview}")
        persisted = Path(args._current_front_preview)
        persisted.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(preview, persisted)
        return persisted


def fit_image(image: Image.Image, box: tuple[int, int], background: tuple[int, int, int]) -> Image.Image:
    target_w, target_h = box
    image = image.convert("RGB")
    scale = min(target_w / image.width, target_h / image.height)
    resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    canvas = Image.new("RGB", (target_w, target_h), background)
    x = (target_w - resized.width) // 2
    y = (target_h - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def make_side_by_side(input_path: Path, preview_path: Path, out_path: Path) -> None:
    panel = (512, 512)
    label_h = 44
    gap = 18
    canvas = Image.new("RGB", (panel[0] * 2 + gap, panel[1] + label_h), (245, 245, 245))
    left = fit_image(Image.open(input_path), panel, (255, 255, 255))
    right = fit_image(Image.open(preview_path), panel, (255, 255, 255))
    canvas.paste(left, (0, label_h))
    canvas.paste(right, (panel[0] + gap, label_h))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text((12, 12), "input image", fill=(30, 30, 30), font=font)
    draw.text((panel[0] + gap + 12, 12), "GarmentCode front preview", fill=(30, 30, 30), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def parse_args(default_model: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model-key", choices=sorted(MODEL_DEFAULTS), default=default_model)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--image-root", type=Path, default=ROOT / "GarmentImages")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--architecture", choices=("auto", "baseline", "modelpy"), default="auto")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--template", type=Path, default=DEFAULT_MODEL_SCHEMA)
    parser.add_argument("--source-mode", choices=("split", "wholebody", "all"), default="split")
    parser.add_argument("--model-schema", type=Path)
    parser.add_argument("--dinov2-dir", type=Path)
    parser.add_argument("--limit", type=int, help="process only the first N images")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--render-resolution-scale", type=float, default=3.0)
    parser.add_argument(
        "--render-python",
        type=Path,
        default=PROJECT_ROOT / "venv" / "bin" / "python",
        help="Python interpreter with Warp, PyRender, and GarmentCode dependencies",
    )
    parser.add_argument(
        "--garmentcode-dir",
        type=Path,
        default=PROJECT_ROOT / "GarmentCodeRC",
        help="GarmentCodeRC checkout containing assets and pygarment",
    )
    parser.add_argument("--max-sim-steps", type=int)
    parser.add_argument("--max-sim-time", type=int)
    parser.add_argument("--render-upper", choices=("none", "FittedShirt", "Shirt"))
    parser.add_argument("--render-wb", choices=("none", "StraightWB", "FittedWB"))
    parser.add_argument(
        "--render-bottom",
        choices=("none", "SkirtCircle", "AsymmSkirtCircle", "GodetSkirt", "PencilSkirt", "Skirt2", "Pants"),
    )
    return parser.parse_args()


def fill_defaults(args: argparse.Namespace) -> None:
    if args.model_key is None and (args.checkpoint is None or args.out_dir is None):
        raise SystemExit("Provide --model-key, or provide both --checkpoint and --out-dir")
    if args.model_key is not None:
        defaults = MODEL_DEFAULTS[args.model_key]
        args.checkpoint = args.checkpoint or defaults["checkpoint"]
        args.out_dir = args.out_dir or defaults["out_dir"]
    assert args.checkpoint is not None
    assert args.out_dir is not None


def main(default_model: str | None = None) -> None:
    args = parse_args(default_model)
    fill_defaults(args)
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

    image_root = args.image_root.expanduser().resolve()
    if not image_root.is_dir():
        raise SystemExit(f"Image root not found: {image_root}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    out_dir = args.out_dir.expanduser().resolve()
    yaml_dir = out_dir / "yaml"
    preview_dir = out_dir / "front_preview"
    compare_dir = out_dir / "side_by_side"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    detected = checkpoint_architecture(checkpoint)
    if args.architecture != "auto" and args.architecture != detected:
        raise SystemExit(
            f"Checkpoint architecture is {detected!r}, not requested {args.architecture!r}"
        )

    if detected == "baseline":
        model = build_baseline_inference_model(checkpoint, device)
        adapter = None
    else:
        ns = SimpleNamespace(model_schema=str(args.model_schema) if args.model_schema else None,
                             dinov2_dir=str(args.dinov2_dir) if args.dinov2_dir else None)
        model, adapter = build_modelpy_inference_model(checkpoint, ns, device)
    image_size = int(checkpoint["args"].get("image_size", 224))

    paths = image_paths(image_root)
    if args.limit is not None:
        paths = paths[: args.limit]
    print(f"model={args.model_key or detected} architecture={detected} images={len(paths)}")
    print(f"outputs={out_dir}")

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=["image", "status", "yaml", "front_preview", "side_by_side", "error"],
        )
        writer.writeheader()

        for index, image_path in enumerate(paths, start=1):
            rel = image_path.relative_to(image_root)
            yaml_path = relative_output_path(yaml_dir, image_root, image_path, ".yaml")
            preview_path = relative_output_path(preview_dir, image_root, image_path, "_front_preview.png")
            compare_path = relative_output_path(compare_dir, image_root, image_path, "_compare.png")
            render_name = safe_stem_for_path(image_root, image_path)
            args._current_front_preview = str(preview_path)
            print(f"[{index}/{len(paths)}] {rel}", flush=True)

            if args.skip_existing and yaml_path.is_file() and preview_path.is_file() and compare_path.is_file():
                writer.writerow({
                    "image": str(rel),
                    "status": "skipped",
                    "yaml": str(yaml_path),
                    "front_preview": str(preview_path),
                    "side_by_side": str(compare_path),
                    "error": "",
                })
                summary_file.flush()
                continue

            try:
                write_model_yaml(
                    detected,
                    checkpoint,
                    model,
                    adapter,
                    image_path,
                    image_size,
                    device,
                    yaml_path,
                    args,
                )
                render_front_preview(yaml_path, render_name, args)
                make_side_by_side(image_path, preview_path, compare_path)
                writer.writerow({
                    "image": str(rel),
                    "status": "ok",
                    "yaml": str(yaml_path),
                    "front_preview": str(preview_path),
                    "side_by_side": str(compare_path),
                    "error": "",
                })
            except Exception as exc:
                writer.writerow({
                    "image": str(rel),
                    "status": "failed",
                    "yaml": str(yaml_path) if yaml_path.is_file() else "",
                    "front_preview": str(preview_path) if preview_path.is_file() else "",
                    "side_by_side": "",
                    "error": str(exc),
                })
                print(f"failed {rel}: {exc}", flush=True)
            summary_file.flush()

    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
