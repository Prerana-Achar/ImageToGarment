#!/usr/bin/env python3
"""End-to-end person image -> garment-only image -> TRELLIS mesh -> GarmentCode mesh -> GNN fit.

The pipeline is intentionally file-oriented so each expensive stage can be inspected or
rerun independently on the cluster.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENSEMBLE_GLOB = "runs/*garment_tree_ensemble*/best.pt"
DEFAULT_SEGMENTATION_MODEL = "mattmdjaga/segformer_b2_clothes"
GARMENT_LABELS = {
    "upper": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "scarf": 17,
}
TOPOLOGY_PATHS = ("meta.upper", "meta.wb", "meta.bottom")


@dataclass
class StageOutputs:
    source_image: str
    garment_rgba: str
    garment_rgb: str
    garment_mask: str
    trellis_glb: str | None
    trellis_obj: str | None
    garmentcode_yaml: str | None
    garmentcode_json: str | None
    garmentcode_mesh: str | None
    confidence_report: str | None
    confidence_gate_passed: bool
    gnn_manifest: str | None
    gnn_run_dir: str | None
    fitted_mesh: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input image of a person wearing a garment")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/person_garment_trellis_gnn")
    parser.add_argument("--checkpoint", type=Path, help="GarmentTree ensemble checkpoint; defaults to latest best.pt")
    parser.add_argument("--segmentation-model", default=DEFAULT_SEGMENTATION_MODEL)
    parser.add_argument(
        "--garment-labels",
        default="upper,skirt,pants,dress,belt,scarf",
        help="Comma-separated garment labels to keep; hair, shoes, skin, bags, and background are always excluded",
    )
    parser.add_argument("--mask-feather", type=float, default=1.5)
    parser.add_argument("--mask-close", type=int, default=7, help="Odd morphological close kernel; 0 disables")
    parser.add_argument("--mask-open", type=int, default=3, help="Odd morphological open kernel; 0 disables")
    parser.add_argument("--trellis-dir", type=Path, default=ROOT / "third_party/TRELLIS.2")
    parser.add_argument("--skip-trellis", action="store_true")
    parser.add_argument("--skip-garmentcode-render", action="store_true")
    parser.add_argument("--skip-gnn", action="store_true")
    parser.add_argument("--trellis-device", default="cuda")
    parser.add_argument("--ensemble-device", default="cuda")
    parser.add_argument("--gnn-device", default="cuda")
    parser.add_argument("--garmentcode-python", type=Path, default=ROOT / ".envs/garmentcode/bin/python")
    parser.add_argument("--garmentcode-dir", type=Path, default=ROOT / "GarmentCodeRC")
    parser.add_argument("--garmentcode-resolution-scale", type=float, default=3.0)
    parser.add_argument("--gnn-epochs", type=int, default=200)
    parser.add_argument("--gnn-surface-samples", type=int, default=8192)
    parser.add_argument("--gnn-learning-rate", type=float, default=1e-3)
    parser.add_argument("--gnn-max-displacement-ratio", type=float, default=0.4)
    parser.add_argument(
        "--min-root-confidence",
        type=float,
        default=0.75,
        help="Minimum confidence for active root topology choices before using GarmentCode output for GNN training",
    )
    parser.add_argument(
        "--min-active-categorical-confidence",
        type=float,
        default=0.65,
        help="Minimum confidence for active non-root categorical GarmentCode predictions",
    )
    parser.add_argument(
        "--min-active-numeric-probability",
        type=float,
        default=0.55,
        help="Minimum active probability for active numeric GarmentCode predictions",
    )
    parser.add_argument(
        "--max-active-numeric-uncertainty",
        type=float,
        default=0.35,
        help="Maximum normalized uncertainty for active numeric GarmentCode predictions",
    )
    parser.add_argument(
        "--allow-low-confidence-garmentcode",
        action="store_true",
        help="Write confidence failures but still allow the pair into GNN training",
    )
    parser.add_argument("--force", action="store_true", help="Recompute stages even if their outputs already exist")
    return parser.parse_args()


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    completed = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout, end="", flush=True)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd, output=completed.stdout)
    return completed


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._") or "image"


def latest_checkpoint(pattern: str = DEFAULT_ENSEMBLE_GLOB) -> Path:
    candidates = [path for path in ROOT.glob(pattern) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No ensemble checkpoints matched {pattern!r}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def odd_kernel(value: int) -> int:
    if value <= 0:
        return 0
    return value if value % 2 == 1 else value + 1


def refine_mask(mask: np.ndarray, close_size: int, open_size: int, feather: float) -> Image.Image:
    try:
        import cv2
    except ImportError:
        cv2 = None
    mask_u8 = (mask.astype(np.uint8) * 255)
    if cv2 is not None:
        if close_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_kernel(close_size), odd_kernel(close_size)))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        if open_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_kernel(open_size), odd_kernel(open_size)))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    image = Image.fromarray(mask_u8, mode="L")
    if feather > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=feather))
    return image


def segment_garment(args: argparse.Namespace, stage_dir: Path) -> tuple[Path, Path, Path]:
    rgba_path = stage_dir / "01_garment_only_rgba.png"
    rgb_path = stage_dir / "01_garment_only_rgb.png"
    mask_path = stage_dir / "01_garment_mask.png"
    if not args.force and rgba_path.exists() and rgb_path.exists() and mask_path.exists():
        print(f"[skip] garment segmentation exists: {rgba_path}", flush=True)
        return rgba_path, rgb_path, mask_path

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

    requested = {item.strip().lower() for item in args.garment_labels.split(",") if item.strip()}
    keep_ids = {GARMENT_LABELS[name] for name in requested if name in GARMENT_LABELS}
    if not keep_ids:
        raise ValueError(f"No valid garment labels in {args.garment_labels!r}; choices: {sorted(GARMENT_LABELS)}")

    image = Image.open(args.image).convert("RGB")
    processor = SegformerImageProcessor.from_pretrained(args.segmentation_model)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.segmentation_model).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
    logits = F.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
    pred = logits.argmax(dim=1)[0].detach().cpu().numpy()
    mask = np.isin(pred, sorted(keep_ids))
    alpha = refine_mask(mask, args.mask_close, args.mask_open, args.mask_feather)

    rgba = image.convert("RGBA")
    rgba.putalpha(alpha)
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=alpha)
    stage_dir.mkdir(parents=True, exist_ok=True)
    rgba.save(rgba_path)
    background.save(rgb_path)
    alpha.save(mask_path)
    print(f"wrote garment cutout: {rgba_path}", flush=True)
    return rgba_path, rgb_path, mask_path


def trellis_script() -> str:
    return r'''
import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
import o_voxel

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--out-glb", required=True)
parser.add_argument("--device", default="cuda")
parser.add_argument("--model", default="microsoft/TRELLIS.2-4B")
args = parser.parse_args()

device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise SystemExit("TRELLIS.2 requires CUDA for practical inference")

forest = cv2.imread("assets/hdri/forest.exr", cv2.IMREAD_UNCHANGED)
if forest is None:
    raise FileNotFoundError("assets/hdri/forest.exr")
envmap = EnvMap(torch.tensor(cv2.cvtColor(forest, cv2.COLOR_BGR2RGB), dtype=torch.float32, device=device))
_ = envmap

pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
pipeline.cuda()
image = Image.open(args.image).convert("RGBA")
mesh = pipeline.run(image)[0]
mesh.simplify(16777216)
glb = o_voxel.postprocess.to_glb(
    vertices=mesh.vertices,
    faces=mesh.faces,
    attr_volume=mesh.attrs,
    coords=mesh.coords,
    attr_layout=mesh.layout,
    voxel_size=mesh.voxel_size,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target=1000000,
    texture_size=4096,
    remesh=True,
    remesh_band=1,
    remesh_project=0,
    verbose=True,
)
Path(args.out_glb).parent.mkdir(parents=True, exist_ok=True)
glb.export(args.out_glb, extension_webp=True)
print(f"wrote TRELLIS GLB: {args.out_glb}")
'''


def run_trellis(args: argparse.Namespace, garment_rgba: Path, stage_dir: Path) -> tuple[Path, Path]:
    glb_path = stage_dir / "02_trellis_mesh.glb"
    obj_path = stage_dir / "02_trellis_mesh.obj"
    if args.skip_trellis:
        if not obj_path.exists():
            raise FileNotFoundError(f"--skip-trellis requires existing {obj_path}")
        return glb_path, obj_path
    if not args.force and obj_path.exists():
        print(f"[skip] TRELLIS OBJ exists: {obj_path}", flush=True)
        return glb_path, obj_path
    if not args.trellis_dir.is_dir():
        raise FileNotFoundError(f"TRELLIS.2 checkout not found: {args.trellis_dir}")
    with tempfile.NamedTemporaryFile("w", suffix="_run_trellis2.py", delete=False, encoding="utf-8") as handle:
        handle.write(trellis_script())
        helper = Path(handle.name)
    try:
        run([sys.executable, helper, "--image", garment_rgba, "--out-glb", glb_path, "--device", args.trellis_device], cwd=args.trellis_dir)
    finally:
        helper.unlink(missing_ok=True)
    convert_mesh(glb_path, obj_path)
    return glb_path, obj_path


def convert_mesh(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    import trimesh
    mesh = trimesh.load(source, force="mesh")
    if mesh.is_empty:
        raise RuntimeError(f"trimesh loaded an empty mesh from {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(destination)
    print(f"converted mesh: {destination}", flush=True)


def run_ensemble(args: argparse.Namespace, garment_rgb: Path, stage_dir: Path) -> tuple[Path, Path, Path]:
    checkpoint = args.checkpoint or latest_checkpoint()
    yaml_path = stage_dir / "03_garmentcode_design.yaml"
    json_path = stage_dir / "03_garmentcode_prediction.json"
    mesh_path = stage_dir / "04_garmentcode_mesh.obj"
    if args.force or not yaml_path.exists() or not json_path.exists():
        run([
            sys.executable,
            ROOT / "infer_garment_tree.py",
            "--checkpoint", checkpoint,
            "--image", garment_rgb,
            "--yaml-out", yaml_path,
            "--json-out", json_path,
            "--device", args.ensemble_device,
        ], cwd=ROOT)
    else:
        print(f"[skip] ensemble prediction exists: {yaml_path}", flush=True)
    if args.skip_garmentcode_render:
        if not mesh_path.exists():
            raise FileNotFoundError(f"--skip-garmentcode-render requires existing {mesh_path}")
        return yaml_path, json_path, mesh_path
    if args.force or not mesh_path.exists():
        render_and_find_mesh(args, yaml_path, mesh_path, stage_dir)
    else:
        print(f"[skip] GarmentCode mesh exists: {mesh_path}", flush=True)
    return yaml_path, json_path, mesh_path


def render_and_find_mesh(args: argparse.Namespace, yaml_path: Path, mesh_path: Path, stage_dir: Path) -> None:
    python_exe = args.garmentcode_python if args.garmentcode_python.exists() else Path(sys.executable)
    render_dir = stage_dir / "04_garmentcode_render"
    completed = run([
        python_exe,
        ROOT / "render_garmentcode.py",
        "--design", yaml_path,
        "--name", "garmentcode_template",
        "--out-dir", render_dir,
        "--garmentcode-dir", args.garmentcode_dir,
        "--resolution-scale", str(args.garmentcode_resolution_scale),
    ], cwd=ROOT)
    match = re.search(r"wrote simulated mesh:\s*(.+)", completed.stdout)
    candidates: list[Path] = []
    if match:
        candidates.append(Path(match.group(1).strip()))
    candidates.extend(render_dir.rglob("*.obj"))
    candidates.extend(render_dir.rglob("*.ply"))
    candidates.extend(render_dir.rglob("*.npz"))
    for candidate in candidates:
        if candidate.exists():
            if candidate.resolve() != mesh_path.resolve():
                convert_or_copy_mesh(candidate, mesh_path)
            return
    raise RuntimeError(f"Could not find a GarmentCode mesh under {render_dir}")


def convert_or_copy_mesh(source: Path, destination: Path) -> None:
    suffix = source.suffix.lower()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".obj":
        shutil.copy2(source, destination)
    elif suffix in {".ply", ".glb", ".gltf", ".stl"}:
        convert_mesh(source, destination)
    elif suffix == ".npz":
        shutil.copy2(source, destination.with_suffix(".npz"))
        destination = destination.with_suffix(".npz")
    else:
        raise ValueError(f"Unsupported GarmentCode mesh format: {source}")
    print(f"wrote GarmentCode template mesh: {destination}", flush=True)

def evaluate_garmentcode_confidence(args: argparse.Namespace, json_path: Path, stage_dir: Path) -> tuple[bool, Path]:
    """Return whether the ensemble prediction is trusted enough for GNN training."""
    report_path = stage_dir / "03_garmentcode_confidence_report.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    prediction = payload.get("prediction", {})
    categorical = prediction.get("categorical", {})
    numeric = prediction.get("numeric", {})
    failures: list[dict[str, object]] = []

    active_categorical_confidences: list[float] = []
    root_confidences: list[float] = []
    for path, item in categorical.items():
        if not item.get("active", False):
            continue
        confidence = float(item.get("confidence", 0.0))
        active_categorical_confidences.append(confidence)
        if path in TOPOLOGY_PATHS:
            root_confidences.append(confidence)
            threshold = args.min_root_confidence
            reason = "root_topology_confidence"
        else:
            threshold = args.min_active_categorical_confidence
            reason = "categorical_confidence"
        if confidence < threshold:
            failures.append(
                {
                    "path": path,
                    "reason": reason,
                    "value": confidence,
                    "threshold": threshold,
                    "predicted_value": item.get("predicted_value", item.get("value")),
                }
            )

    active_numeric_probabilities: list[float] = []
    active_numeric_uncertainties: list[float] = []
    for path, item in numeric.items():
        if not item.get("active", False):
            continue
        probability = float(item.get("active_probability", 0.0))
        uncertainty = float(item.get("uncertainty_normalized", 0.0))
        active_numeric_probabilities.append(probability)
        active_numeric_uncertainties.append(uncertainty)
        if probability < args.min_active_numeric_probability:
            failures.append(
                {
                    "path": path,
                    "reason": "numeric_active_probability",
                    "value": probability,
                    "threshold": args.min_active_numeric_probability,
                }
            )
        if uncertainty > args.max_active_numeric_uncertainty:
            failures.append(
                {
                    "path": path,
                    "reason": "numeric_uncertainty",
                    "value": uncertainty,
                    "threshold": args.max_active_numeric_uncertainty,
                }
            )

    passed = not failures
    if not root_confidences:
        passed = False
        failures.append(
            {
                "path": ",".join(TOPOLOGY_PATHS),
                "reason": "missing_active_root_topology",
                "value": None,
                "threshold": args.min_root_confidence,
            }
        )

    report = {
        "format": "GarmentCodeConfidenceGate/v1",
        "passed": passed,
        "used_for_gnn_training": bool(passed or args.allow_low_confidence_garmentcode),
        "allow_low_confidence_garmentcode": bool(args.allow_low_confidence_garmentcode),
        "thresholds": {
            "min_root_confidence": args.min_root_confidence,
            "min_active_categorical_confidence": args.min_active_categorical_confidence,
            "min_active_numeric_probability": args.min_active_numeric_probability,
            "max_active_numeric_uncertainty": args.max_active_numeric_uncertainty,
        },
        "summary": {
            "root_min_confidence": min(root_confidences) if root_confidences else None,
            "active_categorical_min_confidence": min(active_categorical_confidences) if active_categorical_confidences else None,
            "active_numeric_min_probability": min(active_numeric_probabilities) if active_numeric_probabilities else None,
            "active_numeric_max_uncertainty": max(active_numeric_uncertainties) if active_numeric_uncertainties else None,
            "active_categorical_count": len(active_categorical_confidences),
            "active_numeric_count": len(active_numeric_probabilities),
        },
        "failures": failures,
        "source_prediction": str(json_path),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if passed:
        print(f"GarmentCode confidence gate passed: {report_path}", flush=True)
    else:
        print(f"GarmentCode confidence gate failed: {report_path}", flush=True)
        for failure in failures[:20]:
            print(
                f"  - {failure['reason']} {failure['path']}: "
                f"{failure['value']} vs {failure['threshold']}",
                flush=True,
            )
    return bool(passed or args.allow_low_confidence_garmentcode), report_path


def write_rejected_gnn_manifest(stage_dir: Path, confidence_report: Path) -> tuple[Path, Path, Path]:
    manifest_path = stage_dir / "05_gnn_pairs.json"
    gnn_dir = stage_dir / "05_gnn_fit"
    fitted_mesh = gnn_dir / "previews" / "image_pair.obj"
    manifest = {
        "pairs": [],
        "rejected": True,
        "reason": "garmentcode_confidence_gate_failed",
        "confidence_report": str(confidence_report.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("skipping GNN training because GarmentCode confidence gate failed", flush=True)
    return manifest_path, gnn_dir, fitted_mesh

def run_gnn(args: argparse.Namespace, template_mesh: Path, target_mesh: Path, stage_dir: Path) -> tuple[Path, Path, Path]:
    manifest_path = stage_dir / "05_gnn_pairs.json"
    gnn_dir = stage_dir / "05_gnn_fit"
    fitted_mesh = gnn_dir / "previews" / "image_pair.obj"
    manifest = {
        "pairs": [
            {
                "id": "image_pair",
                "split": "train",
                "template": str(template_mesh.resolve()),
                "target": str(target_mesh.resolve()),
            }
        ]
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.skip_gnn:
        return manifest_path, gnn_dir, fitted_mesh
    if args.force or not (gnn_dir / "best.pt").exists():
        cmd = [
            sys.executable,
            ROOT / "train_mesh_fitting_gnn.py",
            "--manifest", manifest_path,
            "--out-dir", gnn_dir,
            "--epochs", str(args.gnn_epochs),
            "--surface-samples", str(args.gnn_surface_samples),
            "--learning-rate", str(args.gnn_learning_rate),
            "--max-displacement-ratio", str(args.gnn_max_displacement_ratio),
            "--device", args.gnn_device,
            "--overwrite",
            "--preview-count", "1",
        ]
        run(cmd, cwd=ROOT)
    else:
        print(f"[skip] GNN fit exists: {gnn_dir / 'best.pt'}", flush=True)
    if not fitted_mesh.exists():
        run([
            sys.executable,
            ROOT / "infer_mesh_fitting_gnn.py",
            "--checkpoint", gnn_dir / "best.pt",
            "--template", template_mesh,
            "--out-obj", fitted_mesh,
            "--device", args.gnn_device,
        ], cwd=ROOT)
    return manifest_path, gnn_dir, fitted_mesh


def main() -> None:
    args = parse_args()
    image = args.image.expanduser().resolve()
    if not image.is_file():
        raise SystemExit(f"Input image not found: {image}")
    args.image = image
    args.out_dir = args.out_dir.expanduser().resolve()
    stage_dir = args.out_dir / safe_stem(image)
    stage_dir.mkdir(parents=True, exist_ok=True)

    garment_rgba, garment_rgb, garment_mask = segment_garment(args, stage_dir)
    trellis_glb, trellis_obj = run_trellis(args, garment_rgba, stage_dir)
    yaml_path, json_path, garmentcode_mesh = run_ensemble(args, garment_rgb, stage_dir)
    confidence_passed, confidence_report = evaluate_garmentcode_confidence(args, json_path, stage_dir)
    if confidence_passed:
        manifest_path, gnn_dir, fitted_mesh = run_gnn(args, garmentcode_mesh, trellis_obj, stage_dir)
    else:
        manifest_path, gnn_dir, fitted_mesh = write_rejected_gnn_manifest(stage_dir, confidence_report)

    outputs = StageOutputs(
        source_image=str(image),
        garment_rgba=str(garment_rgba),
        garment_rgb=str(garment_rgb),
        garment_mask=str(garment_mask),
        trellis_glb=str(trellis_glb) if trellis_glb else None,
        trellis_obj=str(trellis_obj) if trellis_obj else None,
        garmentcode_yaml=str(yaml_path),
        garmentcode_json=str(json_path),
        garmentcode_mesh=str(garmentcode_mesh),
        confidence_report=str(confidence_report),
        confidence_gate_passed=confidence_passed,
        gnn_manifest=str(manifest_path),
        gnn_run_dir=str(gnn_dir),
        fitted_mesh=str(fitted_mesh),
    )
    summary = stage_dir / "pipeline_outputs.json"
    summary.write_text(json.dumps(asdict(outputs), indent=2) + "\n", encoding="utf-8")
    print(f"wrote pipeline summary: {summary}", flush=True)


if __name__ == "__main__":
    main()