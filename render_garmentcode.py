#!/usr/bin/env python
"""Generate, drape, and render a GarmentCode design YAML without the GUI."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_GARMENTCODE_DIR = ROOT / "third_party" / "GarmentCode"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "garmentcode_reconstructions"
TOPOLOGY_FIELDS = {
    "upper": ("FittedShirt", "Shirt"),
    "wb": ("StraightWB", "FittedWB"),
    "bottom": ("SkirtCircle", "AsymmSkirtCircle", "GodetSkirt", "PencilSkirt", "Skirt2", "Pants"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True, help="GarmentCode design YAML from inference")
    parser.add_argument("--name", help="output garment name; defaults to the YAML stem")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--garmentcode-dir", default=str(DEFAULT_GARMENTCODE_DIR))
    parser.add_argument("--body", default="mean_all", help="body asset stem")
    parser.add_argument(
        "--sim-config",
        help="simulation YAML; defaults to GarmentCode assets/Sim_props/default_sim_props.yaml",
    )
    parser.add_argument("--max-sim-steps", type=int, help="override the simulation step cap")
    parser.add_argument("--max-sim-time", type=int, help="override the simulation timeout in seconds")
    parser.add_argument(
        "--resolution-scale",
        type=float,
        default=3.0,
        help="override mesh edge length in cm; larger values produce a faster, coarser drape",
    )
    parser.add_argument("--pattern-only", action="store_true", help="generate the pattern without draping")
    parser.add_argument(
        "--override-upper",
        choices=("none", *TOPOLOGY_FIELDS["upper"]),
        help="replace the predicted upper-garment class; 'none' removes it",
    )
    parser.add_argument(
        "--override-wb",
        choices=("none", *TOPOLOGY_FIELDS["wb"]),
        help="replace the predicted waistband class; 'none' removes it",
    )
    parser.add_argument(
        "--override-bottom",
        choices=("none", *TOPOLOGY_FIELDS["bottom"]),
        help="replace the predicted bottom-garment class; 'none' removes it",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not name:
        raise ValueError("The output garment name is empty after sanitization")
    return name


def configure_environment(garmentcode_dir: Path) -> None:
    cache_dir = ROOT / ".cache"
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    sys.path.insert(0, str(garmentcode_dir))
    os.chdir(garmentcode_dir)


def write_system_config(garmentcode_dir: Path, simulation_dir: Path) -> None:
    config = {
        "output": str(simulation_dir),
        "datasets_path": "",
        "datasets_sim": "",
        "sim_configs_path": str(garmentcode_dir / "assets" / "Sim_props"),
        "bodies_default_path": str(garmentcode_dir / "assets" / "bodies"),
        "body_samples_path": "",
    }
    with open(garmentcode_dir / "system.json", "w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def white_background_preview(source: Path) -> Path:
    from PIL import Image

    destination = source.with_name(f"{source.stem}_preview.png")
    image = Image.open(source).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image)
    background.convert("RGB").save(destination)
    return destination


def apply_topology_overrides(design_document: dict, args: argparse.Namespace) -> None:
    meta = design_document["design"].get("meta")
    if not isinstance(meta, dict):
        raise ValueError("The design must contain a 'design.meta' mapping")

    for field in TOPOLOGY_FIELDS:
        value = getattr(args, f"override_{field}")
        if value is None:
            continue
        if field not in meta or not isinstance(meta[field], dict):
            raise ValueError(f"The design must contain a 'design.meta.{field}' mapping")
        meta[field]["v"] = None if value == "none" else value
        print(f"overriding {field} topology: {meta[field]['v']}", flush=True)


def main() -> None:
    args = parse_args()
    design_path = Path(args.design).expanduser().resolve()
    garmentcode_dir = Path(args.garmentcode_dir).expanduser().resolve()
    output_dir = Path(args.out_dir).expanduser().resolve()
    if not design_path.is_file():
        raise SystemExit(f"Design YAML not found: {design_path}")
    if not (garmentcode_dir / "assets" / "garment_programs").is_dir():
        raise SystemExit(f"GarmentCode installation not found: {garmentcode_dir}")

    name = safe_name(args.name or design_path.stem)
    body_path = garmentcode_dir / "assets" / "bodies" / f"{args.body}.yaml"
    sim_config = (
        Path(args.sim_config).expanduser().resolve()
        if args.sim_config
        else garmentcode_dir / "assets" / "Sim_props" / "default_sim_props.yaml"
    )
    if not body_path.is_file():
        raise SystemExit(f"Body measurements not found: {body_path}")
    if not sim_config.is_file():
        raise SystemExit(f"Simulation config not found: {sim_config}")

    configure_environment(garmentcode_dir)
    from assets.bodies.body_params import BodyParameters
    from assets.garment_programs.meta_garment import MetaGarment

    with open(design_path) as handle:
        design_document = yaml.safe_load(handle)
    if not isinstance(design_document, dict) or "design" not in design_document:
        raise ValueError(f"{design_path} must contain a top-level 'design' mapping")
    apply_topology_overrides(design_document, args)

    body = BodyParameters(str(body_path))
    garment = MetaGarment(name, body, design_document["design"])
    pattern = garment.assembly()
    if garment.is_self_intersecting():
        print("warning: the assembled pattern has initial self-intersections", flush=True)

    patterns_dir = output_dir / "patterns"
    patterns_dir.mkdir(parents=True, exist_ok=True)
    pattern_dir = Path(
        pattern.serialize(
            patterns_dir,
            to_subfolder=True,
            with_3d=True,
            with_text=False,
            view_ids=False,
        )
    )
    body.save(pattern_dir)
    with open(pattern_dir / "design_params.yaml", "w") as handle:
        yaml.safe_dump(design_document, handle, sort_keys=False, default_flow_style=False)
    specification = pattern_dir / f"{name}_specification.json"
    print(f"wrote pattern: {specification}", flush=True)

    if args.pattern_only:
        print(f"wrote positioned-panel preview: {pattern_dir / f'{name}_3d_pattern.png'}")
        return

    from pygarment import data_config
    from pygarment.meshgen import boxmeshgen
    from pygarment.meshgen.sim_config import PathCofig
    from pygarment.meshgen.simulation import run_sim

    # GarmentCode's coarse-edge warning references Edge.name before defining it.
    if not hasattr(boxmeshgen.Edge, "name"):
        boxmeshgen.Edge.name = "unnamed"
    BoxMesh = boxmeshgen.BoxMesh

    simulation_dir = output_dir / "simulation"
    simulation_dir.mkdir(parents=True, exist_ok=True)
    write_system_config(garmentcode_dir, simulation_dir)
    properties = data_config.Properties(str(sim_config))
    properties.set_section_stats(
        "sim", fails={}, sim_time={}, spf={}, fin_frame={}, body_collisions={}, self_collisions={}
    )
    properties.set_section_stats("render", render_time={})
    if args.max_sim_steps is not None:
        properties["sim"]["config"]["max_sim_steps"] = args.max_sim_steps
    if args.max_sim_time is not None:
        properties["sim"]["config"]["max_sim_time"] = args.max_sim_time
    if args.resolution_scale is not None:
        properties["sim"]["config"]["resolution_scale"] = args.resolution_scale

    paths = PathCofig(
        in_element_path=pattern_dir,
        out_path=simulation_dir,
        in_name=name,
        body_name=args.body,
        smpl_body=False,
        add_timestamp=False,
    )
    print(f"generating box mesh in {paths.out_el}", flush=True)
    box_mesh = BoxMesh(paths.in_g_spec, properties["sim"]["config"]["resolution_scale"])
    try:
        box_mesh.load()
    except boxmeshgen.DegenerateTrianglesError as exc:
        topology = {
            field: design_document["design"]["meta"][field].get("v")
            for field in TOPOLOGY_FIELDS
        }
        raise SystemExit(
            "GarmentCode could not mesh the predicted design because it contains "
            f"degenerate panels. Effective topology: {topology}. "
            "This is a design-prediction/geometry failure, not a CUDA failure. "
            "Use --override-upper/--override-wb/--override-bottom when the inferred "
            f"classes are wrong. Original error: {exc}"
        ) from None
    box_mesh.serialize(
        paths,
        store_panels=False,
        uv_config=properties["render"]["config"]["uv_texture"],
    )
    properties.serialize(paths.element_sim_props)
    run_sim(
        box_mesh.name,
        properties,
        paths,
        save_v_norms=False,
        store_usd=False,
        optimize_storage=False,
        verbose=False,
    )
    properties.serialize(paths.element_sim_props)

    front = paths.render_path("front")
    back = paths.render_path("back")
    if not front.is_file():
        raise RuntimeError(f"GarmentCode did not produce the expected front render: {front}")
    print(f"wrote front render: {front}")
    print(f"wrote front preview: {white_background_preview(front)}")
    if back.is_file():
        print(f"wrote back render: {back}")
        print(f"wrote back preview: {white_background_preview(back)}")
    print(f"wrote simulated mesh: {paths.g_sim}")


if __name__ == "__main__":
    main()
