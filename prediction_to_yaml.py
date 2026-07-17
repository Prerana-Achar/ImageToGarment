#!/usr/bin/env python
"""Convert an inference JSON into a GarmentCode-style design YAML."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


TOP_PREFIXES = ("wholebody_garment", "upperbody_garment", "lowerbody_garment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, help="JSON written by infer_dinov2.py")
    parser.add_argument(
        "--template",
        default="GarmentCodeRC/assets/design_params/default_new.yaml",
        help="GarmentCode design YAML/template carrying v/range/type fields",
    )
    parser.add_argument("--out", required=True, help="output design YAML")
    parser.add_argument(
        "--source-mode",
        choices=("split", "wholebody", "all"),
        default="split",
        help="which top-level prediction namespace to write into the single design tree",
    )
    return parser.parse_args()


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


def apply_prediction(design: dict[str, Any], pred: dict[str, Any], source_mode: str) -> list[str]:
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

    return warnings


def main() -> None:
    args = parse_args()
    with open(args.template) as f:
        template = yaml.safe_load(f)
    with open(args.prediction) as f:
        inference = json.load(f)

    design = copy.deepcopy(template["design"])
    warnings = apply_prediction(design, inference["prediction"], args.source_mode)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump({"design": design}, f, sort_keys=False, default_flow_style=False)

    print(f"wrote {args.out}")
    if warnings:
        print(f"warnings: {len(warnings)} paths were not written")
        for warning in warnings[:20]:
            print(f"  {warning}")


if __name__ == "__main__":
    main()
