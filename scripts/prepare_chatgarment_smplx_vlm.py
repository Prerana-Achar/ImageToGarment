#!/usr/bin/env python3
"""Convert completed GarmentCodeSMPLX samples to ChatGarment VLM JSON."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
CHATGARMENT = ROOT / "ChatGarment"
DEFAULT_DATASET = Path("/is/cluster/fast/pachar/Data/GarmentCodeSMPLX")
DEFAULT_OUT = Path("/is/cluster/fast/pachar/Data/ChatGarmentSMPLXVLM")
TARGET_VERSION = 1
SEG = "[SEG]"


def flatten_schema(node: dict[str, Any], prefix: tuple[str, ...] = ()) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if "type" in node and "v" in node:
        result[".".join(prefix)] = node
        return result
    for key, value in node.items():
        if isinstance(value, dict):
            result.update(flatten_schema(value, (*prefix, key)))
    return result


def readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return path.stat().st_size > 0
    except Exception:
        return False


def set_nested(root: dict[str, Any], path: str, value: Any) -> None:
    node = root
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def normalized_float(value: float, spec: dict[str, Any]) -> tuple[float, bool]:
    lo, hi = map(float, spec["range"])
    if not math.isfinite(value) or hi <= lo:
        raise ValueError(f"invalid float/range: value={value}, range={spec['range']}")
    raw = (value - lo) / (hi - lo)
    clipped = not 0.0 <= raw <= 1.0
    return min(max(raw, 0.0), 1.0), clipped


def collect_float_labels(
    node: Any,
    prefix: tuple[str, ...],
    normalized_by_path: dict[str, float],
    output: list[float],
) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            collect_float_labels(value, (*prefix, key), normalized_by_path, output)
    elif node == SEG:
        path = ".".join(prefix)
        output.append(normalized_by_path[path])


def answer_text(design: dict[str, Any]) -> str:
    text = json.dumps(
        {"wholebody_garment": design},
        ensure_ascii=True,
        separators=(", ", ": "),
    )
    text = text.replace(f'"{SEG}"', SEG)
    return text.replace('"', "'")


def convert_sample(
    metadata: dict[str, Any],
    schema: dict[str, dict[str, Any]],
    float_paths: set[str],
) -> tuple[str, list[float], int, list[str]]:
    values = metadata["design_values"]
    active = metadata["design_active_paths"]
    unsupported = set(metadata.get("design_unsupported_paths") or ())
    design: dict[str, Any] = {}
    normalized_by_path: dict[str, float] = {}
    clipped = 0
    skipped: list[str] = []

    for path in active:
        if path in unsupported:
            skipped.append(path)
            continue
        spec = schema.get(path)
        if spec is None:
            raise ValueError(f"active path {path!r} is absent from the GarmentCode schema")
        value = values[path]
        full_path = f"design.{path}"
        if spec["type"] == "float":
            if full_path not in float_paths:
                raise ValueError(
                    f"active float path {full_path!r} is unsupported by ChatGarment's 76-float head"
                )
            normalized, was_clipped = normalized_float(float(value), spec)
            normalized_by_path[full_path] = normalized
            clipped += int(was_clipped)
            set_nested(design, path, SEG)
        else:
            set_nested(design, path, value)

    floats: list[float] = []
    collect_float_labels(design, ("design",), normalized_by_path, floats)
    if not floats:
        raise ValueError("sample has no active ChatGarment float targets")
    return answer_text(design), floats, clipped, skipped


def scan_split(
    dataset_root: Path,
    split: str,
    schema: dict[str, dict[str, Any]],
    float_paths: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples_root = dataset_root / split / "samples"
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    garments = 0
    clipped_targets = 0
    skipped_unsupported: Counter[str] = Counter()
    bodies: set[str] = set()
    categories: Counter[str] = Counter()

    if not samples_root.is_dir():
        return records, {
            "garments": 0, "examples": 0, "bodies": [], "categories": {},
            "clipped_targets": 0, "skipped_unsupported": {}, "rejected": [],
        }

    for folder in sorted(path for path in samples_root.iterdir() if path.is_dir()):
        name = folder.name
        metadata_path = folder / f"{name}.json"
        pkl_path = folder / f"{name}.pkl"
        poses = [folder / f"{name}_pose{index}.png" for index in (1, 2, 3)]
        try:
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("design_target_version") != TARGET_VERSION:
                raise ValueError(f"expected design_target_version={TARGET_VERSION}")
            if not pkl_path.is_file() or pkl_path.stat().st_size == 0:
                raise ValueError("missing garment PKL")
            if not all(readable_image(path) for path in poses):
                raise ValueError("pose1, pose2, or pose3 is missing/corrupt")
            answer, floats, clipped, skipped = convert_sample(
                metadata, schema, float_paths
            )
        except Exception as exc:
            rejected.append({"folder": str(folder), "reason": str(exc)})
            continue

        garments += 1
        clipped_targets += clipped
        skipped_unsupported.update(skipped)
        bodies.add(str(metadata["body_name"]))
        categories[str(metadata["category"])] += 1
        for pose_index, image_path in enumerate(poses, start=1):
            records.append(
                {
                    "id": f"{split}:{name}:pose{pose_index}",
                    "image": str(image_path.resolve()),
                    "body_name": metadata["body_name"],
                    "garment_name": name,
                    "category": metadata["category"],
                    "pose": pose_index,
                    "conversations": [
                        {
                            "from": "human",
                            "value": (
                                "<image>\nEstimate the complete GarmentCode sewing "
                                "pattern for the outfit in this image."
                            ),
                        },
                        {"from": "gpt", "value": answer},
                    ],
                    "all_floats": floats,
                }
            )

    return records, {
        "garments": garments,
        "examples": len(records),
        "bodies": sorted(bodies),
        "categories": dict(sorted(categories.items())),
        "clipped_targets": clipped_targets,
        "skipped_unsupported": dict(sorted(skipped_unsupported.items())),
        "rejected_count": len(rejected),
        "rejected": rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-train-bodies", type=int, default=2)
    parser.add_argument("--min-val-bodies", type=int, default=2)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--schema",
        type=Path,
        default=ROOT / "GarmentCodeRC/assets/design_params/default_new.yaml",
    )
    parser.add_argument(
        "--float-paths",
        type=Path,
        default=CHATGARMENT / "docs/all_float_paths.json",
    )
    args = parser.parse_args()

    schema_doc = yaml.safe_load(args.schema.read_text())
    schema = flatten_schema(schema_doc["design"])
    float_path_list = json.loads(args.float_paths.read_text())
    if len(float_path_list) != 76 or len(set(float_path_list)) != 76:
        raise RuntimeError("ChatGarment float path file must contain 76 unique paths")
    float_paths = set(float_path_list)

    output: dict[str, list[dict[str, Any]]] = {}
    report: dict[str, Any] = {
        "dataset_root": str(args.dataset_root.resolve()),
        "schema": str(args.schema.resolve()),
        "float_paths": str(args.float_paths.resolve()),
        "target_version": TARGET_VERSION,
        "splits": {},
    }
    for split in ("train", "val"):
        output[split], report["splits"][split] = scan_split(
            args.dataset_root.resolve(), split, schema, float_paths
        )

    train_bodies = set(report["splits"]["train"]["bodies"])
    val_bodies = set(report["splits"]["val"]["bodies"])
    report["body_overlap"] = sorted(train_bodies & val_bodies)
    report["ready"] = (
        bool(output["train"])
        and bool(output["val"])
        and not report["body_overlap"]
        and len(train_bodies) >= args.min_train_bodies
        and len(val_bodies) >= args.min_val_bodies
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for split, records in output.items():
        (args.out / f"{split}.json").write_text(json.dumps(records, indent=2))
    (args.out / "report.json").write_text(json.dumps(report, indent=2))

    print(
        f"train garments={report['splits']['train']['garments']} "
        f"examples={len(output['train'])} bodies={len(train_bodies)}"
    )
    print(
        f"val garments={report['splits']['val']['garments']} "
        f"examples={len(output['val'])} bodies={len(val_bodies)}"
    )
    print(f"body overlap={report['body_overlap']} ready={report['ready']}")
    print(f"wrote {args.out.resolve()}")
    if not report["ready"] and not args.allow_incomplete:
        raise SystemExit(
            "VLM export is not training-ready; inspect report.json and finish validation renders."
        )


if __name__ == "__main__":
    main()
