#!/usr/bin/env python3
"""Map ChatGarment prepared data into an existing GarmentCode target schema."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PREFIXES = (
    "upperbody_garment.",
    "lowerbody_garment.",
    "wholebody_garment.",
)
ROOT_PATHS = ("meta.upper", "meta.wb", "meta.bottom")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat-prepared-dir", required=True, type=Path)
    parser.add_argument("--target-prepared-dir", required=True, type=Path)
    parser.add_argument(
        "--chat-design-schema",
        type=Path,
        default=Path(__file__).resolve().parent
        / "GarmentCodeRC/assets/design_params/design_used.yaml",
        help="Schema that normalized ChatGarment continuous [SEG] values.",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--minimum-mapped-fraction", type=float, default=0.95)
    parser.add_argument("--numeric-select-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--allow-unseen-routes",
        action="store_true",
        help="Keep ChatGarment root tuples not observed in target training data.",
    )
    return parser.parse_args()


def canonical_path(path: str) -> str:
    for prefix in PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def flatten_parameter_specs(
    node: dict[str, Any], prefix: str = ""
) -> dict[str, dict[str, Any]]:
    flattened: dict[str, dict[str, Any]] = {}
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and isinstance(value.get("type"), str):
            flattened[path] = value
        elif isinstance(value, dict):
            flattened.update(flatten_parameter_specs(value, path))
    return flattened


def value_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, json.dumps(value, sort_keys=True)


def is_numeric_vocab(vocab: list[Any]) -> bool:
    return (
        len(vocab) >= 2
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in vocab
        )
    )


def load_prepared(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    required = ("schema.json", "splits.json", "images.json", "targets.npz")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{path} is missing prepared files: {missing}")
    schema = json.loads((path / "schema.json").read_text())
    splits = json.loads((path / "splits.json").read_text())
    images = json.loads((path / "images.json").read_text())
    archive = np.load(path / "targets.npz", allow_pickle=True)
    arrays = {key: archive[key] for key in archive.files}
    archive.close()
    return schema, splits, images, arrays


def target_routes(
    schema: dict[str, Any], splits: dict[str, Any], arrays: dict[str, np.ndarray]
) -> set[tuple[int, ...]]:
    categorical_paths = list(schema["cat_vocab"])
    missing = [path for path in ROOT_PATHS if path not in categorical_paths]
    if missing:
        raise ValueError(f"Target schema is missing topology roots: {missing}")
    root_indices = [categorical_paths.index(path) for path in ROOT_PATHS]
    row_of = {str(gid): index for index, gid in enumerate(arrays["gids"])}
    routes = set()
    for gid in splits["train"]:
        row = arrays["y_cat"][row_of[str(gid)]]
        route = tuple(int(row[index]) for index in root_indices)
        if min(route) < 0:
            raise ValueError(f"Target training garment {gid!r} has inactive roots")
        routes.add(route)
    return routes




def route_semantically_valid(schema: dict[str, Any], route: tuple[int, ...]) -> bool:
    """Return whether a canonical root tuple is legal for garment-tree decoding."""
    if len(route) != len(ROOT_PATHS) or min(route) < 0:
        return False
    root_vocabs = [schema["cat_vocab"][path] for path in ROOT_PATHS]
    for position, value in enumerate(route):
        if value >= len(root_vocabs[position]):
            return False
    upper, waistband, bottom = (
        root_vocabs[position][route[position]] for position in range(3)
    )
    if upper is None and bottom is None:
        return False
    if bottom is None and waistband is not None:
        return False
    return True

def main() -> None:
    args = parse_args()
    if not 0.0 <= args.minimum_mapped_fraction <= 1.0:
        raise ValueError("--minimum-mapped-fraction must be in [0, 1]")
    chat_schema, chat_splits, chat_images, chat = load_prepared(
        args.chat_prepared_dir
    )
    target_schema, target_splits, _, target = load_prepared(
        args.target_prepared_dir
    )
    design_document = yaml.safe_load(args.chat_design_schema.read_text())
    if not isinstance(design_document, dict) or "design" not in design_document:
        raise ValueError(f"Invalid ChatGarment design schema: {args.chat_design_schema}")
    chat_specs = flatten_parameter_specs(design_document["design"])
    known_routes = target_routes(target_schema, target_splits, target)

    target_cont_paths = list(target_schema.get("cont_slots", {}))
    target_const_paths = list(target_schema.get("const_slots", {}))
    target_cat_paths = list(target_schema["cat_vocab"])
    target_cont_index = {path: index for index, path in enumerate(target_cont_paths)}
    target_const_index = {path: index for index, path in enumerate(target_const_paths)}
    target_cat_index = {path: index for index, path in enumerate(target_cat_paths)}
    target_cat_values = {
        path: {value_key(value): index for index, value in enumerate(vocab)}
        for path, vocab in target_schema["cat_vocab"].items()
    }
    numeric_select = {
        path: np.asarray(vocab, dtype=np.float64)
        for path, vocab in target_schema["cat_vocab"].items()
        if is_numeric_vocab(vocab)
    }
    const_ranges = {
        path: tuple(map(float, target_schema["const_ranges"][path]))
        for path in target_const_paths
    }
    root_indices = [target_cat_index[path] for path in ROOT_PATHS]

    chat_cont_paths = [canonical_path(path) for path in chat_schema.get("cont_slots", {})]
    chat_const_paths = [canonical_path(path) for path in chat_schema.get("const_slots", {})]
    chat_cat_paths_raw = list(chat_schema["cat_vocab"])
    chat_cat_paths = [canonical_path(path) for path in chat_cat_paths_raw]
    chat_cat_vocab = [chat_schema["cat_vocab"][path] for path in chat_cat_paths_raw]
    row_of = {str(gid): index for index, gid in enumerate(chat["gids"])}

    y_cont_source = chat["y_cont"]
    cont_mask_source = chat["mask"].astype(bool)
    y_const_source = chat["y_const"]
    const_mask_source = chat["const_mask"].astype(bool)
    y_cat_source = chat["y_cat"]

    output_gids: list[str] = []
    output_y_cont: list[np.ndarray] = []
    output_cont_mask: list[np.ndarray] = []
    output_y_const: list[np.ndarray] = []
    output_const_mask: list[np.ndarray] = []
    output_y_cat: list[np.ndarray] = []
    output_images: dict[str, Any] = {}
    output_splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_report: dict[str, Counter[str]] = {}
    unmapped_paths: Counter[str] = Counter()
    unmapped_values: Counter[str] = Counter()
    mapping_histogram: Counter[str] = Counter()
    numeric_adjustments: Counter[str] = Counter()

    for split in ("train", "val", "test"):
        counts: Counter[str] = Counter()
        for source_gid in chat_splits.get(split, []):
            source_gid = str(source_gid)
            if source_gid not in row_of or source_gid not in chat_images:
                counts["missing_prepared_record"] += 1
                continue
            row = row_of[source_gid]
            y_cont = np.zeros(len(target_cont_paths), dtype=np.float32)
            cont_mask = np.zeros(len(target_cont_paths), dtype=np.float32)
            y_const = np.zeros(len(target_const_paths), dtype=np.float32)
            const_mask = np.zeros(len(target_const_paths), dtype=np.float32)
            y_cat = np.full(len(target_cat_paths), -1, dtype=np.int64)
            total_active = 0
            mapped_active = 0
            record_unmapped_paths: list[str] = []
            record_unmapped_values: list[str] = []
            record_numeric_adjustments: list[str] = []

            for index, path in enumerate(chat_cont_paths):
                if not cont_mask_source[row, index]:
                    continue
                total_active += 1
                value = float(y_cont_source[row, index])
                if path in target_cont_index:
                    target_index = target_cont_index[path]
                    if cont_mask[target_index] and not np.isclose(
                        y_cont[target_index], value, atol=1e-6
                    ):
                        raise ValueError(f"Conflicting active continuous value for {source_gid}:{path}")
                    y_cont[target_index] = value
                    cont_mask[target_index] = 1.0
                    mapped_active += 1
                elif path in target_const_index and path in chat_specs:
                    target_index = target_const_index[path]
                    source_lo, source_hi = map(float, chat_specs[path]["range"])
                    normalized_value = float(np.clip(value, 0.0, 1.0))
                    raw_value = source_lo + normalized_value * (source_hi - source_lo)
                    lo, hi = const_ranges[path]
                    target_value = float(np.clip(raw_value, lo, hi))
                    if not np.isclose(value, normalized_value, atol=1e-7):
                        record_numeric_adjustments.append(
                            f"{path}:source_normalized_clipped"
                        )
                    if not np.isclose(raw_value, target_value, atol=1e-7):
                        record_numeric_adjustments.append(
                            f"{path}:target_range_clipped"
                        )
                    raw_value = target_value
                    if const_mask[target_index] and not np.isclose(
                        y_const[target_index], raw_value, atol=1e-5
                    ):
                        raise ValueError(f"Conflicting active numeric value for {source_gid}:{path}")
                    y_const[target_index] = raw_value
                    const_mask[target_index] = 1.0
                    mapped_active += 1
                else:
                    record_unmapped_paths.append(path)

            for index, path in enumerate(chat_const_paths):
                if not const_mask_source[row, index]:
                    continue
                total_active += 1
                value = float(y_const_source[row, index])
                if path in target_const_index:
                    target_index = target_const_index[path]
                    lo, hi = const_ranges[path]
                    target_value = float(np.clip(value, lo, hi))
                    if not np.isclose(value, target_value, atol=1e-7):
                        record_numeric_adjustments.append(
                            f"{path}:target_range_clipped"
                        )
                    value = target_value
                    if const_mask[target_index] and not np.isclose(
                        y_const[target_index], value, atol=1e-5
                    ):
                        raise ValueError(f"Conflicting active constant for {source_gid}:{path}")
                    y_const[target_index] = value
                    const_mask[target_index] = 1.0
                    mapped_active += 1
                elif path in numeric_select:
                    distances = np.abs(numeric_select[path] - value)
                    target_value = int(distances.argmin())
                    if float(distances[target_value]) <= args.numeric_select_tolerance:
                        categorical_index = target_cat_index[path]
                        if y_cat[categorical_index] >= 0 and y_cat[categorical_index] != target_value:
                            raise ValueError(f"Conflicting numeric-select value for {source_gid}:{path}")
                        y_cat[categorical_index] = target_value
                        mapped_active += 1
                    else:
                        record_unmapped_values.append(f"{path}={value:.8g}")
                else:
                    record_unmapped_paths.append(path)

            for index, path in enumerate(chat_cat_paths):
                source_value_index = int(y_cat_source[row, index])
                if source_value_index < 0:
                    continue
                total_active += 1
                value = chat_cat_vocab[index][source_value_index]
                if path in target_cat_index:
                    target_value = target_cat_values[path].get(value_key(value))
                    if target_value is not None:
                        categorical_index = target_cat_index[path]
                        if y_cat[categorical_index] >= 0 and y_cat[categorical_index] != target_value:
                            raise ValueError(f"Conflicting categorical value for {source_gid}:{path}")
                        y_cat[categorical_index] = target_value
                        mapped_active += 1
                    else:
                        record_unmapped_values.append(
                            f"{path}={json.dumps(value, sort_keys=True)}"
                        )
                else:
                    record_unmapped_paths.append(path)

            fraction = mapped_active / max(total_active, 1)
            mapping_histogram[f"{int(fraction * 20) * 5:02d}-{min(100, int(fraction * 20) * 5 + 4):02d}"] += 1
            route = tuple(int(y_cat[index]) for index in root_indices)
            roots_complete = min(route) >= 0
            route_known = roots_complete and route in known_routes
            route_valid = route_semantically_valid(target_schema, route)
            counts["scanned"] += 1
            counts["mapped_active_fields"] += mapped_active
            counts["active_fields"] += total_active
            counts["mapped_fraction_ge_threshold"] += fraction >= args.minimum_mapped_fraction
            counts["known_route"] += route_known
            counts["semantic_route"] += route_valid
            if fraction < args.minimum_mapped_fraction:
                counts["rejected_low_mapping"] += 1
                continue
            if not roots_complete:
                counts["rejected_incomplete_roots"] += 1
                continue
            if not route_valid:
                counts["rejected_invalid_semantic_route"] += 1
                continue
            if not args.allow_unseen_routes and not route_known:
                counts["rejected_unseen_route"] += 1
                continue

            for path in record_unmapped_paths:
                unmapped_paths[path] += 1
            for value in record_unmapped_values:
                unmapped_values[value] += 1
            for adjustment in record_numeric_adjustments:
                numeric_adjustments[adjustment] += 1
            output_gid = f"chat:{source_gid}"
            output_gids.append(output_gid)
            output_y_cont.append(y_cont)
            output_cont_mask.append(cont_mask)
            output_y_const.append(y_const)
            output_const_mask.append(const_mask)
            output_y_cat.append(y_cat)
            output_images[output_gid] = chat_images[source_gid]
            output_splits[split].append(output_gid)
            counts["selected"] += 1
        split_report[split] = counts

    if not output_splits["train"]:
        raise RuntimeError("Compatibility filtering selected no ChatGarment training garments")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    schema_bytes = json.dumps(target_schema, indent=2).encode("utf-8")
    (args.out_dir / "schema.json").write_bytes(schema_bytes)
    (args.out_dir / "images.json").write_text(json.dumps(output_images, indent=2))
    (args.out_dir / "splits.json").write_text(
        json.dumps(
            {
                "train": output_splits["train"],
                "val": output_splits["val"],
                "test": output_splits["test"],
                "source": str(args.chat_prepared_dir),
                "target_schema_source": str(args.target_prepared_dir),
            },
            indent=2,
        )
    )
    np.savez_compressed(
        args.out_dir / "targets.npz",
        gids=np.asarray(output_gids),
        y_cont=np.stack(output_y_cont),
        mask=np.stack(output_cont_mask),
        y_const=np.stack(output_y_const),
        const_mask=np.stack(output_const_mask),
        y_cat=np.stack(output_y_cat),
    )
    report = {
        "format": "chatgarment_current_schema_auxiliary/v1",
        "chat_prepared_dir": str(args.chat_prepared_dir),
        "target_prepared_dir": str(args.target_prepared_dir),
        "target_schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "chat_design_schema": str(args.chat_design_schema),
        "minimum_mapped_fraction": args.minimum_mapped_fraction,
        "numeric_select_tolerance": args.numeric_select_tolerance,
        "require_known_target_route": not args.allow_unseen_routes,
        "known_target_routes": len(known_routes),
        "splits": {name: dict(counts) for name, counts in split_report.items()},
        "selected_garments": len(output_gids),
        "selected_images": sum(
            sum(image is not None for views in record["frames"].values() for image in views)
            for record in output_images.values()
        ),
        "mapping_fraction_histogram_percent": dict(sorted(mapping_histogram.items())),
        "unmapped_active_paths_in_selected": dict(unmapped_paths.most_common()),
        "unmapped_active_values_in_selected": dict(unmapped_values.most_common()),
        "numeric_adjustments_in_selected": dict(numeric_adjustments.most_common()),
    }
    (args.out_dir / "compatibility_report.json").write_text(
        json.dumps(report, indent=2)
    )
    print(
        "selected "
        + " ".join(
            f"{split}={split_report[split]['selected']}/{split_report[split]['scanned']}"
            for split in ("train", "val", "test")
        )
    )
    print(f"images={report['selected_images']} wrote={args.out_dir}")


if __name__ == "__main__":
    main()
