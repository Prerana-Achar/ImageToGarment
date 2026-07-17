#!/usr/bin/env python3
"""Convert a balanced GarmentCodeSMPLX tree into ImageToGarment prepared data.

The converter accepts only DESIGN_TARGET_VERSION=1 samples, preserves body-level
split isolation, masks topology-inactive parameters, and writes an auditable
balance_report.json. Existing source samples are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

CATEGORIES = (
    "top", "hoodie", "dress", "jumpsuit", "bottoms", "skirts",
    "top_and_bottom", "top_and_skirt", "hoodie_and_bottom", "hoodie_and_skirt",
)
TARGET_VERSION = 1
TOPOLOGY_CONTROLLED = {
    "meta.upper", "meta.wb", "meta.bottom", "shirt.strapless",
    "sleeve.sleeveless", "left.enable_asym", "left.shirt.strapless",
    "left.sleeve.sleeveless", "collar.component.style",
}


def flatten(node: dict[str, Any], prefix: tuple[str, ...] = ()) -> OrderedDict[str, dict[str, Any]]:
    out: OrderedDict[str, dict[str, Any]] = OrderedDict()
    if "type" in node and "v" in node:
        out[".".join(prefix)] = node
        return out
    for key, value in node.items():
        if isinstance(value, dict):
            out.update(flatten(value, prefix + (key,)))
    return out


def choices(spec: dict[str, Any]) -> list[Any]:
    values = list(spec.get("range") or [])
    if spec.get("type") == "select_null" and None not in values:
        values.append(None)
    return values


def value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def readable_png(path: Path) -> bool:
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
        return path.stat().st_size > 0
    except Exception:
        return False


def pick_views(folder: Path, name: str) -> list[str] | None:
    textured = [folder / f"{name}_render_{view}_textured.png" for view in ("front", "back")]
    plain = [folder / f"{name}_render_{view}.png" for view in ("front", "back")]
    selected = textured if all(readable_png(path) for path in textured) else plain
    if not all(readable_png(path) for path in selected):
        return None
    return [str(path.resolve()) for path in selected]


def design_hash(meta: dict[str, Any]) -> str:
    active = sorted(meta["design_active_paths"])
    payload = [(path, meta["design_values"][path]) for path in active]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf8")).hexdigest()


def scan(root: Path, specs: OrderedDict[str, dict[str, Any]]):
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    expected = set(specs)
    for source_split in ("train", "val", "test"):
        samples = root / source_split / "samples"
        if not samples.is_dir():
            continue
        for folder in sorted(path for path in samples.iterdir() if path.is_dir()):
            name = folder.name
            meta_path = folder / f"{name}.json"
            pkl_path = folder / f"{name}.pkl"
            reason = None
            try:
                meta = json.loads(meta_path.read_text())
            except Exception as exc:
                meta = None
                reason = f"metadata unreadable: {exc}"
            if meta is not None:
                if meta.get("design_target_version") != TARGET_VERSION:
                    reason = "legacy sample: missing design_target_version=1"
                elif not pkl_path.is_file() or pkl_path.stat().st_size == 0:
                    reason = "missing garment pkl"
                elif set(meta.get("design_values") or {}) != expected:
                    missing = sorted(expected - set(meta.get("design_values") or {}))
                    extra = sorted(set(meta.get("design_values") or {}) - expected)
                    reason = f"target paths differ (missing={missing}, extra={extra})"
                elif not set(meta.get("design_active_paths") or ()).issubset(expected):
                    reason = "active mask contains unknown target paths"
                elif meta.get("category") not in CATEGORIES:
                    reason = f"unknown category {meta.get('category')!r}"
            views = pick_views(folder, name) if reason is None else None
            if reason is None and views is None:
                reason = "missing/corrupt front or back render"
            if reason is not None:
                rejected.append({"split": source_split, "folder": str(folder), "reason": reason})
                continue
            records.append({
                "gid": f"{source_split}:{name}", "name": name,
                "source_split": source_split, "folder": folder, "meta": meta,
                "views": views, "design_hash": design_hash(meta),
            })
    return records, rejected


def split_by_body(records: list[dict[str, Any]], test_fraction: float, seed: int):
    assigned: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    val_bodies = sorted({r["meta"]["body_name"] for r in records if r["source_split"] == "val"})
    rng = np.random.default_rng(seed)
    rng.shuffle(val_bodies)
    n_test = int(round(len(val_bodies) * test_fraction))
    if len(val_bodies) >= 2 and test_fraction > 0:
        n_test = min(max(n_test, 1), len(val_bodies) - 1)
    test_bodies = set(val_bodies[:n_test])
    for record in records:
        source = record["source_split"]
        if source == "train":
            split = "train"
        elif source == "test" or record["meta"]["body_name"] in test_bodies:
            split = "test"
        else:
            split = "val"
        record["split"] = split
        assigned[split].append(record["gid"])
    return {key: sorted(value) for key, value in assigned.items()}


def build_arrays(records: list[dict[str, Any]], specs: OrderedDict[str, dict[str, Any]]):
    active_any = {path for record in records for path in record["meta"]["design_active_paths"]}
    regression = [path for path, spec in specs.items()
                  if spec["type"] in ("float", "int") and path in active_any]
    categorical = [path for path, spec in specs.items()
                   if spec["type"] in ("bool", "select", "select_null") and path in active_any]
    observed = {path: set() for path in categorical}
    for record in records:
        active = set(record["meta"]["design_active_paths"])
        values = record["meta"]["design_values"]
        for path in categorical:
            if path in active:
                observed[path].add(value_key(values[path]))
    cat_vocab = OrderedDict()
    for path in categorical:
        cat_vocab[path] = [value for value in choices(specs[path]) if value_key(value) in observed[path]]
        if not cat_vocab[path]:
            raise RuntimeError(f"No observed class for active categorical path {path}")

    cont_slots = OrderedDict((path, index) for index, path in enumerate(regression))
    cat_paths = list(cat_vocab)
    gids = [record["gid"] for record in records]
    y_cont = np.zeros((len(records), len(regression)), dtype=np.float32)
    mask = np.zeros_like(y_cont)
    y_const = np.zeros((len(records), 0), dtype=np.float32)
    const_mask = np.zeros_like(y_const)
    y_cat = np.full((len(records), len(categorical)), -1, dtype=np.int64)
    for row, record in enumerate(records):
        values = record["meta"]["design_values"]
        active = set(record["meta"]["design_active_paths"])
        for path, col in cont_slots.items():
            if path not in active:
                continue
            lo, hi = map(float, specs[path]["range"])
            raw = float(values[path])
            y_cont[row, col] = np.clip((raw - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
            mask[row, col] = 1.0
        for col, path in enumerate(cat_paths):
            if path in active:
                y_cat[row, col] = next(i for i, value in enumerate(cat_vocab[path])
                                            if value_key(value) == value_key(values[path]))
    schema = {
        "n_cont": len(regression), "n_const": 0, "n_cat": len(categorical),
        "cont_slots": cont_slots, "const_slots": {}, "const_ranges": {},
        "cat_vocab": cat_vocab,
        "target_contract": "GarmentCodeSMPLX/design_target_version=1",
        "inactive_target_semantics": "mask=0 and y_cat=-1",
    }
    arrays = dict(gids=np.asarray(gids), y_cont=y_cont, mask=mask,
                  y_const=y_const, const_mask=const_mask, y_cat=y_cat)
    return schema, arrays


def balance_report(records, rejected, splits, specs, schema, strict_spread):
    by_gid = {record["gid"]: record for record in records}
    split_report = {}
    free_failures = []
    for split, gids in splits.items():
        category_counts = Counter(by_gid[gid]["meta"]["category"] for gid in gids)
        cat_counts = {}
        reg_counts = {}
        for path, vocab in schema["cat_vocab"].items():
            counts = Counter()
            for gid in gids:
                meta = by_gid[gid]["meta"]
                if path in meta["design_active_paths"]:
                    counts[value_key(meta["design_values"][path])] += 1
            ordered = {value_key(value): counts[value_key(value)] for value in vocab}
            vals = list(ordered.values())
            spread = max(vals) - min(vals) if vals else 0
            cat_counts[path] = {"counts": ordered, "spread": spread,
                                "topology_controlled": path in TOPOLOGY_CONTROLLED}
            mean_count = (sum(vals) / len(vals)) if vals else 0.0
            tolerance = max(strict_spread, int(math.ceil(0.10 * mean_count)))
            if split == "train" and path not in TOPOLOGY_CONTROLLED and spread > tolerance:
                free_failures.append({"path": path, "spread": spread, "tolerance": tolerance, "counts": ordered})
        for path in schema["cont_slots"]:
            vals = [float(by_gid[gid]["meta"]["design_values"][path]) for gid in gids
                    if path in by_gid[gid]["meta"]["design_active_paths"]]
            lo, hi = map(float, specs[path]["range"])
            hist = np.histogram(vals, bins=10, range=(lo, hi))[0].tolist() if vals else [0] * 10
            reg_counts[path] = {"active": len(vals), "min": min(vals) if vals else None,
                                "max": max(vals) if vals else None, "histogram_10": hist}
        split_report[split] = {
            "samples": len(gids),
            "bodies": len({by_gid[gid]["meta"]["body_name"] for gid in gids}),
            "category_counts": {category: category_counts[category] for category in CATEGORIES},
            "categorical": cat_counts, "regression": reg_counts,
        }
    body_sets = {split: {by_gid[gid]["meta"]["body_name"] for gid in gids}
                 for split, gids in splits.items()}
    body_overlap = {"train_val": sorted(body_sets["train"] & body_sets["val"]),
                    "train_test": sorted(body_sets["train"] & body_sets["test"]),
                    "val_test": sorted(body_sets["val"] & body_sets["test"])}
    hash_sets = {split: {by_gid[gid]["design_hash"] for gid in gids} for split, gids in splits.items()}
    design_overlap = {"train_val": len(hash_sets["train"] & hash_sets["val"]),
                      "train_test": len(hash_sets["train"] & hash_sets["test"]),
                      "val_test": len(hash_sets["val"] & hash_sets["test"])}
    category_failures = []
    for split in ("train", "val"):
        vals = list(split_report[split]["category_counts"].values())
        if not vals or min(vals) == 0 or max(vals) - min(vals) > 1:
            category_failures.append({"split": split, "counts": split_report[split]["category_counts"]})
    unsupported = sorted(set(specs) - set(schema["cont_slots"]) - set(schema["cat_vocab"]))
    unobserved_choices = {
        path: [value for value in choices(specs[path]) if value not in vocab]
        for path, vocab in schema["cat_vocab"].items()
        if any(value not in vocab for value in choices(specs[path]))
    }
    ready = (not rejected and not category_failures and not free_failures
             and not any(body_overlap.values()) and bool(splits["train"]) and bool(splits["val"]))
    return {
        "ready": ready, "target_version": TARGET_VERSION,
        "ready_definition": "no rejected folders; body-disjoint nonempty train/val; all ten public categories differ by at most one; every non-topology-controlled active categorical field is within max(strict_spread, 10% of its mean class count)",
        "strict_categorical_spread": strict_spread,
        "splits": split_report, "body_overlap": body_overlap,
        "exact_design_overlap_counts": design_overlap,
        "unsupported_or_never_active_schema_paths": unsupported,
        "unobserved_schema_choices": unobserved_choices,
        "category_failures": category_failures,
        "categorical_failures": free_failures,
        "rejected_count": len(rejected), "rejected": rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--schema", default=str(Path(__file__).resolve().parent / "GarmentCodeRC/assets/design_params/default_new.yaml"))
    parser.add_argument("--test-fraction-of-val-bodies", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-categorical-spread", type=int, default=1)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.test_fraction_of_val_bodies < 1:
        parser.error("--test-fraction-of-val-bodies must be in [0,1)")
    root = Path(args.dataset_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    document = yaml.safe_load(Path(args.schema).read_text())
    specs = flatten(document["design"])
    records, rejected = scan(root, specs)
    if not records:
        raise RuntimeError(f"No version-{TARGET_VERSION} training-ready samples found under {root}")
    splits = split_by_body(records, args.test_fraction_of_val_bodies, args.seed)
    schema, arrays = build_arrays(records, specs)
    images = {record["gid"]: {"meta": [record["meta"]["design_values"].get(f"meta.{key}")
                                             for key in ("upper", "wb", "bottom")],
                                      "frames": {"0": record["views"]}}
              for record in records}
    report = balance_report(records, rejected, splits, specs, schema, args.strict_categorical_spread)
    out.mkdir(parents=True, exist_ok=True)
    (out / "schema.json").write_text(json.dumps(schema, indent=2))
    (out / "images.json").write_text(json.dumps(images, indent=2))
    (out / "splits.json").write_text(json.dumps(splits, indent=2))
    (out / "balance_report.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(out / "targets.npz", **arrays)
    print(f"Prepared {len(records)} samples -> {out}")
    print(f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} ready={report['ready']}")
    print(f"unsupported/never-active heads: {report['unsupported_or_never_active_schema_paths']}")
    if not report["ready"] and not args.allow_incomplete:
        raise SystemExit("Export written, but balance_report.json is not training-ready. Regenerate rejected/deficit samples or pass --allow-incomplete only for diagnostics.")


if __name__ == "__main__":
    main()