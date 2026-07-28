#!/usr/bin/env python3
"""Convert a balanced GarmentCodeSMPLX tree into ImageToGarment prepared data.

The converter accepts only DESIGN_TARGET_VERSION=1 samples, preserves body-level
split isolation, masks topology-inactive parameters, and writes an auditable
balance_report.json. Existing source samples are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
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
    """Return three independent one-pose inputs for this garment."""
    poses = [folder / f"{name}_pose{index}.png" for index in (1, 2, 3)]
    if not all(readable_png(path) for path in poses):
        return None
    return [str(path.resolve()) for path in poses]


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
            try:
                garment_pkl_present = (
                    pkl_path.is_file() and pkl_path.stat().st_size > 0
                )
            except OSError:
                garment_pkl_present = False
            reason = None
            try:
                meta = json.loads(meta_path.read_text())
            except Exception as exc:
                meta = None
                reason = f"metadata unreadable: {exc}"
            if meta is not None:
                if meta.get("design_target_version") != TARGET_VERSION:
                    reason = "legacy sample: missing design_target_version=1"

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
                reason = "missing/corrupt pose1, pose2, or pose3 image"
            if reason is not None:
                rejected.append({"split": source_split, "folder": str(folder), "reason": reason})
                continue
            records.append({
                "gid": f"{source_split}:{name}", "name": name,
                "source_split": source_split, "folder": folder, "meta": meta,
                "views": views, "design_hash": design_hash(meta),
                "optional_artifacts": {
                    "garment_pkl_present": garment_pkl_present,
                    "garment_pkl_path": str(pkl_path),
                },
            })
    return records, rejected


def _select_validation_bodies(
    train_records: list[dict[str, Any]],
    val_fraction: float,
    specs: OrderedDict[str, dict[str, Any]],
    seed: int,
) -> tuple[set[str], dict[str, Any]]:
    """Choose a representative validation subset while keeping bodies intact."""
    by_body: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in train_records:
        by_body[record["meta"]["body_name"]].append(record)
    bodies = sorted(by_body)
    if len(bodies) < 2 or val_fraction <= 0:
        return set(), {
            "mode": "disabled", "selected_bodies": [], "seed": seed,
            "val_fraction": val_fraction, "candidate_subsets_evaluated": 0,
        }

    n_val = min(max(int(round(len(bodies) * val_fraction)), 1), len(bodies) - 1)
    categorical_paths = {
        path for path, spec in specs.items()
        if spec["type"] in ("bool", "select", "select_null")
    }
    profiles = {}
    global_categories: Counter[str] = Counter()
    global_fields: dict[str, Counter[str]] = defaultdict(Counter)
    for body, body_records in by_body.items():
        categories: Counter[str] = Counter()
        fields: dict[str, Counter[str]] = defaultdict(Counter)
        for record in body_records:
            meta = record["meta"]
            categories[meta["category"]] += 1
            for path in set(meta["design_active_paths"]) & categorical_paths:
                fields[path][value_key(meta["design_values"][path])] += 1
        profiles[body] = {"samples": len(body_records), "categories": categories, "fields": fields}
        global_categories.update(categories)
        for path, counts in fields.items():
            global_fields[path].update(counts)

    target_samples = len(train_records) * val_fraction
    evaluated = 0

    def score(candidate: tuple[str, ...]):
        nonlocal evaluated
        evaluated += 1
        samples = sum(profiles[body]["samples"] for body in candidate)
        categories: Counter[str] = Counter()
        fields: dict[str, Counter[str]] = defaultdict(Counter)
        for body in candidate:
            categories.update(profiles[body]["categories"])
            for path, counts in profiles[body]["fields"].items():
                fields[path].update(counts)

        sample_error = abs(samples - target_samples) / max(target_samples, 1.0)
        category_tv = 0.5 * sum(
            abs(categories[name] / max(samples, 1)
                - global_categories[name] / len(train_records))
            for name in global_categories
        )
        missing_fraction = (
            sum(categories[name] == 0 for name in global_categories)
            / max(len(global_categories), 1)
        )
        field_tvs = []
        for path, global_counts in global_fields.items():
            selected_counts = fields[path]
            selected_total = sum(selected_counts.values())
            global_total = sum(global_counts.values())
            if selected_total == 0:
                field_tvs.append(1.0)
            else:
                field_tvs.append(0.5 * sum(
                    abs(selected_counts[label] / selected_total
                        - global_counts[label] / global_total)
                    for label in global_counts
                ))
        categorical_tv = float(np.mean(field_tvs)) if field_tvs else 0.0
        components = {
            "sample_relative_error": float(sample_error),
            "garment_category_tv": float(category_tv),
            "missing_garment_category_fraction": float(missing_fraction),
            "categorical_label_tv": categorical_tv,
            "selected_samples": samples,
            "target_samples": float(target_samples),
        }
        total = sample_error + 2.0 * category_tv + 0.5 * missing_fraction + categorical_tv
        return (round(float(total), 12), candidate), float(total), components

    combination_count = math.comb(len(bodies), n_val)
    if combination_count <= 200_000:
        mode = "exact_stratified_body_holdout"
        best = min(
            (score(candidate) for candidate in itertools.combinations(bodies, n_val)),
            key=lambda result: result[0],
        )
        _, best_score, components = best
        selected = best[0][1]
    else:
        mode = "greedy_stratified_body_holdout"
        selected: tuple[str, ...] = ()
        while len(selected) < n_val:
            best = min(
                (score(tuple(sorted((*selected, body))))
                 for body in bodies if body not in selected),
                key=lambda result: result[0],
            )
            selected = best[0][1]
        improved = True
        while improved:
            improved = False
            current = score(selected)
            for old_body in selected:
                for new_body in bodies:
                    if new_body in selected:
                        continue
                    candidate = tuple(sorted((set(selected) - {old_body}) | {new_body}))
                    replacement = score(candidate)
                    if replacement[0] < current[0]:
                        selected, current, improved = candidate, replacement, True
                        break
                if improved:
                    break
        _, best_score, components = current

    return set(selected), {
        "mode": mode, "selected_bodies": list(selected), "seed": seed,
        "val_fraction": val_fraction, "body_count": len(bodies),
        "selected_body_count": n_val, "candidate_subsets_total": combination_count,
        "candidate_subsets_evaluated": evaluated, "score": best_score,
        "score_weights": {
            "sample_relative_error": 1.0, "garment_category_tv": 2.0,
            "missing_garment_category_fraction": 0.5, "categorical_label_tv": 1.0,
        },
        "components": components,
    }


def split_by_body(
    records: list[dict[str, Any]],
    val_fraction: float,
    test_fraction: float | None,
    seed: int,
    specs: OrderedDict[str, dict[str, Any]],
):
    """Create deterministic body-disjoint splits with a stratified holdout."""
    assigned: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    rng = np.random.default_rng(seed)
    source_val_bodies = sorted({
        record["meta"]["body_name"] for record in records
        if record["source_split"] == "val"
    })
    if source_val_bodies:
        validation_bodies = set()
        selection_info = {
            "mode": "source_validation", "selected_bodies": source_val_bodies,
            "seed": seed, "val_fraction": None, "candidate_subsets_evaluated": 0,
        }
    else:
        validation_bodies, selection_info = _select_validation_bodies(
            [record for record in records if record["source_split"] == "train"],
            val_fraction, specs, seed,
        )

    test_bodies: set[str] = set()
    if test_fraction is not None:
        candidate_bodies = source_val_bodies.copy()
        rng.shuffle(candidate_bodies)
        n_test = int(round(len(candidate_bodies) * test_fraction))
        if len(candidate_bodies) >= 2 and test_fraction > 0:
            n_test = min(max(n_test, 1), len(candidate_bodies) - 1)
        test_bodies = set(candidate_bodies[:n_test])

    for record in records:
        source = record["source_split"]
        body_name = record["meta"]["body_name"]
        if source == "train":
            split = "val" if body_name in validation_bodies else "train"
        elif source == "test" or body_name in test_bodies:
            split = "test"
        else:
            split = "val"
        record["split"] = split
        assigned[split].append(record["gid"])
    return ({key: sorted(value) for key, value in assigned.items()}, selection_info)

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

    # Preserve raw numeric ground truth. GarmentDataset normalizes const slots,
    # and the model.py adapter expands its sigmoid output range when observed
    # body-adjusted values exceed the schema's sampling range.
    const_slots = OrderedDict((path, index) for index, path in enumerate(regression))
    const_ranges = OrderedDict()
    for path in regression:
        observed_values = [
            float(record["meta"]["design_values"][path])
            for record in records
            if path in record["meta"]["design_active_paths"]
        ]
        schema_lo, schema_hi = map(float, specs[path]["range"])
        const_ranges[path] = [
            min([schema_lo, *observed_values]),
            max([schema_hi, *observed_values]),
        ]

    cat_paths = list(cat_vocab)
    gids = [record["gid"] for record in records]
    y_cont = np.zeros((len(records), 0), dtype=np.float32)
    mask = np.zeros_like(y_cont)
    y_const = np.zeros((len(records), len(regression)), dtype=np.float32)
    const_mask = np.zeros_like(y_const)
    y_cat = np.full((len(records), len(categorical)), -1, dtype=np.int64)
    for row, record in enumerate(records):
        values = record["meta"]["design_values"]
        active = set(record["meta"]["design_active_paths"])
        for path, col in const_slots.items():
            if path not in active:
                continue
            y_const[row, col] = float(values[path])
            const_mask[row, col] = 1.0
        for col, path in enumerate(cat_paths):
            if path in active:
                y_cat[row, col] = next(i for i, value in enumerate(cat_vocab[path])
                                            if value_key(value) == value_key(values[path]))
    schema = {
        "n_cont": 0, "n_const": len(regression), "n_cat": len(categorical),
        "cont_slots": {}, "const_slots": const_slots, "const_ranges": const_ranges,
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
        for path in schema["const_slots"]:
            vals = [float(by_gid[gid]["meta"]["design_values"][path]) for gid in gids
                    if path in by_gid[gid]["meta"]["design_active_paths"]]
            lo, hi = map(float, schema["const_ranges"][path])
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
    unsupported = sorted(
        set(specs) - set(schema["cont_slots"]) - set(schema["const_slots"])
        - set(schema["cat_vocab"])
    )
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
    parser.add_argument(
        "--val-fraction-of-train-bodies",
        type=float,
        default=0.1,
        help="body-level validation holdout used only when no source val samples exist",
    )
    parser.add_argument(
        "--test-fraction-of-val-bodies",
        type=float,
        help="optional explicit fraction of validation bodies to move to test; omitted preserves source splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-categorical-spread", type=int, default=1)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if not 0 < args.val_fraction_of_train_bodies < 1:
        parser.error("--val-fraction-of-train-bodies must be in (0,1)")
    if (
        args.test_fraction_of_val_bodies is not None
        and not 0 <= args.test_fraction_of_val_bodies < 1
    ):
        parser.error("--test-fraction-of-val-bodies must be in [0,1)")
    root = Path(args.dataset_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    document = yaml.safe_load(Path(args.schema).read_text())
    specs = flatten(document["design"])
    records, rejected = scan(root, specs)
    if not records:
        raise RuntimeError(f"No version-{TARGET_VERSION} training-ready samples found under {root}")
    splits, split_selection = split_by_body(
        records,
        args.val_fraction_of_train_bodies,
        args.test_fraction_of_val_bodies,
        args.seed,
        specs,
    )
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
    (out / "split_selection.json").write_text(json.dumps(split_selection, indent=2))
    (out / "balance_report.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(out / "targets.npz", **arrays)
    print(f"Prepared {len(records)} samples -> {out}")
    print(f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} ready={report['ready']}")
    print(f"unsupported/never-active heads: {report['unsupported_or_never_active_schema_paths']}")
    if not report["ready"] and not args.allow_incomplete:
        raise SystemExit("Export written, but balance_report.json is not training-ready. Regenerate rejected/deficit samples or pass --allow-incomplete only for diagnostics.")


if __name__ == "__main__":
    main()
