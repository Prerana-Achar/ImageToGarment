#!/usr/bin/env python3
"""Select and compile a balanced, body-disjoint GarmentCodeSMPLX dataset.

The compiler never copies or modifies raw samples.  It scans all completed
sample folders, pools their bodies, chooses a representative validation body
holdout, selects an equal number of garments from every public category, and
prefers records that improve categorical-label and numeric-bin coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from prepare_garmentcode_smplx import (
    CATEGORIES,
    build_arrays,
    flatten,
    scan,
    value_key,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--schema",
        default=str(root / "GarmentCodeRC/assets/design_params/default_new.yaml"),
    )
    parser.add_argument("--val-body-fraction", type=float, default=0.15)
    parser.add_argument(
        "--train-per-category",
        type=int,
        help="optional cap; default is the largest equal count available",
    )
    parser.add_argument(
        "--val-per-category",
        type=int,
        help="optional cap; default is the largest equal count available",
    )
    parser.add_argument("--numeric-bins", type=int, default=10)
    parser.add_argument("--search-restarts", type=int, default=80)
    parser.add_argument("--search-passes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-missing-category",
        action="store_true",
        help="write a diagnostic export instead of failing if a category is absent",
    )
    return parser.parse_args()


def record_body(record: dict[str, Any]) -> str:
    return str(record["meta"]["body_name"])


def record_category(record: dict[str, Any]) -> str:
    return str(record["meta"]["category"])


def category_counts(records: Iterable[dict[str, Any]]) -> Counter[str]:
    return Counter(record_category(record) for record in records)


def categorical_paths(specs: OrderedDict[str, dict[str, Any]]) -> set[str]:
    return {
        path
        for path, specification in specs.items()
        if specification["type"] in {"bool", "select", "select_null"}
    }


def body_profiles(
    records: list[dict[str, Any]], categorical: set[str]
) -> tuple[dict[str, dict[str, Any]], Counter[str], dict[str, Counter[str]]]:
    profiles: dict[str, dict[str, Any]] = {}
    global_categories: Counter[str] = Counter()
    global_labels: dict[str, Counter[str]] = defaultdict(Counter)
    by_body: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_body[record_body(record)].append(record)
    for body, body_records in by_body.items():
        categories = category_counts(body_records)
        labels: dict[str, Counter[str]] = defaultdict(Counter)
        for record in body_records:
            active = set(record["meta"]["design_active_paths"])
            values = record["meta"]["design_values"]
            for path in categorical & active:
                labels[path][value_key(values[path])] += 1
        profiles[body] = {
            "records": body_records,
            "samples": len(body_records),
            "categories": categories,
            "labels": labels,
        }
        global_categories.update(categories)
        for path, counts in labels.items():
            global_labels[path].update(counts)
    return profiles, global_categories, global_labels


def holdout_score(
    selected: tuple[str, ...],
    profiles: dict[str, dict[str, Any]],
    global_categories: Counter[str],
    global_labels: dict[str, Counter[str]],
    target_samples: float,
) -> tuple[Any, ...]:
    validation_categories: Counter[str] = Counter()
    validation_labels: dict[str, Counter[str]] = defaultdict(Counter)
    validation_samples = 0
    for body in selected:
        profile = profiles[body]
        validation_samples += profile["samples"]
        validation_categories.update(profile["categories"])
        for path, counts in profile["labels"].items():
            validation_labels[path].update(counts)

    train_categories = global_categories - validation_categories
    missing = sum(validation_categories[name] == 0 for name in CATEGORIES)
    missing += sum(train_categories[name] == 0 for name in CATEGORIES)
    validation_values = [validation_categories[name] for name in CATEGORIES]
    train_values = [train_categories[name] for name in CATEGORIES]
    minimum_validation = min(validation_values)
    minimum_train = min(train_values)
    validation_spread = max(validation_values) - minimum_validation
    train_spread = max(train_values) - minimum_train
    sample_error = abs(validation_samples - target_samples) / max(target_samples, 1.0)

    label_distances = []
    for path, global_counts in global_labels.items():
        selected_counts = validation_labels[path]
        selected_total = sum(selected_counts.values())
        global_total = sum(global_counts.values())
        if selected_total == 0:
            label_distances.append(1.0)
            continue
        label_distances.append(
            0.5
            * sum(
                abs(
                    selected_counts[label] / selected_total
                    - global_counts[label] / global_total
                )
                for label in global_counts
            )
        )
    label_distance = float(np.mean(label_distances)) if label_distances else 0.0
    return (
        missing,
        -minimum_validation,
        validation_spread,
        -minimum_train,
        train_spread,
        round(sample_error, 8),
        round(label_distance, 8),
        selected,
    )


def choose_validation_bodies(
    records: list[dict[str, Any]],
    specs: OrderedDict[str, dict[str, Any]],
    fraction: float,
    seed: int,
    restarts: int,
    passes: int,
) -> tuple[set[str], dict[str, Any]]:
    categorical = categorical_paths(specs)
    profiles, global_categories, global_labels = body_profiles(records, categorical)
    bodies = sorted(profiles)
    if len(bodies) < 2:
        raise RuntimeError("At least two complete bodies are required")
    count = min(max(round(len(bodies) * fraction), 1), len(bodies) - 1)
    target_samples = len(records) * fraction
    rng = random.Random(seed)
    best_bodies: tuple[str, ...] | None = None
    best_score: tuple[Any, ...] | None = None
    evaluations = 0

    starts = [tuple(bodies[:count])]
    starts.extend(tuple(sorted(rng.sample(bodies, count))) for _ in range(restarts))
    for start in starts:
        selected = start
        score = holdout_score(
            selected, profiles, global_categories, global_labels, target_samples
        )
        evaluations += 1
        for _ in range(passes):
            selected_set = set(selected)
            candidates = []
            for outgoing in selected:
                for incoming in bodies:
                    if incoming in selected_set:
                        continue
                    proposal = tuple(sorted((selected_set - {outgoing}) | {incoming}))
                    proposal_score = holdout_score(
                        proposal,
                        profiles,
                        global_categories,
                        global_labels,
                        target_samples,
                    )
                    evaluations += 1
                    candidates.append((proposal_score, proposal))
            candidate_score, candidate = min(candidates, key=lambda item: item[0])
            if candidate_score >= score:
                break
            score, selected = candidate_score, candidate
        if best_score is None or score < best_score:
            best_score, best_bodies = score, selected
    assert best_bodies is not None and best_score is not None
    return set(best_bodies), {
        "mode": "multi_start_body_swap_search",
        "body_count": len(bodies),
        "validation_body_count": count,
        "validation_bodies": list(best_bodies),
        "target_fraction": fraction,
        "target_samples": target_samples,
        "score": list(best_score[:-1]),
        "evaluations": evaluations,
        "restarts": restarts,
        "passes": passes,
        "seed": seed,
    }


def numeric_bin(value: Any, specification: dict[str, Any], bins: int) -> int:
    low, high = map(float, specification["range"])
    if high <= low:
        return 0
    normalized = (float(value) - low) / (high - low)
    return min(max(int(math.floor(normalized * bins)), 0), bins - 1)


def coverage_tokens(
    record: dict[str, Any],
    specs: OrderedDict[str, dict[str, Any]],
    bins: int,
) -> list[tuple[str, str]]:
    active = set(record["meta"]["design_active_paths"])
    values = record["meta"]["design_values"]
    tokens = []
    for path in sorted(active):
        specification = specs[path]
        if specification["type"] in {"float", "int"}:
            label = f"bin:{numeric_bin(values[path], specification, bins)}"
        else:
            label = f"value:{value_key(values[path])}"
        tokens.append((path, label))
    return tokens


def select_diverse(
    candidates: list[dict[str, Any]],
    count: int,
    specs: OrderedDict[str, dict[str, Any]],
    bins: int,
    seed: int,
) -> list[dict[str, Any]]:
    if count >= len(candidates):
        return sorted(candidates, key=lambda record: record["gid"])
    rng = random.Random(seed)
    candidates = list(candidates)
    rng.shuffle(candidates)
    token_cache = {record["gid"]: coverage_tokens(record, specs, bins) for record in candidates}
    selected = []
    token_counts: Counter[tuple[str, str]] = Counter()
    path_counts: Counter[str] = Counter()
    remaining = candidates
    while len(selected) < count:
        def gain(record: dict[str, Any]) -> tuple[float, str]:
            tokens = token_cache[record["gid"]]
            if not tokens:
                return (0.0, record["gid"])
            score = sum(
                1.0
                / ((1.0 + token_counts[token]) * math.sqrt(1.0 + path_counts[token[0]]))
                for token in tokens
            ) / len(tokens)
            return (-score, record["gid"])

        chosen = min(remaining, key=gain)
        selected.append(chosen)
        for token in token_cache[chosen["gid"]]:
            token_counts[token] += 1
            path_counts[token[0]] += 1
        remaining = [record for record in remaining if record is not chosen]
    return sorted(selected, key=lambda record: record["gid"])


def balanced_subset(
    records: list[dict[str, Any]],
    requested_per_category: int | None,
    specs: OrderedDict[str, dict[str, Any]],
    bins: int,
    seed: int,
    allow_missing: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        category: [] for category in CATEGORIES
    }
    for record in records:
        grouped[record_category(record)].append(record)
    available = {category: len(grouped[category]) for category in CATEGORIES}
    missing = [category for category, count in available.items() if count == 0]
    if missing and not allow_missing:
        raise RuntimeError(
            "Cannot create a balanced split; no complete samples for: " + ", ".join(missing)
        )
    nonzero = [count for count in available.values() if count > 0]
    maximum = min(nonzero) if nonzero else 0
    quota = maximum if requested_per_category is None else requested_per_category
    if quota <= 0:
        raise RuntimeError("The per-category quota must be positive")
    deficits = {
        category: quota - count
        for category, count in available.items()
        if count < quota
    }
    if deficits and not allow_missing:
        details = ", ".join(f"{category}: need {needed}" for category, needed in deficits.items())
        raise RuntimeError(f"Requested quota is not available ({details})")
    selected = []
    for category_index, category in enumerate(CATEGORIES):
        candidates = grouped[category]
        if not candidates:
            continue
        selected.extend(
            select_diverse(
                candidates,
                min(quota, len(candidates)),
                specs,
                bins,
                seed + 1009 * category_index,
            )
        )
    return selected, {
        "available": available,
        "quota": quota,
        "selected": dict(category_counts(selected)),
        "deficits": deficits,
    }


def label_audit(
    records: list[dict[str, Any]], specs: OrderedDict[str, dict[str, Any]]
) -> dict[str, Any]:
    categorical = categorical_paths(specs)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    active_counts: Counter[str] = Counter()
    for record in records:
        active = set(record["meta"]["design_active_paths"])
        values = record["meta"]["design_values"]
        for path in categorical & active:
            counts[path][value_key(values[path])] += 1
            active_counts[path] += 1
    return {
        path: {
            "active": active_counts[path],
            "counts": dict(sorted(path_counts.items())),
            "spread": max(path_counts.values()) - min(path_counts.values()),
        }
        for path, path_counts in sorted(counts.items())
        if path_counts
    }


def compile_dataset(
    selected_train: list[dict[str, Any]],
    selected_val: list[dict[str, Any]],
    specs: OrderedDict[str, dict[str, Any]],
    out: Path,
    audit: dict[str, Any],
) -> None:
    selected = [*selected_train, *selected_val]
    for record in selected_train:
        record["split"] = "train"
    for record in selected_val:
        record["split"] = "val"
    schema, arrays = build_arrays(selected, specs)
    images = {
        record["gid"]: {
            "meta": [
                record["meta"]["design_values"].get(f"meta.{key}")
                for key in ("upper", "wb", "bottom")
            ],
            "category": record_category(record),
            "body_name": record_body(record),
            "frames": {"0": record["views"]},
        }
        for record in selected
    }
    splits = {
        "train": sorted(record["gid"] for record in selected_train),
        "val": sorted(record["gid"] for record in selected_val),
        "test": [],
    }
    manifest = {
        record["gid"]: {
            "source_folder": str(record["folder"]),
            "source_split": record["source_split"],
            "compiled_split": record["split"],
            "body_name": record_body(record),
            "category": record_category(record),
            "design_hash": record["design_hash"],
        }
        for record in selected
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "schema.json").write_text(json.dumps(schema, indent=2))
    (out / "images.json").write_text(json.dumps(images, indent=2))
    (out / "splits.json").write_text(json.dumps(splits, indent=2))
    (out / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "balance_report.json").write_text(json.dumps(audit, indent=2))
    np.savez_compressed(out / "targets.npz", **arrays)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_body_fraction < 1.0:
        raise SystemExit("--val-body-fraction must be in (0, 1)")
    if args.numeric_bins < 2:
        raise SystemExit("--numeric-bins must be at least 2")
    schema_path = Path(args.schema).expanduser().resolve()
    schema_document = yaml.safe_load(schema_path.read_text())
    specs = flatten(schema_document["design"])
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    records, rejected = scan(dataset_root, specs)
    if not records:
        raise RuntimeError(f"No complete version-1 records under {dataset_root}")

    validation_bodies, holdout = choose_validation_bodies(
        records,
        specs,
        args.val_body_fraction,
        args.seed,
        args.search_restarts,
        args.search_passes,
    )
    train_pool = [record for record in records if record_body(record) not in validation_bodies]
    val_pool = [record for record in records if record_body(record) in validation_bodies]
    selected_train, train_selection = balanced_subset(
        train_pool,
        args.train_per_category,
        specs,
        args.numeric_bins,
        args.seed,
        args.allow_missing_category,
    )
    selected_val, val_selection = balanced_subset(
        val_pool,
        args.val_per_category,
        specs,
        args.numeric_bins,
        args.seed + 1,
        args.allow_missing_category,
    )
    train_bodies = {record_body(record) for record in selected_train}
    val_bodies = {record_body(record) for record in selected_val}
    body_overlap = sorted(train_bodies & val_bodies)
    selected_ids = {record["gid"] for record in (*selected_train, *selected_val)}
    audit = {
        "ready": (
            not body_overlap
            and bool(selected_train)
            and bool(selected_val)
            and len(set(category_counts(selected_train).values())) == 1
            and len(set(category_counts(selected_val).values())) == 1
            and not train_selection["deficits"]
            and not val_selection["deficits"]
        ),
        "compiler": "balanced_garmentcode_smplx/v1",
        "dataset_root": str(dataset_root),
        "schema": str(schema_path),
        "schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
        "complete_records_scanned": len(records),
        "rejected_incomplete_records": len(rejected),
        "selected_records": len(selected_ids),
        "discarded_complete_records": len(records) - len(selected_ids),
        "holdout": holdout,
        "train": {
            "samples": len(selected_train),
            "bodies": len(train_bodies),
            **train_selection,
            "categorical_labels": label_audit(selected_train, specs),
        },
        "val": {
            "samples": len(selected_val),
            "bodies": len(val_bodies),
            **val_selection,
            "categorical_labels": label_audit(selected_val, specs),
        },
        "body_overlap": body_overlap,
        "rejected": rejected,
    }
    out = Path(args.out).expanduser().resolve()
    compile_dataset(selected_train, selected_val, specs, out, audit)
    print(f"complete={len(records)} rejected={len(rejected)}")
    print(
        f"train={len(selected_train)} ({train_selection['quota']}/category) "
        f"val={len(selected_val)} ({val_selection['quota']}/category)"
    )
    print(f"bodies train={len(train_bodies)} val={len(val_bodies)} overlap={len(body_overlap)}")
    print(f"wrote {out} ready={audit['ready']}")
    if not audit["ready"] and not args.allow_missing_category:
        raise SystemExit("Export is not balance-ready; inspect balance_report.json")


if __name__ == "__main__":
    main()

