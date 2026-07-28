#!/usr/bin/env python3
"""Compile all complete garments and balance them in model-output space.

Unlike category-quota selection, this compiler keeps every complete garment.
It balances the body-disjoint holdout and training exposure using the actual
prediction targets: active output heads, categorical head/classes, and binned
numeric head values.
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

from prepare_garmentcode_smplx import CATEGORIES, build_arrays, flatten, scan, value_key


Token = tuple[str, str, str]


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
    parser.add_argument("--numeric-bins", type=int, default=10)
    parser.add_argument("--extra-sampling-fraction", type=float, default=0.5)
    parser.add_argument("--max-sample-weight", type=float, default=5.0)
    parser.add_argument("--min-active-head-support", type=int, default=20)
    parser.add_argument("--min-categorical-class-support", type=int, default=5)
    parser.add_argument("--min-train-categorical-support", type=int, default=2)
    parser.add_argument("--min-numeric-bin-support", type=int, default=3)
    parser.add_argument("--search-restarts", type=int, default=40)
    parser.add_argument("--search-passes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def record_body(record: dict[str, Any]) -> str:
    return str(record["meta"]["body_name"])


def record_category(record: dict[str, Any]) -> str:
    return str(record["meta"]["category"])


def numeric_bin(value: Any, specification: dict[str, Any], bins: int) -> int:
    low, high = map(float, specification["range"])
    if high <= low:
        return 0
    normalized = (float(value) - low) / (high - low)
    return min(max(int(math.floor(normalized * bins)), 0), bins - 1)


def output_tokens(
    record: dict[str, Any], specs: OrderedDict[str, dict[str, Any]], bins: int
) -> tuple[Token, ...]:
    active = set(record["meta"]["design_active_paths"])
    values = record["meta"]["design_values"]
    tokens: list[Token] = []
    for path in sorted(active):
        specification = specs[path]
        tokens.append((path, "active", "1"))
        if specification["type"] in {"float", "int"}:
            tokens.append(
                (path, "numeric_bin", str(numeric_bin(values[path], specification, bins)))
            )
        else:
            tokens.append((path, "categorical", value_key(values[path])))
    return tuple(tokens)


def token_profiles(
    records: list[dict[str, Any]], specs: OrderedDict[str, dict[str, Any]], bins: int
) -> tuple[
    dict[str, dict[str, Any]],
    Counter[Token],
    dict[str, set[Token]],
    dict[str, tuple[Token, ...]],
]:
    by_body: dict[str, list[dict[str, Any]]] = defaultdict(list)
    token_cache = {}
    global_counts: Counter[Token] = Counter()
    tokens_by_path: dict[str, set[Token]] = defaultdict(set)
    for record in records:
        tokens = output_tokens(record, specs, bins)
        token_cache[record["gid"]] = tokens
        global_counts.update(tokens)
        for token in tokens:
            tokens_by_path[token[0]].add(token)
        by_body[record_body(record)].append(record)
    profiles = {}
    for body, body_records in by_body.items():
        counts: Counter[Token] = Counter()
        for record in body_records:
            counts.update(token_cache[record["gid"]])
        profiles[body] = {
            "records": body_records,
            "samples": len(body_records),
            "tokens": counts,
        }
    return profiles, global_counts, tokens_by_path, token_cache


def split_score(
    selected: tuple[str, ...],
    profiles: dict[str, dict[str, Any]],
    global_counts: Counter[Token],
    tokens_by_path: dict[str, set[Token]],
    target_samples: float,
    min_train_categorical_support: int,
    global_categories: set[str],
) -> tuple[Any, ...]:
    selected_counts: Counter[Token] = Counter()
    selected_samples = 0
    for body in selected:
        selected_samples += profiles[body]["samples"]
        selected_counts.update(profiles[body]["tokens"])
    train_counts = global_counts - selected_counts
    selected_set = set(selected)
    train_categories = {
        record_category(record)
        for body, profile in profiles.items()
        if body not in selected_set
        for record in profile["records"]
    }
    missing_train_categories = len(global_categories - train_categories)
    unseen_train_output_tokens = sum(
        count > 0
        and token[1] != "numeric_bin"
        and train_counts[token] == 0
        for token, count in global_counts.items()
    )
    validation_categorical_tokens = [
        token
        for token, count in selected_counts.items()
        if token[1] == "categorical" and count > 0
    ]
    unseen_validation_categorical = sum(
        train_counts[token] == 0 for token in validation_categorical_tokens
    )
    weak_validation_categorical = sum(
        0 < train_counts[token] < min_train_categorical_support
        for token in validation_categorical_tokens
    )

    missing_val_heads = 0
    missing_train_heads = 0
    path_distances = []
    missing_token_fraction = []
    for path, tokens in tokens_by_path.items():
        global_total = sum(global_counts[token] for token in tokens)
        val_total = sum(selected_counts[token] for token in tokens)
        train_total = sum(train_counts[token] for token in tokens)
        if val_total == 0:
            missing_val_heads += 1
            continue
        if train_total == 0:
            missing_train_heads += 1
            continue
        path_distances.append(
            0.5
            * sum(
                abs(
                    selected_counts[token] / val_total
                    - global_counts[token] / global_total
                )
                for token in tokens
            )
        )
        missing_token_fraction.append(
            sum(selected_counts[token] == 0 for token in tokens) / len(tokens)
        )
    sample_error = abs(selected_samples - target_samples) / max(target_samples, 1.0)
    distribution_distance = float(np.mean(path_distances)) if path_distances else 1.0
    missing_fraction = float(np.mean(missing_token_fraction)) if missing_token_fraction else 1.0
    return (
        missing_train_categories,
        unseen_train_output_tokens,
        unseen_validation_categorical,
        weak_validation_categorical,
        missing_train_heads + missing_val_heads,
        round(sample_error, 8),
        round(distribution_distance, 8),
        round(missing_fraction, 8),
        selected,
    )


def choose_validation_bodies(
    records: list[dict[str, Any]],
    specs: OrderedDict[str, dict[str, Any]],
    bins: int,
    fraction: float,
    min_train_categorical_support: int,
    seed: int,
    restarts: int,
    passes: int,
) -> tuple[set[str], dict[str, Any], dict[str, tuple[Token, ...]]]:
    profiles, global_counts, tokens_by_path, token_cache = token_profiles(
        records, specs, bins
    )
    global_categories = {record_category(record) for record in records}
    bodies = sorted(profiles)
    if len(bodies) < 2:
        raise RuntimeError("At least two complete bodies are required")
    validation_count = min(max(round(len(bodies) * fraction), 1), len(bodies) - 1)
    target_samples = len(records) * fraction
    rng = random.Random(seed)
    starts = [tuple(bodies[:validation_count])]
    starts.extend(
        tuple(sorted(rng.sample(bodies, validation_count))) for _ in range(restarts)
    )
    best_bodies = None
    best_score = None
    evaluations = 0
    for start in starts:
        selected = start
        score = split_score(
            selected,
            profiles,
            global_counts,
            tokens_by_path,
            target_samples,
            min_train_categorical_support,
            global_categories,
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
                    proposal_score = split_score(
                        proposal,
                        profiles,
                        global_counts,
                        tokens_by_path,
                        target_samples,
                        min_train_categorical_support,
                        global_categories,
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
        "mode": "output_token_body_swap_search",
        "validation_bodies": list(best_bodies),
        "body_count": len(bodies),
        "validation_body_count": validation_count,
        "target_fraction": fraction,
        "min_train_categorical_support": min_train_categorical_support,
        "missing_train_garment_categories": best_score[0],
        "unseen_train_output_tokens": best_score[1],
        "unseen_validation_categorical_classes": best_score[2],
        "weakly_supported_validation_categorical_classes": best_score[3],
        "score": list(best_score[:-1]),
        "score_fields": [
            "missing_train_garment_categories",
            "unseen_train_output_tokens",
            "unseen_validation_categorical_classes",
            "weakly_supported_validation_categorical_classes",
            "missing_output_heads",
            "sample_fraction_error",
            "mean_output_distribution_distance",
            "mean_missing_output_token_fraction",
        ],
        "evaluations": evaluations,
        "seed": seed,
    }, token_cache


def sampling_weights(
    train_records: list[dict[str, Any]],
    token_cache: dict[str, tuple[Token, ...]],
    maximum: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    frequencies: Counter[Token] = Counter()
    for record in train_records:
        frequencies.update(token_cache[record["gid"]])
    raw_weights = {}
    token_total = max(len(train_records), 1)
    for record in train_records:
        rarity = sorted(
            (
                math.sqrt(token_total / max(frequencies[token], 1))
                for token in token_cache[record["gid"]]
            ),
            reverse=True,
        )
        focus_count = max(1, math.ceil(len(rarity) * 0.25))
        raw_weights[record["gid"]] = sum(rarity[:focus_count]) / focus_count
    mean_weight = sum(raw_weights.values()) / max(len(raw_weights), 1)
    normalized = {
        gid: min(max(value / max(mean_weight, 1e-8), 0.25), maximum)
        for gid, value in raw_weights.items()
    }
    return normalized, {
        "minimum": min(normalized.values()),
        "maximum": max(normalized.values()),
        "mean": sum(normalized.values()) / len(normalized),
        "token_count": len(frequencies),
    }


def balanced_sampling_plan(
    train_records: list[dict[str, Any]],
    token_cache: dict[str, tuple[Token, ...]],
    extra_fraction: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create equal-count anchor groups over every model-relevant value.

    A garment belongs to several groups because GarmentCode parameters co-occur.
    Equal anchor counts are therefore the exact, honest balancing guarantee; raw
    per-field occurrences cannot generally all be made equal by duplicating whole
    garments.
    """
    groups: dict[Token, list[str]] = defaultdict(list)
    for record in train_records:
        gid = record["gid"]
        groups[("__garment__", "garment_category", record_category(record))].append(gid)
        for token in token_cache[gid]:
            if token[1] != "numeric_bin":
                groups[token].append(gid)
    ordered = [
        {
            "token": list(token),
            "garment_ids": sorted(set(gids)),
            "raw_unique_garments": len(set(gids)),
        }
        for token, gids in sorted(groups.items())
    ]
    if not ordered:
        raise RuntimeError("No training output groups were generated")
    requested = max(
        math.ceil(len(train_records) * (1.0 + extra_fraction)),
        len(ordered),
    )
    quota = max(1, math.ceil(requested / len(ordered)))
    epoch_items = quota * len(ordered)
    raw_counts = [group["raw_unique_garments"] for group in ordered]
    plan = {
        "format": "equal_output_token_anchors/v1",
        "definition": (
            "Every observed garment category, active head, and categorical value "
            "is selected as an anchor exactly the same number "
            "of times per epoch. Whole-garment co-occurrence can add incidental "
            "exposure to other values."
        ),
        "group_count": len(ordered),
        "anchor_draws_per_group_per_epoch": quota,
        "items_per_epoch": epoch_items,
        "groups": ordered,
    }
    audit = {
        "guarantee": "equal anchor draws per observed output token",
        "group_count": len(ordered),
        "anchor_draws_per_group_per_epoch": quota,
        "items_per_epoch": epoch_items,
        "raw_unique_garments_min": min(raw_counts),
        "raw_unique_garments_max": max(raw_counts),
        "raw_unique_garments_equal": len(set(raw_counts)) == 1,
        "low_support_groups": [
            {
                "token": group["token"],
                "observed": group["raw_unique_garments"],
                "additional_unique_garments_needed_for_5": max(
                    0, 5 - group["raw_unique_garments"]
                ),
            }
            for group in ordered
            if group["raw_unique_garments"] < 5
        ],
    }
    return plan, audit


def coverage_audit(
    all_records: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    token_cache: dict[str, tuple[Token, ...]],
) -> dict[str, Any]:
    all_tokens: Counter[Token] = Counter()
    train_tokens: Counter[Token] = Counter()
    for record in all_records:
        all_tokens.update(token_cache[record["gid"]])
    for record in train_records:
        train_tokens.update(token_cache[record["gid"]])
    all_categories = Counter(record_category(record) for record in all_records)
    train_categories = Counter(record_category(record) for record in train_records)
    missing_categories = [
        {"category": category, "dataset_count": count, "train_count": 0}
        for category, count in sorted(all_categories.items())
        if train_categories[category] == 0
    ]
    required_tokens = {
        token for token in all_tokens if token[1] != "numeric_bin"
    }
    missing_tokens = [
        {
            "path": token[0],
            "kind": token[1],
            "value": token[2],
            "dataset_count": all_tokens[token],
            "train_count": 0,
        }
        for token in sorted(required_tokens)
        if train_tokens[token] == 0
    ]
    missing_numeric_bins = [
        {
            "path": token[0],
            "bin": token[2],
            "dataset_count": all_tokens[token],
            "train_count": 0,
        }
        for token in sorted(all_tokens)
        if token[1] == "numeric_bin" and train_tokens[token] == 0
    ]
    categorical_seen = {
        token for token in all_tokens if token[1] == "categorical"
    }
    categorical_in_train = {
        token for token in train_tokens if token[1] == "categorical"
    }
    return {
        "all_observed_garment_categories_in_train": not missing_categories,
        "all_observed_output_tokens_in_train": not missing_tokens,
        "all_observed_categorical_values_in_train": categorical_seen <= categorical_in_train,
        "observed_garment_categories": len(all_categories),
        "observed_output_tokens": len(required_tokens),
        "observed_categorical_values": len(categorical_seen),
        "missing_train_garment_categories": missing_categories,
        "missing_train_output_tokens": missing_tokens,
        "numeric_bins_absent_from_train_informational": missing_numeric_bins,
        "numeric_regression_policy": (
            "Numeric bins are audited only; continuous regression is expected "
            "to interpolate and is not blocked or oversampled by bin."
        ),
        "missing_train_categorical_values": [
            {
                "path": token[0],
                "value": token[2],
                "dataset_count": all_tokens[token],
                "train_count": 0,
            }
            for token in sorted(categorical_seen - categorical_in_train)
        ],
    }

def output_requirements(
    records: list[dict[str, Any]],
    specs: OrderedDict[str, dict[str, Any]],
    bins: int,
    min_head: int,
    min_class: int,
    min_bin: int,
) -> dict[str, Any]:
    active_counts: Counter[str] = Counter()
    categorical_counts: dict[str, Counter[str]] = defaultdict(Counter)
    numeric_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for record in records:
        active = set(record["meta"]["design_active_paths"])
        values = record["meta"]["design_values"]
        for path in active:
            active_counts[path] += 1
            specification = specs[path]
            if specification["type"] in {"float", "int"}:
                numeric_counts[path][numeric_bin(values[path], specification, bins)] += 1
            else:
                categorical_counts[path][value_key(values[path])] += 1
    low_heads = {
        path: {"observed": count, "additional_needed": min_head - count}
        for path, count in active_counts.items()
        if count < min_head
    }
    low_classes = {}
    for path, counts in categorical_counts.items():
        deficits = {
            label: {"observed": count, "additional_needed": min_class - count}
            for label, count in counts.items()
            if count < min_class
        }
        if deficits:
            low_classes[path] = deficits
    low_bins = {}
    for path, counts in numeric_counts.items():
        deficits = {
            str(bin_index): {
                "observed": counts[bin_index],
                "additional_needed": min_bin - counts[bin_index],
            }
            for bin_index in range(bins)
            if counts[bin_index] < min_bin
        }
        if deficits:
            low_bins[path] = deficits
    return {
        "thresholds": {
            "active_head": min_head,
            "categorical_class": min_class,
            "numeric_bin": min_bin,
        },
        "low_active_heads": low_heads,
        "low_categorical_classes": low_classes,
        "low_numeric_bins": low_bins,
    }


def category_audit(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record_category(record) for record in records)
    return {category: counts[category] for category in CATEGORIES}


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_body_fraction < 1.0:
        raise SystemExit("--val-body-fraction must be in (0,1)")
    if args.numeric_bins < 2:
        raise SystemExit("--numeric-bins must be at least 2")
    if not 0.0 <= args.extra_sampling_fraction <= 2.0:
        raise SystemExit("--extra-sampling-fraction must be in [0,2]")
    schema_path = Path(args.schema).expanduser().resolve()
    specs = flatten(yaml.safe_load(schema_path.read_text())["design"])
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    records, rejected = scan(dataset_root, specs)
    if not records:
        raise RuntimeError(f"No complete records under {dataset_root}")

    validation_bodies, split_audit, token_cache = choose_validation_bodies(
        records,
        specs,
        args.numeric_bins,
        args.val_body_fraction,
        args.min_train_categorical_support,
        args.seed,
        args.search_restarts,
        args.search_passes,
    )
    train_records = [record for record in records if record_body(record) not in validation_bodies]
    val_records = [record for record in records if record_body(record) in validation_bodies]
    weights, weight_audit = sampling_weights(
        train_records, token_cache, args.max_sample_weight
    )
    sampling_plan, equalization_audit = balanced_sampling_plan(
        train_records, token_cache, args.extra_sampling_fraction
    )
    coverage = coverage_audit(records, train_records, token_cache)
    requirements = output_requirements(
        train_records,
        specs,
        args.numeric_bins,
        args.min_active_head_support,
        args.min_categorical_class_support,
        args.min_numeric_bin_support,
    )
    for record in train_records:
        record["split"] = "train"
    for record in val_records:
        record["split"] = "val"
    selected = [*train_records, *val_records]
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
        "train": sorted(record["gid"] for record in train_records),
        "val": sorted(record["gid"] for record in val_records),
        "test": [],
    }
    train_bodies = {record_body(record) for record in train_records}
    val_bodies = {record_body(record) for record in val_records}
    body_overlap = sorted(train_bodies & val_bodies)
    missing_optional_pkl = [
        {
            "gid": record["gid"],
            "path": record["optional_artifacts"]["garment_pkl_path"],
        }
        for record in selected
        if not record["optional_artifacts"]["garment_pkl_present"]
    ]
    audit = {
        "ready": (
            not body_overlap
            and bool(train_records)
            and bool(val_records)
            and split_audit["unseen_validation_categorical_classes"] == 0
            and coverage["all_observed_garment_categories_in_train"]
            and coverage["all_observed_output_tokens_in_train"]
            and coverage["all_observed_categorical_values_in_train"]
        ),
        "compiler": "output_balanced_garmentcode_smplx/v4",
        "balance_unit": "model output heads/classes/numeric bins",
        "complete_records_scanned": len(records),
        "selected_records": len(selected),
        "discarded_complete_records": 0,
        "optional_artifacts": {
            "garment_pkl_policy": (
                "optional; not used by image-to-parameter compilation, "
                "training, or YAML-based GarmentCode inference"
            ),
            "missing_garment_pkl_count": len(missing_optional_pkl),
            "missing_garment_pkl": missing_optional_pkl,
        },
        "rejected_incomplete_records": len(rejected),
        "train": {
            "samples": len(train_records),
            "bodies": len(train_bodies),
            "category_counts_descriptive_only": category_audit(train_records),
        },
        "val": {
            "samples": len(val_records),
            "bodies": len(val_bodies),
            "category_counts_descriptive_only": category_audit(val_records),
        },
        "body_overlap": body_overlap,
        "split_selection": split_audit,
        "coverage": coverage,
        "equalized_training_schedule": equalization_audit,
        "sampling_weight_summary_legacy": weight_audit,
        "output_requirements": requirements,
        "rejected": rejected,
        "schema": str(schema_path),
        "schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
    }
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "schema.json").write_text(json.dumps(schema, indent=2))
    (out / "images.json").write_text(json.dumps(images, indent=2))
    (out / "splits.json").write_text(json.dumps(splits, indent=2))
    (out / "balance_report.json").write_text(json.dumps(audit, indent=2))
    (out / "output_requirements.json").write_text(json.dumps(requirements, indent=2))
    (out / "balanced_sampling.json").write_text(
        json.dumps(sampling_plan, indent=2)
    )
    (out / "sample_weights.json").write_text(
        json.dumps(
            {
                "format": "output_token_sampling/v1",
                "extra_sampling_fraction": args.extra_sampling_fraction,
                "weights": weights,
            },
            indent=2,
        )
    )
    np.savez_compressed(out / "targets.npz", **arrays)

    print(f"complete={len(records)} selected={len(selected)} rejected={len(rejected)}")
    print(f"optional missing garment pkl={len(missing_optional_pkl)}")
    print(f"train={len(train_records)} val={len(val_records)} body_overlap={len(body_overlap)}")
    print(
        "training coverage: "
        f"missing_garment_categories={len(coverage['missing_train_garment_categories'])} "
        f"missing_output_tokens={len(coverage['missing_train_output_tokens'])} "
        f"missing_categorical_values={len(coverage['missing_train_categorical_values'])}"
    )
    print(
        "validation categorical support: "
        f"unseen={split_audit['unseen_validation_categorical_classes']} "
        f"weak={split_audit['weakly_supported_validation_categorical_classes']}"
    )
    print(
        "equalized epoch schedule: "
        f"groups={equalization_audit['group_count']} "
        f"draws_per_group={equalization_audit['anchor_draws_per_group_per_epoch']} "
        f"items={equalization_audit['items_per_epoch']} "
        f"low_support_groups={len(equalization_audit['low_support_groups'])}"
    )
    print("public category counts are descriptive only; no category downsampling was used")
    print(
        f"model-output deficits: active_heads={len(requirements['low_active_heads'])} "
        f"categorical_fields={len(requirements['low_categorical_classes'])} "
        f"numeric_fields={len(requirements['low_numeric_bins'])}"
    )
    print(f"wrote {out} ready={audit['ready']}")


if __name__ == "__main__":
    main()

