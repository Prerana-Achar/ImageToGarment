#!/usr/bin/env python3
"""Train a five-member route-constrained GarmentCode ensemble."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

from garment_ensemble_model import (
    GarmentEnsembleConfig,
    GarmentTreeEnsemble,
    ROOT_PATHS,
    RouteConstraints,
    count_trainable_parameters,
)
from prepare_data import GarmentDataset
from train_garment_tree import (
    consistency_objective,
    objective_weights,
    readiness_gate,
    repeat_targets,
    split_output,
    supervised_objective,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--aux-prepared-dir")
    parser.add_argument("--aux-pretrain-epochs", type=int, default=15)
    parser.add_argument("--aux-batch-fraction", type=float, default=0.50)
    parser.add_argument("--aux-feature-cache-augmentations", type=int, default=1)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--oof-epochs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--router-warmup-epochs", type=int, default=12)
    parser.add_argument("--samples-per-garment", type=int, default=3)
    parser.add_argument("--root-balance-strength", type=float, default=1.0)
    parser.add_argument("--bagging-fraction", type=float, default=0.85)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--route-dimension", type=int, default=32)
    parser.add_argument("--route-class-token-scale", type=float, default=1.0)
    parser.add_argument("--route-freeze-epoch", type=int, default=15)
    parser.add_argument("--query-layers", type=int, default=1)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--feature-dropout", type=float, default=0.20)
    parser.add_argument("--feature-noise", type=float, default=0.03)
    parser.add_argument("--feature-cache-augmentations", type=int, default=3)
    parser.add_argument("--class-token-scale", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.10)
    parser.add_argument("--router-label-smoothing", type=float, default=0.05)
    parser.add_argument("--lambda-router", type=float, default=0.35)
    parser.add_argument("--aux-router-weight", type=float, default=0.0)
    parser.add_argument("--lambda-pose-consistency", type=float, default=0.15)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--early-validation-epochs", type=int, default=20)
    parser.add_argument("--teacher-force-epochs", type=int, default=10)
    parser.add_argument("--teacher-force-start", type=float, default=0.5)
    parser.add_argument("--stacker-steps", type=int, default=400)
    parser.add_argument("--stacker-lr", type=float, default=0.05)
    parser.add_argument("--backbone-name", default="dinov2_vits14")
    parser.add_argument("--backbone-repo", default="DINOv2")
    parser.add_argument("--aspect-pad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--allow-unbalanced-data", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="ImageToGarment")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-tags", nargs="*", default=("ensemble", "router", "output-balanced"))
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested but CUDA is not available"
        )
    return torch.device(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rows_for_gids(dataset: GarmentDataset, gids: Sequence[str]) -> list[int]:
    return [dataset.row_of[gid] for gid in gids]


def validate_dataset_contract(dataset: GarmentDataset, name: str) -> None:
    if not dataset.garment_ids:
        raise ValueError(f"{name} contains no garments")
    if len(set(dataset.garment_ids)) != len(dataset.garment_ids):
        raise ValueError(f"{name} contains duplicate garment IDs")
    rows = rows_for_gids(dataset, dataset.garment_ids)
    numeric_fields = len(dataset.schema.get("cont_slots", {}))
    constant_fields = len(dataset.schema.get("const_slots", {}))
    categorical_fields = len(dataset.schema["cat_vocab"])
    expected_widths = {
        "y_cont": numeric_fields,
        "mask": numeric_fields,
        "y_const": constant_fields,
        "const_mask": constant_fields,
        "y_cat": categorical_fields,
    }
    for key, expected_width in expected_widths.items():
        value = np.asarray(getattr(dataset, key))
        if value.ndim != 2 or value.shape[1] != expected_width:
            raise ValueError(
                f"{name}.{key} has shape {value.shape}; expected (*, {expected_width})"
            )
    continuous = np.asarray(dataset.y_cont)[rows]
    raw_continuous_mask = np.asarray(dataset.mask)[rows]
    continuous_mask = raw_continuous_mask.astype(bool)
    constants = np.asarray(dataset.y_const)[rows]
    raw_constant_mask = np.asarray(dataset.const_mask)[rows]
    constant_mask = raw_constant_mask.astype(bool)
    if not np.isin(raw_continuous_mask, (0, 1)).all():
        raise ValueError(f"{name} has non-binary continuous activity masks")
    if not np.isin(raw_constant_mask, (0, 1)).all():
        raise ValueError(f"{name} has non-binary constant activity masks")
    if not np.isfinite(continuous[continuous_mask]).all():
        raise ValueError(f"{name} has non-finite active continuous targets")
    if not np.isfinite(constants[constant_mask]).all():
        raise ValueError(f"{name} has non-finite active constant targets")
    if np.any((continuous[continuous_mask] < -1e-6) | (continuous[continuous_mask] > 1.0 + 1e-6)):
        raise ValueError(f"{name} has active continuous targets outside [0, 1]")
    if np.any((constants[constant_mask] < -1e-6) | (constants[constant_mask] > 1.0 + 1e-6)):
        raise ValueError(f"{name} has active constant targets outside [0, 1]")
    categories = np.asarray(dataset.y_cat)[rows]
    for index, (path, vocabulary) in enumerate(dataset.schema["cat_vocab"].items()):
        values = categories[:, index]
        if np.any(values < -1) or np.any(values >= len(vocabulary)):
            raise ValueError(
                f"{name} has out-of-range categorical targets for {path}"
            )
    for path in ROOT_PATHS:
        index = list(dataset.schema["cat_vocab"]).index(path)
        if np.any(categories[:, index] < 0):
            raise ValueError(f"{name} has inactive topology root {path}")


def validate_root_component_coverage(
    dataset: GarmentDataset,
    name: str,
) -> None:
    rows = rows_for_gids(dataset, dataset.garment_ids)
    categories = np.asarray(dataset.y_cat)[rows]
    categorical_paths = list(dataset.schema["cat_vocab"])
    missing: list[str] = []
    for path in ROOT_PATHS:
        index = categorical_paths.index(path)
        observed = set(map(int, categories[:, index]))
        for value, semantic in enumerate(dataset.schema["cat_vocab"][path]):
            if value not in observed:
                missing.append(f"{path}={semantic!r}")
    if missing:
        raise ValueError(
            f"{name} lacks examples for topology components: {missing}"
        )


def validate_route_targets(
    dataset: GarmentDataset,
    constraints: RouteConstraints,
    name: str,
) -> None:
    route_index = {
        route: index for index, route in enumerate(constraints.valid_root_tuples)
    }
    rows = rows_for_gids(dataset, dataset.garment_ids)
    categories = np.asarray(dataset.y_cat)[rows]
    numeric_active = np.concatenate(
        (
            np.asarray(dataset.mask)[rows].astype(bool),
            np.asarray(dataset.const_mask)[rows].astype(bool),
        ),
        axis=1,
    )
    for offset, row in enumerate(rows):
        route = tuple(
            int(categories[offset, index]) for index in constraints.root_indices
        )
        if route not in route_index:
            raise ValueError(f"{name} contains illegal topology route {route}")
        selected = route_index[route]
        blocked_numeric = numeric_active[offset] & ~np.asarray(
            constraints.numeric_masks[selected], dtype=bool
        )
        blocked_categorical = categories[offset] >= 0
        blocked_categorical &= ~np.asarray(
            constraints.categorical_masks[selected], dtype=bool
        )
        if blocked_numeric.any() or blocked_categorical.any():
            gid = dataset.garment_ids[offset]
            numeric_paths = [
                *dataset.schema.get("cont_slots", {}),
                *dataset.schema.get("const_slots", {}),
            ]
            categorical_paths = list(dataset.schema["cat_vocab"])
            blocked_paths = [
                path
                for path, blocked in zip(numeric_paths, blocked_numeric)
                if blocked
            ] + [
                path
                for path, blocked in zip(categorical_paths, blocked_categorical)
                if blocked
            ]
            raise ValueError(
                f"{name} garment {gid} has active targets forbidden by route "
                f"{route}: {blocked_paths}"
            )


def build_constraints(
    dataset: GarmentDataset,
    support_datasets: Sequence[GarmentDataset] = (),
) -> RouteConstraints:
    rows = rows_for_gids(dataset, dataset.garment_ids)
    numeric_mask = np.concatenate(
        (dataset.mask[rows], dataset.const_mask[rows]), axis=1
    )
    support_categories = [np.asarray(dataset.y_cat)[rows]]
    support_numeric = [numeric_mask]
    for support in support_datasets:
        if support.schema != dataset.schema:
            raise ValueError("Route-support dataset schema differs from target schema")
        support_rows = rows_for_gids(support, support.garment_ids)
        support_categories.append(np.asarray(support.y_cat)[support_rows])
        support_numeric.append(
            np.concatenate(
                (
                    np.asarray(support.mask)[support_rows],
                    np.asarray(support.const_mask)[support_rows],
                ),
                axis=1,
            )
        )
    return RouteConstraints.from_targets(
        dataset.schema,
        dataset.y_cat[rows],
        numeric_mask,
        support_y_cat=np.concatenate(support_categories, axis=0),
        support_numeric_mask=np.concatenate(support_numeric, axis=0),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encoder_sha256(model: GarmentTreeEnsemble) -> str:
    cached = getattr(model.encoder, "_feature_cache_sha256", None)
    if cached is not None:
        return str(cached)
    digest = hashlib.sha256()
    for name, value in sorted(model.encoder.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    result = digest.hexdigest()
    setattr(model.encoder, "_feature_cache_sha256", result)
    return result


def cache_input_hashes(dataset: GarmentDataset) -> dict[str, str]:
    prepared = Path(dataset.prepared_dir)
    inputs = [
        prepared / name
        for name in ("schema.json", "targets.npz", "images.json", "splits.json")
    ]
    inputs.extend(
        Path(__file__).with_name(name)
        for name in (
            "prepare_data.py",
            "garment_ensemble_model.py",
            "train_garment_ensemble.py",
        )
    )
    return {str(path.resolve()): file_sha256(path) for path in inputs}


def dataset_image_paths(dataset: GarmentDataset, name: str) -> set[str]:
    samples = getattr(dataset, "image_samples", ())
    paths = [str(Path(raw_path).resolve()) for _, raw_path in samples]
    duplicates = [
        path for path, count in Counter(paths).items() if count > 1
    ]
    if duplicates:
        raise ValueError(
            f"{name} contains duplicate image paths: {duplicates[:10]}"
        )
    return set(paths)


def assert_image_disjoint(
    left: GarmentDataset,
    left_name: str,
    right: GarmentDataset,
    right_name: str,
) -> None:
    overlap = sorted(
        dataset_image_paths(left, left_name)
        & dataset_image_paths(right, right_name)
    )
    if overlap:
        raise ValueError(
            f"{left_name} and {right_name} share images: {overlap[:10]}"
        )


def dataset_image_signature(dataset: GarmentDataset) -> str:
    digest = hashlib.sha256()
    samples = getattr(dataset, "image_samples", ())
    if not samples:
        raise ValueError(
            f"{dataset.split} feature caching requires mode='all_images' "
            "with at least one image"
        )
    sampled_gids = {str(gid) for gid, _ in samples}
    missing_gids = sorted(set(map(str, dataset.garment_ids)) - sampled_gids)
    if missing_gids:
        raise ValueError(
            f"{dataset.split} garments have no cacheable image: "
            f"{missing_gids[:10]}"
        )
    for gid, raw_path in sorted((str(gid), str(path)) for gid, path in samples):
        path = Path(raw_path)
        stat = path.stat()
        digest.update(gid.encode("utf-8"))
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def cache_features(
    model: GarmentTreeEnsemble,
    dataset: GarmentDataset,
    path: Path,
    device: torch.device,
    args: argparse.Namespace,
    *,
    variant: str,
    repeats: int = 1,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("Feature-cache repeats must be positive")
    expected = {
        "format": "garment_feature_cache/v2",
        "input_sha256": cache_input_hashes(dataset),
        "image_signature": dataset_image_signature(dataset),
        "encoder_sha256": encoder_sha256(model),
        "items_per_repeat": len(dataset),
        "encoder_embed_dim": model.encoder.embed_dim,
        "split": dataset.split,
        "variant": variant,
        "repeats": repeats,
        "image_size": args.image_size,
        "aspect_pad": args.aspect_pad,
        "backbone_name": args.backbone_name,
        "gids": sorted(set(dataset.garment_ids)),
        "root_indices": list(model.root_indices),
    }
    if path.is_file() and not args.rebuild_feature_cache:
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if cached.get("metadata") == expected:
            expected_rows = len(dataset) * repeats
            required = (
                "patches", "class_token", "y_cont", "mask",
                "y_const", "const_mask", "y_cat",
            )
            shapes_valid = (
                len(cached.get("gid", ())) == expected_rows
                and all(
                    isinstance(cached.get(key), Tensor)
                    and cached[key].ndim >= 1
                    and cached[key].shape[0] == expected_rows
                    for key in required
                )
                and cached["patches"].ndim == 3
                and cached["class_token"].ndim == 2
                and math.isqrt(cached["patches"].shape[1]) ** 2
                == cached["patches"].shape[1]
                and cached["patches"].shape[-1] == model.encoder.embed_dim
                and cached["class_token"].shape[-1] == model.encoder.embed_dim
                and cached["y_cont"].shape == (
                    expected_rows, len(model.schema.get("cont_slots", {}))
                )
                and cached["mask"].shape == cached["y_cont"].shape
                and cached["y_const"].shape == (
                    expected_rows, len(model.schema.get("const_slots", {}))
                )
                and cached["const_mask"].shape == cached["y_const"].shape
                and cached["y_cat"].shape == (
                    expected_rows, len(model.schema["cat_vocab"])
                )
            )
            if shapes_valid:
                print(f"loaded feature cache {path}", flush=True)
                return cached
            print(f"ignoring malformed feature cache {path}", flush=True)
        else:
            print(f"ignoring incompatible feature cache {path}", flush=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model.encoder.eval()
    collected: dict[str, list[Any]] = defaultdict(list)
    with torch.inference_mode():
        for repeat in range(repeats):
            for step, batch in enumerate(loader):
                images = batch["image"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=args.amp and device.type == "cuda",
                ):
                    patches, class_token = model.encode(images)
                if not bool(torch.isfinite(patches).all()) or not bool(
                    torch.isfinite(class_token).all()
                ):
                    raise RuntimeError("Frozen encoder produced non-finite features")
                collected["patches"].append(patches.detach().cpu().half())
                collected["class_token"].append(class_token.detach().cpu().half())
                for key in ("y_cont", "mask", "y_const", "const_mask", "y_cat"):
                    collected[key].append(batch[key].cpu())
                collected["gid"].extend(map(str, batch["gid"]))
                if step % 20 == 0:
                    print(
                        f"cache {dataset.split}/{variant} pass={repeat + 1}/{repeats}: "
                        f"{step + 1}/{len(loader)}",
                        flush=True,
                    )
    cached = {
        "metadata": expected,
        "gid": collected["gid"],
        **{
            key: torch.cat(collected[key], dim=0)
            for key in (
                "patches", "class_token", "y_cont", "mask",
                "y_const", "const_mask", "y_cat",
            )
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(cached, temporary)
    temporary.replace(path)
    print(f"wrote feature cache {path}", flush=True)
    return cached
def categorical_tokens(
    y_cat: Tensor, root_indices: Sequence[int]
) -> set[tuple[int, int]]:
    tokens = {
        (index, int(value))
        for index, value in enumerate(y_cat.tolist())
        if int(value) >= 0
    }
    root_values = [int(y_cat[index]) for index in root_indices]
    route_code = sum(value * (1000 ** position) for position, value in enumerate(root_values))
    tokens.add((-1, route_code))
    return tokens


def gid_tokens(cache: dict[str, Any]) -> dict[str, set[tuple[int, int]]]:
    output: dict[str, set[tuple[int, int]]] = {}
    for index, gid in enumerate(cache["gid"]):
        tokens = categorical_tokens(
            cache["y_cat"][index], cache["metadata"]["root_indices"]
        )
        numeric_active = torch.cat(
            (cache["mask"][index], cache["const_mask"][index])
        ).bool()
        tokens.update(
            (-2 - field, 1)
            for field in numeric_active.nonzero(as_tuple=False).flatten().tolist()
        )
        output.setdefault(gid, set()).update(tokens)
    return output


def make_coverage_core(
    tokens_by_gid: dict[str, set[tuple[int, int]]],
) -> list[str]:
    if not tokens_by_gid:
        raise ValueError("Cannot build an output coverage core from no garments")
    universe = set().union(*tokens_by_gid.values())
    frequency = Counter(token for tokens in tokens_by_gid.values() for token in tokens)
    uncovered = set(universe)
    core: set[str] = set()
    while uncovered:
        candidates = [gid for gid, tokens in tokens_by_gid.items() if tokens & uncovered]
        if not candidates:
            raise RuntimeError(f"No garment covers output tokens: {sorted(uncovered)[:20]}")
        selected = max(
            candidates,
            key=lambda gid: (
                sum(1.0 / frequency[token] for token in tokens_by_gid[gid] & uncovered),
                len(tokens_by_gid[gid] & uncovered),
            ),
        )
        core.add(selected)
        uncovered -= tokens_by_gid[selected]
    return sorted(core)


def make_coverage_folds(
    tokens_by_gid: dict[str, set[tuple[int, int]]], folds: int, seed: int
) -> tuple[list[str], list[list[str]]]:
    """Legacy cross-fit helper retained for compatibility with existing tests."""
    if folds < 2:
        raise ValueError("Out-of-fold stacking requires at least two folds")
    core = set(make_coverage_core(tokens_by_gid))
    universe = set().union(*tokens_by_gid.values())
    frequency = Counter(token for tokens in tokens_by_gid.values() for token in tokens)
    remaining = sorted(set(tokens_by_gid) - core)
    if len(remaining) < folds:
        raise RuntimeError("The mandatory coverage core leaves too few OOF garments")
    rng = random.Random(seed)
    rng.shuffle(remaining)
    assignments = [[] for _ in range(folds)]
    counts = [Counter() for _ in range(folds)]
    ordered = sorted(
        remaining,
        key=lambda gid: sum(1.0 / frequency[token] for token in tokens_by_gid[gid]),
        reverse=True,
    )
    for gid in ordered:
        options = list(range(folds))
        rng.shuffle(options)
        fold = min(
            options,
            key=lambda item: (
                sum(counts[item][token] / frequency[token] for token in tokens_by_gid[gid]),
                len(assignments[item]),
            ),
        )
        assignments[fold].append(gid)
        counts[fold].update(tokens_by_gid[gid])
    core_tokens = set().union(*(tokens_by_gid[gid] for gid in core))
    if core_tokens != universe:
        raise RuntimeError("Mandatory coverage core is incomplete")
    return sorted(core), assignments

def coverage_bagging_subsets(
    all_gids: Sequence[str],
    coverage_core: Sequence[str],
    members: int,
    fraction: float,
    seed: int,
) -> list[list[str]]:
    if members < 1:
        raise ValueError("Bagging requires at least one ensemble member")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Bagging fraction must be in (0, 1]")
    universe = sorted(set(all_gids))
    if not universe:
        raise ValueError("Bagging requires at least one garment")
    core = sorted(set(coverage_core))
    unknown_core = sorted(set(core) - set(universe))
    if unknown_core:
        raise ValueError(
            f"Coverage core contains unknown garments: {unknown_core[:10]}"
        )
    remaining = sorted(set(universe) - set(core))
    target_size = max(len(core), math.ceil(len(universe) * fraction))
    draw = min(max(target_size - len(core), 0), len(remaining))
    subsets: list[list[str]] = []
    for member in range(members):
        shuffled = list(remaining)
        random.Random(seed + member * 7919).shuffle(shuffled)
        subsets.append(sorted([*core, *shuffled[:draw]]))

    # Keep the ensemble-level union complete without removing mandatory core
    # examples. A member can exceed target_size by a few garments if necessary.
    covered = set().union(*(set(subset) for subset in subsets))
    for offset, gid in enumerate(sorted(set(universe) - covered)):
        subsets[offset % members].append(gid)
        subsets[offset % members].sort()
    if set().union(*(set(subset) for subset in subsets)) != set(universe):
        raise RuntimeError("Bagging subsets do not cover the full training set")
    return subsets


def coverage_sample_indices(
    cache: dict[str, Any],
    allowed_gids: Sequence[str],
    samples_per_garment: int,
    seed: int,
    epoch: int,
) -> list[int]:
    """Sample every garment equally, using distinct cached views when possible.

    Rare output values are handled by the loss weights. Repeating the same one- or
    two-garment class many times only teaches identity memorisation.
    """
    allowed = set(allowed_gids)
    if not allowed:
        raise ValueError("Coverage sampling requires at least one garment")
    if samples_per_garment < 1:
        raise ValueError("samples_per_garment must be positive")
    views: dict[str, list[int]] = defaultdict(list)
    for index, gid in enumerate(cache["gid"]):
        if gid in allowed:
            views[gid].append(index)
    missing = allowed - set(views)
    if missing:
        raise RuntimeError(f"No cached views for garments: {sorted(missing)[:10]}")
    rng = random.Random(seed + epoch * 1009)
    selected: list[int] = []
    for gid in sorted(allowed):
        choices = list(views[gid])
        rng.shuffle(choices)
        if samples_per_garment <= len(choices):
            selected.extend(choices[:samples_per_garment])
        else:
            selected.extend(choices)
            selected.extend(
                rng.choices(choices, k=samples_per_garment - len(choices))
            )
    rng.shuffle(selected)
    return selected


def cache_view_groups(cache: dict[str, Any]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, gid in enumerate(cache["gid"]):
        groups[str(gid)].append(index)
    return groups


def paired_view_indices(
    cache: dict[str, Any],
    indices: Sequence[int],
    groups: dict[str, list[int]],
    seed: int,
) -> list[int]:
    """Choose a different underlying pose for every cached garment view."""
    repeats = max(int(cache.get("metadata", {}).get("repeats", 1)), 1)
    rng = random.Random(seed)
    paired: list[int] = []
    for index in indices:
        gid = str(cache["gid"][index])
        options = groups[gid]
        if len(options) < 2:
            paired.append(index)
            continue
        base_views = (
            len(options) // repeats
            if len(options) % repeats == 0
            else len(options)
        )
        position = options.index(index)
        candidates = [
            candidate
            for candidate_position, candidate in enumerate(options)
            if candidate != index
            and (
                base_views < 2
                or candidate_position % base_views != position % base_views
            )
        ]
        if not candidates:
            candidates = [candidate for candidate in options if candidate != index]
        paired.append(rng.choice(candidates))
    return paired


def auxiliary_sample_indices(
    cache: dict[str, Any],
    allowed_gids: Sequence[str],
    count: int,
    seed: int,
    epoch: int,
) -> list[int]:
    """Draw broad auxiliary garment coverage with one random view per draw."""
    if count <= 0:
        return []
    allowed = sorted(set(allowed_gids))
    if not allowed:
        raise ValueError("Auxiliary sampling requires at least one garment")
    views: dict[str, list[int]] = defaultdict(list)
    allowed_set = set(allowed)
    for index, gid in enumerate(cache["gid"]):
        if gid in allowed_set:
            views[gid].append(index)
    missing = allowed_set - set(views)
    if missing:
        raise RuntimeError(f"No auxiliary cached views for: {sorted(missing)[:10]}")
    rng = random.Random(seed + epoch * 2029)
    selected_gids: list[str] = []
    while len(selected_gids) < count:
        cycle = list(allowed)
        rng.shuffle(cycle)
        selected_gids.extend(cycle[: count - len(selected_gids)])
    indices = [rng.choice(views[gid]) for gid in selected_gids]
    rng.shuffle(indices)
    return indices


def root_balanced_extra_indices(
    cache: dict[str, Any],
    allowed_gids: Sequence[str],
    primary_indices: Sequence[int],
    root_position: int,
    strength: float,
    seed: int,
    epoch: int,
) -> list[int]:
    """Add moderate extra views for rare topology root classes."""
    if strength <= 0.0 or not primary_indices:
        return []
    root_indices = cache.get("metadata", {}).get("root_indices")
    if root_indices is None or root_position >= len(root_indices):
        raise ValueError("Feature cache is missing topology root metadata")
    root_index = int(root_indices[root_position])
    allowed = set(allowed_gids)
    views_by_class: dict[int, list[int]] = defaultdict(list)
    for index, gid in enumerate(cache["gid"]):
        if gid in allowed:
            value = int(cache["y_cat"][index, root_index])
            views_by_class[value].append(index)
    selected_by_class = Counter(
        int(cache["y_cat"][index, root_index]) for index in primary_indices
    )
    if not selected_by_class:
        return []
    max_count = max(selected_by_class.values())
    rng = random.Random(seed + epoch * 6029 + root_position * 131)
    extra: list[int] = []
    for value, count in sorted(selected_by_class.items()):
        target = int(math.ceil(count + (math.sqrt(count * max_count) - count) * strength))
        needed = max(0, target - count)
        if needed:
            extra.extend(rng.choices(views_by_class[value], k=needed))
    rng.shuffle(extra)
    return extra

def cached_batch(
    cache: dict[str, Any], indices: Sequence[int], device: torch.device
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    selection = torch.tensor(indices, dtype=torch.long)
    patches = cache["patches"][selection].to(
        device, dtype=torch.float32, non_blocking=True
    )
    class_token = cache["class_token"][selection].to(
        device, dtype=torch.float32, non_blocking=True
    )
    batch = {
        key: cache[key][selection].to(device, non_blocking=True)
        for key in ("y_cont", "mask", "y_const", "const_mask", "y_cat")
    }
    batch["gid"] = [cache["gid"][index] for index in indices]
    return patches, class_token, batch


def loss_options(
    args: argparse.Namespace, *, evaluation: bool = False
) -> SimpleNamespace:
    return SimpleNamespace(
        label_smoothing=0.0 if evaluation else args.label_smoothing,
        lambda_category=0.45,
        lambda_ordinal=0.20,
        lambda_activity=0.10,
        lambda_uncertainty=0.02,
    )


def router_objective(
    output: dict[str, Any],
    batch: dict[str, Any],
    root_indices: Sequence[int],
    constraints: RouteConstraints,
    label_smoothing: float,
    weights: Any | None = None,
) -> Tensor:
    routes, _, _ = constraints.tensors(output["route_logits"].device)
    truth = batch["y_cat"][:, list(root_indices)]
    matches = truth[:, None, :].eq(routes[None, :, :]).all(dim=-1)
    match_count = matches.sum(dim=-1)
    if not bool(match_count.eq(1).all()):
        bad = truth[match_count.ne(1)][:5].detach().cpu().tolist()
        raise RuntimeError(f"Targets do not map to exactly one semantic route: {bad}")
    route_target = matches.to(torch.long).argmax(dim=-1)
    observed_set = set(constraints.observed_root_tuples)
    observed_indices = [
        index
        for index, route in enumerate(constraints.valid_root_tuples)
        if route in observed_set
    ]
    full_to_observed = route_target.new_full(
        (len(constraints.valid_root_tuples),), -1
    )
    full_to_observed[
        torch.tensor(observed_indices, device=route_target.device)
    ] = torch.arange(len(observed_indices), device=route_target.device)
    observed_target = full_to_observed[route_target]
    observed_rows = observed_target.ge(0)
    if bool(observed_rows.any()):
        joint_route = F.cross_entropy(
            output["route_logits"][observed_rows][:, observed_indices],
            observed_target[observed_rows],
            label_smoothing=label_smoothing,
        )
    else:
        joint_route = output["route_logits"].new_zeros(())
    root_losses = []
    for index in root_indices:
        logits = output["categorical_logits"][index]
        target = batch["y_cat"][:, index]
        if weights is None:
            root_losses.append(
                F.cross_entropy(
                    logits,
                    target,
                    label_smoothing=label_smoothing,
                )
            )
            continue
        class_weight = weights.categorical_class[index].to(
            device=logits.device,
            dtype=logits.dtype,
        )
        root_losses.append(
            F.cross_entropy(
                logits,
                target,
                weight=class_weight,
                label_smoothing=label_smoothing,
            )
        )
    root_auxiliary = torch.stack(root_losses).mean()
    # Every legal auxiliary tuple teaches its factorized roots; only target
    # tuples train the target-specific joint route residual.
    return joint_route + root_auxiliary


def epoch_learning_rate(epoch: int, epochs: int, args: argparse.Namespace) -> float:
    progress = epoch / max(epochs - 1, 1)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))


def scheduled_teacher_force(
    epoch: int, duration: int, start: float = 0.5
) -> float:
    """Linearly decay scheduled sampling without a train/inference cliff."""
    if duration <= 0 or start <= 0.0:
        return 0.0
    progress = min(max(epoch / duration, 0.0), 1.0)
    return start * (1.0 - progress)


def should_validate_epoch(
    epoch: int,
    epochs: int,
    validation_every: int,
    early_validation_epochs: int,
) -> bool:
    """Validate densely while early overfitting is most likely."""
    completed = epoch + 1
    return (
        completed <= early_validation_epochs
        or completed == epochs
        or completed % validation_every == 0
    )


def regularize_cached_features(
    patches: Tensor, class_token: Tensor, args: argparse.Namespace
) -> tuple[Tensor, Tensor]:
    if args.feature_dropout > 0.0:
        keep = 1.0 - args.feature_dropout
        feature_mask = torch.rand(
            (patches.shape[0], patches.shape[1], 1), device=patches.device
        ).lt(keep)
        patches = patches * feature_mask.to(patches.dtype) / keep
        if args.class_token_scale > 0.0:
            class_token = F.dropout(
                class_token,
                p=min(args.feature_dropout * 2.0, 0.50),
                training=True,
            )
    if args.feature_noise > 0.0:
        patches = patches + torch.randn_like(patches) * args.feature_noise
        if args.class_token_scale > 0.0:
            class_token = class_token + torch.randn_like(class_token) * args.feature_noise
    return patches, class_token


def set_trainable_member(model: GarmentTreeEnsemble, member_index: int) -> torch.nn.Module:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    member = model.members[member_index]
    for parameter in member.parameters():
        parameter.requires_grad_(True)
    if model.config.class_token_scale == 0.0:
        member.class_projection.requires_grad_(False)
    return member


def set_route_trainable(member: torch.nn.Module, enabled: bool) -> None:
    for parameter in member.route_parameters():
        parameter.requires_grad_(enabled)
    for module in (
        member.route_patch_projection,
        member.root_routers,
        member.route_router,
    ):
        if module is not None:
            module.train(enabled)


def enforce_route_mode(member: torch.nn.Module) -> None:
    route_trainable = any(
        parameter.requires_grad for parameter in member.route_parameters()
    )
    for module in (
        member.route_patch_projection,
        member.root_routers,
        member.route_router,
    ):
        if module is not None:
            module.train(route_trainable)


def clone_member_state_on_device(member: torch.nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().clone()
        for name, value in member.state_dict().items()
    }


@torch.no_grad()
def update_ema_state(
    member: torch.nn.Module,
    ema_state: dict[str, Tensor],
    decay: float,
) -> None:
    current = member.state_dict()
    if current.keys() != ema_state.keys():
        raise RuntimeError("EMA and member states have different keys")
    for name, value in current.items():
        if value.is_floating_point():
            ema_state[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        else:
            ema_state[name].copy_(value.detach())


def load_member_states(
    model: GarmentTreeEnsemble,
    states: Sequence[dict[str, Tensor]],
) -> None:
    if len(states) != len(model.members):
        raise ValueError("Member state count does not match the ensemble")
    for member, state in zip(model.members, states):
        member.load_state_dict(state)


def clone_states_to_cpu(
    states: Sequence[dict[str, Tensor]],
) -> list[dict[str, Tensor]]:
    return [
        {name: value.detach().cpu().clone() for name, value in state.items()}
        for state in states
    ]


def train_member_epoch(
    model: GarmentTreeEnsemble,
    member_index: int,
    cache: dict[str, Any],
    allowed_gids: Sequence[str],
    epoch: int,
    epochs: int,
    weights: Any,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    *,
    auxiliary_cache: dict[str, Any] | None = None,
    auxiliary_gids: Sequence[str] = (),
    auxiliary_fraction: float = 0.0,
    router_warmup_epochs: int | None = None,
    teacher_force_epochs: int | None = None,
    teacher_force_override: float | None = None,
    router_loss_weight: float | None = None,
    auxiliary_router_weight: float = 0.0,
    ema_state: dict[str, Tensor] | None = None,
    root_balance_strength: float | None = None,
) -> dict[str, float]:
    member = model.members[member_index]
    member.train()
    enforce_route_mode(member)
    lr = epoch_learning_rate(epoch, epochs, args)
    for group in optimizer.param_groups:
        group["lr"] = lr
    primary_indices = coverage_sample_indices(
        cache, allowed_gids, args.samples_per_garment, seed, epoch
    )
    balance_strength = args.root_balance_strength if root_balance_strength is None else root_balance_strength
    primary_indices.extend(
        root_balanced_extra_indices(
            cache, allowed_gids, primary_indices, 2, balance_strength, seed, epoch
        )
    )
    work_batches: list[tuple[dict[str, Any], list[int]]] = [
        (cache, primary_indices[start : start + args.batch_size])
        for start in range(0, len(primary_indices), args.batch_size)
    ]
    if auxiliary_cache is not None and auxiliary_gids and auxiliary_fraction > 0.0:
        if not 0.0 < auxiliary_fraction < 1.0:
            raise ValueError("Auxiliary batch fraction must be strictly between 0 and 1")
        auxiliary_count = max(
            1,
            round(
                len(primary_indices)
                * auxiliary_fraction
                / (1.0 - auxiliary_fraction)
            ),
        )
        auxiliary_indices = auxiliary_sample_indices(
            auxiliary_cache,
            auxiliary_gids,
            auxiliary_count,
            seed + 700001,
            epoch,
        )
        work_batches.extend(
            (auxiliary_cache, auxiliary_indices[start : start + args.batch_size])
            for start in range(0, len(auxiliary_indices), args.batch_size)
        )
    random.Random(seed + epoch * 3011).shuffle(work_batches)
    view_groups = {id(cache): cache_view_groups(cache)}
    if auxiliary_cache is not None:
        view_groups[id(auxiliary_cache)] = cache_view_groups(auxiliary_cache)

    options = loss_options(args)
    totals = Counter()
    batches = 0
    warmup_epochs = (
        args.router_warmup_epochs
        if router_warmup_epochs is None
        else router_warmup_epochs
    )
    teacher_epochs = (
        args.teacher_force_epochs
        if teacher_force_epochs is None
        else teacher_force_epochs
    )
    teacher_force = (
        scheduled_teacher_force(epoch, teacher_epochs, args.teacher_force_start)
        if teacher_force_override is None
        else float(teacher_force_override)
    )
    primary_router_weight = (
        args.lambda_router if router_loss_weight is None else router_loss_weight
    )
    for batch_cache, chosen in work_batches:
        is_auxiliary = auxiliary_cache is not None and batch_cache is auxiliary_cache
        batch_teacher_force = teacher_force
        batch_router_weight = (
            auxiliary_router_weight if is_auxiliary else primary_router_weight
        )
        patches, class_token, batch = cached_batch(batch_cache, chosen, device)
        use_pose_consistency = args.lambda_pose_consistency > 0.0
        if use_pose_consistency:
            paired_indices = paired_view_indices(
                batch_cache,
                chosen,
                view_groups[id(batch_cache)],
                seed + epoch * 100003 + batches * 97,
            )
            paired_patches, paired_class_token, paired_batch = cached_batch(
                batch_cache, paired_indices, device
            )
            for key in ("y_cont", "mask", "y_const", "const_mask", "y_cat"):
                if not torch.equal(batch[key], paired_batch[key]):
                    raise RuntimeError("Paired views have different garment targets")
            paired_patches, paired_class_token = regularize_cached_features(
                paired_patches, paired_class_token, args
            )
        patches, class_token = regularize_cached_features(
            patches, class_token, args
        )
        if use_pose_consistency:
            model_patches = torch.cat((patches, paired_patches), dim=0)
            model_class_token = torch.cat((class_token, paired_class_token), dim=0)
            objective_batch = repeat_targets(batch)
        else:
            model_patches = patches
            model_class_token = class_token
            objective_batch = batch
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            output = model.forward_member_tokens(
                member_index,
                model_patches,
                model_class_token,
                objective_batch["y_cat"],
                batch_teacher_force,
            )
            if batch_router_weight > 0.0:
                router_loss = router_objective(
                    output,
                    objective_batch,
                    model.root_indices,
                    model.constraints,
                    args.router_label_smoothing,
                    weights,
                )
            else:
                router_loss = output["numeric_mean"].new_zeros(())
            if epoch < warmup_epochs and batch_router_weight > 0.0:
                loss = router_loss
                pieces = {"router": router_loss.detach()}
            else:
                supervised_loss, pieces = supervised_objective(
                    output, objective_batch, weights, options
                )
                consistency = supervised_loss.new_zeros(())
                if use_pose_consistency:
                    first, second = split_output(output, len(chosen))
                    consistency = consistency_objective(first, second, batch)
                loss = (
                    supervised_loss
                    + batch_router_weight * router_loss
                    + args.lambda_pose_consistency * consistency
                )
                pieces["router"] = router_loss.detach()
                pieces["consistency"] = consistency.detach()
        if not bool(torch.isfinite(loss)):
            domain = "auxiliary" if is_auxiliary else "target"
            raise FloatingPointError(
                f"Non-finite {domain} loss for member {member_index} "
                f"at epoch {epoch + 1}, batch {batches + 1}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in member.parameters() if parameter.requires_grad],
            1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        if ema_state is not None:
            update_ema_state(member, ema_state, args.ema_decay)
        totals["loss"] += float(loss.detach())
        for key, value in pieces.items():
            totals[key] += float(value)
        batches += 1
    member.eval()
    return {key: value / max(batches, 1) for key, value in totals.items()}

def train_member(
    model: GarmentTreeEnsemble,
    member_index: int,
    cache: dict[str, Any],
    allowed_gids: Sequence[str],
    epochs: int,
    weights: Any,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    log: Any = None,
    phase: str = "oof",
) -> dict[str, float]:
    member = set_trainable_member(model, member_index)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in member.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    last_metrics: dict[str, float] = {}
    for epoch in range(epochs):
        last_metrics = train_member_epoch(
            model,
            member_index,
            cache,
            allowed_gids,
            epoch,
            epochs,
            weights,
            args,
            device,
            seed,
            optimizer,
            scaler,
        )
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"{phase} member={member_index} epoch={epoch + 1}/{epochs} "
                f"loss={last_metrics['loss']:.4f} "
                f"router={last_metrics.get('router', 0.0):.4f}",
                flush=True,
            )
    return last_metrics
@torch.no_grad()
def predict_member_by_gid(
    model: GarmentTreeEnsemble,
    member_index: int,
    cache: dict[str, Any],
    gids: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    allowed = set(gids)
    indices = [index for index, gid in enumerate(cache["gid"]) if gid in allowed]
    gathered: dict[str, dict[str, Any]] = {}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    model.members[member_index].eval()
    for start in range(0, len(indices), args.batch_size):
        chosen = indices[start : start + args.batch_size]
        patches, class_token, _ = cached_batch(cache, chosen, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            output = model.forward_member_tokens(
                member_index, patches, class_token
            )
        for row, index in enumerate(chosen):
            buckets[cache["gid"][index]].append(
                {
                    "numeric_mean": output["numeric_mean"][row].detach().cpu().float(),
                    "numeric_log_scale": output["numeric_log_scale"][row].detach().cpu().float(),
                    "numeric_activity": output["numeric_activity"][row].detach().cpu().float(),
                    "categorical_logits": [
                        logits[row].detach().cpu().float()
                        for logits in output["categorical_logits"]
                    ],
                    "categorical_activity": output["categorical_activity"][row].detach().cpu().float(),
                    "route_logits": output["route_logits"][row].detach().cpu().float(),
                }
            )
    for gid, predictions in buckets.items():
        gathered[gid] = {
            "numeric_mean": torch.stack([item["numeric_mean"] for item in predictions]).mean(0),
            "numeric_log_scale": torch.stack(
                [item["numeric_log_scale"] for item in predictions]
            ).mean(0),
            "numeric_activity": torch.stack(
                [item["numeric_activity"] for item in predictions]
            ).mean(0),
            "categorical_logits": [
                torch.stack([item["categorical_logits"][field] for item in predictions]).mean(0)
                for field in range(len(model.categorical_paths))
            ],
            "categorical_activity": torch.stack(
                [item["categorical_activity"] for item in predictions]
            ).mean(0),
            "route_logits": torch.stack(
                [item["route_logits"] for item in predictions]
            ).mean(0),
        }
    missing = allowed - set(gathered)
    if missing:
        raise RuntimeError(f"No cached views for garments: {sorted(missing)[:10]}")
    return gathered


def oof_member_batches(
    oof: Sequence[dict[str, dict[str, Any]]],
    gids: Sequence[str],
    device: torch.device,
) -> list[dict[str, Any]]:
    members = []
    categorical_count = len(oof[0][gids[0]]["categorical_logits"])
    for predictions in oof:
        members.append(
            {
                "numeric_mean": torch.stack(
                    [predictions[gid]["numeric_mean"] for gid in gids]
                ).to(device),
                "numeric_log_scale": torch.stack(
                    [predictions[gid]["numeric_log_scale"] for gid in gids]
                ).to(device),
                "numeric_activity": torch.stack(
                    [predictions[gid]["numeric_activity"] for gid in gids]
                ).to(device),
                "categorical_logits": [
                    torch.stack(
                        [predictions[gid]["categorical_logits"][field] for gid in gids]
                    ).to(device)
                    for field in range(categorical_count)
                ],
                "categorical_activity": torch.stack(
                    [predictions[gid]["categorical_activity"] for gid in gids]
                ).to(device),
                "route_logits": torch.stack(
                    [predictions[gid]["route_logits"] for gid in gids]
                ).to(device),
            }
        )
    return members


def target_batch_for_gids(
    cache: dict[str, Any], gids: Sequence[str], device: torch.device
) -> dict[str, Any]:
    first_index: dict[str, int] = {}
    for index, gid in enumerate(cache["gid"]):
        first_index.setdefault(gid, index)
    selection = torch.tensor([first_index[gid] for gid in gids], dtype=torch.long)
    return {
        key: cache[key][selection].to(device)
        for key in ("y_cont", "mask", "y_const", "const_mask", "y_cat")
    }


def evaluation_objective_output(
    output: dict[str, Any], *, detach_to_cpu: bool = False
) -> dict[str, Any]:
    """Return training-compatible predictions for objective evaluation.

    Route-masked activity logits are a decoding constraint.  Feeding their
    sentinel value (-30) into BCE makes one route mistake look like many
    independent activity mistakes, even though expert heads train on the raw
    activity logits.
    """

    def prepare(value: Tensor) -> Tensor:
        if not detach_to_cpu:
            return value
        return value.detach().float().cpu()

    return {
        "numeric_mean": prepare(output["numeric_mean"]),
        "numeric_log_scale": prepare(output["numeric_log_scale"]),
        "numeric_activity": prepare(
            output.get("raw_numeric_activity", output["numeric_activity"])
        ),
        "categorical_logits": [
            prepare(logits) for logits in output["categorical_logits"]
        ],
        "categorical_activity": prepare(
            output.get(
                "raw_categorical_activity", output["categorical_activity"]
            )
        ),
    }


def concatenate_evaluation_objective(
    outputs: Sequence[dict[str, Any]], batches: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not outputs or len(outputs) != len(batches):
        raise ValueError("Evaluation requires matching non-empty outputs and batches")
    merged_output = {
        key: torch.cat([item[key] for item in outputs], dim=0)
        for key in (
            "numeric_mean",
            "numeric_log_scale",
            "numeric_activity",
            "categorical_activity",
        )
    }
    merged_output["categorical_logits"] = [
        torch.cat([item["categorical_logits"][field] for item in outputs], dim=0)
        for field in range(len(outputs[0]["categorical_logits"]))
    ]
    merged_batch = {
        key: torch.cat([item[key] for item in batches], dim=0)
        for key in ("y_cont", "mask", "y_const", "const_mask", "y_cat")
    }
    return merged_output, merged_batch


def decoded_categorical_prediction(
    output: dict[str, Any],
    field: int,
    root_position_by_field: dict[int, int],
) -> Tensor:
    root_position = root_position_by_field.get(field)
    if root_position is not None:
        return output["root_selection"][:, root_position]
    return output["categorical_logits"][field].argmax(dim=-1)


def fit_stacker(
    model: GarmentTreeEnsemble,
    oof: Sequence[dict[str, dict[str, Any]]],
    gids: Sequence[str],
    cache: dict[str, Any],
    weights: Any,
    args: argparse.Namespace,
    device: torch.device,
    log: Any = None,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.stacker_parameters():
        parameter.requires_grad_(True)
        parameter.data.zero_()
    optimizer = torch.optim.Adam(model.stacker_parameters(), lr=args.stacker_lr)
    outputs = oof_member_batches(oof, gids, device)
    targets = target_batch_for_gids(cache, gids, device)
    options = loss_options(args)
    for step in range(args.stacker_steps):
        optimizer.zero_grad(set_to_none=True)
        ensemble = model.aggregate(outputs)
        loss, pieces = supervised_objective(
            evaluation_objective_output(ensemble), targets, weights, options
        )
        regularization = sum(parameter.square().mean() for parameter in model.stacker_parameters())


        total = loss + 0.01 * regularization
        total.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == args.stacker_steps:
            print(
                f"stacker step={step + 1}/{args.stacker_steps} loss={float(loss):.4f}",
                flush=True,
            )
    for parameter in model.stacker_parameters():
        parameter.requires_grad_(False)
@torch.no_grad()
def evaluate_cached(
    model: GarmentTreeEnsemble,
    cache: dict[str, Any],
    weights: Any,
    args: argparse.Namespace,
    device: torch.device,
    member_only: int | None = None,
) -> dict[str, float]:
    model.eval()
    # Keep reported train/validation loss comparable across experiments.
    # Label smoothing is a training regularizer, not an evaluation metric.
    options = loss_options(args, evaluation=True)
    objective_outputs: list[dict[str, Any]] = []
    objective_batches: list[dict[str, Any]] = []
    numeric_error = 0.0
    numeric_count = 0.0
    field_correct = torch.zeros(len(model.categorical_paths), dtype=torch.float64)
    field_count = torch.zeros_like(field_correct)
    root_correct = 0
    root_total = 0
    root_class_correct = [
        torch.zeros(len(model.schema["cat_vocab"][path]), dtype=torch.float64)
        for path in ROOT_PATHS
    ]
    root_class_count = [
        torch.zeros_like(correct)
        for correct in root_class_correct
    ]
    predicted_blocked_ground_truth_paths = 0.0
    route_nll = 0.0
    member_disagreement = 0.0
    invalid_emissions = 0.0
    emitted_paths = 0.0
    raw_invalid_emissions = 0.0
    raw_emitted_paths = 0.0
    blocked_ground_truth_paths = 0.0
    active_ground_truth_paths = 0.0
    samples = 0
    root_position_by_field = {
        field: position
        for position, field in enumerate(model.root_indices)
    }
    all_indices = list(range(len(cache["gid"])))
    for start in range(0, len(all_indices), args.batch_size):
        chosen = all_indices[start : start + args.batch_size]
        patches, class_token, batch = cached_batch(cache, chosen, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            if member_only is None:
                member_outputs = [
                    model.forward_member_tokens(index, patches, class_token)
                    for index in range(model.config.member_count)
                ]
            else:
                single = model.forward_member_tokens(member_only, patches, class_token)
                member_outputs = [single] * model.config.member_count
            output = model.aggregate(member_outputs)
        predicted_numeric_active = output["numeric_activity"].sigmoid().ge(0.5)
        predicted_categorical_active = output["categorical_activity"].sigmoid().ge(0.5)
        raw_numeric_active = output["raw_numeric_activity"].sigmoid().ge(0.5)
        raw_categorical_active = output["raw_categorical_activity"].sigmoid().ge(0.5)
        raw_invalid_emissions += float(
            (raw_numeric_active & ~output["route_numeric_mask"]).sum()
            + (raw_categorical_active & ~output["route_categorical_mask"]).sum()
        )
        raw_emitted_paths += float(
            raw_numeric_active.sum() + raw_categorical_active.sum()
        )
        invalid_emissions += float(
            (predicted_numeric_active & ~output["route_numeric_mask"]).sum()
            + (predicted_categorical_active & ~output["route_categorical_mask"]).sum()
        )
        emitted_paths += float(
            predicted_numeric_active.sum() + predicted_categorical_active.sum()
        )
        objective_outputs.append(
            evaluation_objective_output(output, detach_to_cpu=True)
        )
        objective_batches.append(
            {
                key: batch[key].detach().cpu()
                for key in ("y_cont", "mask", "y_const", "const_mask", "y_cat")
            }
        )
        target_numeric = torch.cat((batch["y_cont"], batch["y_const"]), dim=-1)
        active_numeric = torch.cat((batch["mask"], batch["const_mask"]), dim=-1).bool()
        numeric_error += float(
            (output["numeric_mean"] - target_numeric).abs()[active_numeric].sum()
        )
        numeric_count += float(active_numeric.sum())
        for field in range(len(output["categorical_logits"])):
            active = batch["y_cat"][:, field].ge(0)
            predicted = decoded_categorical_prediction(
                output, field, root_position_by_field
            )
            field_correct[field] += float(
                predicted[active].eq(batch["y_cat"][:, field][active]).sum()
            )
            field_count[field] += float(active.sum())
        truth_roots = batch["y_cat"][:, list(model.root_indices)]
        root_correct += int(output["root_selection"].eq(truth_roots).all(dim=1).sum())
        root_total += truth_roots.shape[0]
        for position, (class_correct, class_count) in enumerate(
            zip(root_class_correct, root_class_count)
        ):
            truth = truth_roots[:, position]
            predicted = output["root_selection"][:, position]
            classes = class_count.numel()
            class_count += torch.bincount(
                truth.detach().cpu(), minlength=classes
            ).to(torch.float64)
            matched_truth = truth[predicted.eq(truth)]
            class_correct += torch.bincount(
                matched_truth.detach().cpu(), minlength=classes
            ).to(torch.float64)
        routes, numeric_masks, categorical_masks = model.constraints.tensors(output["route_logits"].device)
        matches = truth_roots[:, None, :].eq(routes[None, :, :]).all(dim=-1)
        if not bool(matches.sum(dim=-1).eq(1).all()):
            raise RuntimeError("Evaluation target has no unique semantic route")
        route_target = matches.to(torch.long).argmax(dim=-1)
        truth_numeric_mask = numeric_masks[route_target]
        truth_categorical_mask = categorical_masks[route_target]
        active_categorical = batch["y_cat"].ge(0)
        predicted_blocked_ground_truth_paths += float(
            (active_numeric & ~output["route_numeric_mask"]).sum()
            + (active_categorical & ~output["route_categorical_mask"]).sum()
        )
        blocked_ground_truth_paths += float(
            (active_numeric & ~truth_numeric_mask).sum()
            + (active_categorical & ~truth_categorical_mask).sum()
        )
        active_ground_truth_paths += float(
            active_numeric.sum() + active_categorical.sum()
        )
        route_nll += float(
            F.cross_entropy(
                output["route_logits"],

                route_target,
                reduction="sum",
            )
        )
        stacked_numeric = torch.stack(
            [item["numeric_mean"] for item in member_outputs], dim=0
        )
        member_disagreement += float(stacked_numeric.std(dim=0, unbiased=False).mean()) * len(chosen)
        samples += len(chosen)
    full_output, full_batch = concatenate_evaluation_objective(
        objective_outputs, objective_batches
    )
    cpu_weights = weights.to(torch.device("cpu"))
    loss, pieces = supervised_objective(
        full_output,
        full_batch,
        cpu_weights,
        options,
    )
    detail_fields = tuple(
        field
        for field in range(len(model.categorical_paths))
        if field not in root_position_by_field
    )
    primary_options = SimpleNamespace(
        **{
            **vars(options),
            "categorical_field_indices": detail_fields,
            "lambda_ordinal": 0.0,
            "lambda_activity": 0.0,
            "lambda_uncertainty": 0.0,
        }
    )
    primary_loss, primary_pieces = supervised_objective(
        full_output,
        full_batch,
        cpu_weights,
        primary_options,
    )

    active_fields = field_count > 0
    field_accuracy = field_correct / field_count.clamp_min(1.0)
    macro_accuracy = field_accuracy[active_fields].mean()
    detail_active_fields = active_fields.clone()
    detail_active_fields[list(model.root_indices)] = False
    detail_macro_accuracy = (
        field_accuracy[detail_active_fields].mean()
        if bool(detail_active_fields.any())
        else field_accuracy.new_tensor(1.0)
    )
    root_class_recalls = torch.cat(
        [
            correct[count > 0] / count[count > 0]
            for correct, count in zip(root_class_correct, root_class_count)
        ]
    )
    root_macro_accuracy = root_class_recalls.mean()
    root_worst_class_recall = root_class_recalls.min()
    metrics = {
        "loss": float(loss),
        "primary_loss": float(primary_loss),
        "loss_numeric": float(pieces["numeric"]),
        "loss_category": float(pieces["category"]),
        "loss_detail_category": float(primary_pieces["category"]),
        "loss_ordinal": float(pieces["ordinal"]),
        "loss_activity": float(pieces["activity"]),
        "loss_uncertainty": float(pieces["uncertainty"]),
        "numeric_mae": numeric_error / max(numeric_count, 1.0),
        "categorical_macro_accuracy": float(macro_accuracy),
        "detail_categorical_macro_accuracy": float(detail_macro_accuracy),
        "root_macro_accuracy": float(root_macro_accuracy),
        "root_worst_class_recall": float(root_worst_class_recall),
        "route_nll": route_nll / max(root_total, 1),
        "root_exact": root_correct / max(root_total, 1),
        "member_numeric_disagreement": member_disagreement / max(samples, 1),
        "predicted_route_block_rate": (
            predicted_blocked_ground_truth_paths / max(active_ground_truth_paths, 1.0)
        ),
        "invalid_path_emission_rate": invalid_emissions / max(emitted_paths, 1.0),
        "raw_invalid_path_emission_rate": (
            raw_invalid_emissions / max(raw_emitted_paths, 1.0)
        ),
        "ground_truth_route_block_rate": (
            blocked_ground_truth_paths / max(active_ground_truth_paths, 1.0)
        ),
    }
    non_finite = {
        key: value for key, value in metrics.items() if not math.isfinite(value)
    }
    if non_finite:
        raise FloatingPointError(
            f"Evaluation produced non-finite metrics: {non_finite}"
        )
    return metrics


def clone_member_states(model: GarmentTreeEnsemble) -> list[dict[str, Tensor]]:
    return [
        {
            key: value.detach().cpu().clone()
            for key, value in member.state_dict().items()
        }
        for member in model.members
    ]


def train_final_ensemble(
    model: GarmentTreeEnsemble,
    train_cache: dict[str, Any],
    train_eval_cache: dict[str, Any],
    validation_cache: dict[str, Any],
    member_gids: Sequence[Sequence[str]],
    weights: Any,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    wandb_run: Any,
    *,
    auxiliary_cache: dict[str, Any] | None = None,
    auxiliary_eval_cache: dict[str, Any] | None = None,
    auxiliary_gids: Sequence[str] = (),
) -> dict[str, Any]:
    """Auxiliary-pretrain, bagged-joint-train, and restore the best target epoch.

    Training always runs for all requested epochs. Only the body-disjoint target
    validation set selects the final checkpoint; auxiliary validation is diagnostic.
    """
    if len(member_gids) != len(model.members) or any(
        not gids for gids in member_gids
    ):
        raise ValueError("Every ensemble member requires a non-empty bagged subset")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for member_index in range(model.config.member_count):
        model.reset_member(member_index)
    for member in model.members:
        for parameter in member.parameters():
            parameter.requires_grad_(True)
        if model.config.class_token_scale == 0.0:
            member.class_projection.requires_grad_(False)

    optimizers = [
        torch.optim.AdamW(
            [parameter for parameter in member.parameters() if parameter.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        for member in model.members
    ]
    scalers = [
        torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
        for _ in model.members
    ]
    member_seeds = [
        args.seed + 100000 + index * 10000
        for index in range(len(model.members))
    ]
    auxiliary_pretrain: dict[str, Any] | None = None
    for member in model.members:
        set_route_trainable(member, args.aux_router_weight > 0.0)
    if (
        auxiliary_cache is not None
        and auxiliary_gids
        and args.aux_pretrain_epochs > 0
    ):
        auxiliary_last: list[dict[str, float]] = []
        for epoch in range(args.aux_pretrain_epochs):
            auxiliary_last = []
            for member_index in range(len(model.members)):
                seed_everything(member_seeds[member_index] + 500000 + epoch)
                metrics = train_member_epoch(
                    model,
                    member_index,
                    auxiliary_cache,
                    auxiliary_gids,
                    epoch,
                    args.aux_pretrain_epochs,
                    weights,
                    args,
                    device,
                    member_seeds[member_index] + 500000,
                    optimizers[member_index],
                    scalers[member_index],
                    router_warmup_epochs=0,
                    teacher_force_epochs=min(
                        args.teacher_force_epochs,
                        max(args.aux_pretrain_epochs // 2, 1),
                    ),
                    router_loss_weight=args.aux_router_weight,
                    root_balance_strength=0.0,
                )
                auxiliary_last.append(
                    {"member": float(member_index), **metrics}
                )
            if (
                epoch == 0
                or (epoch + 1) % 5 == 0
                or epoch + 1 == args.aux_pretrain_epochs
            ):
                mean_loss = sum(item["loss"] for item in auxiliary_last) / max(
                    len(auxiliary_last), 1
                )
                print(
                    f"auxiliary pretrain epoch={epoch + 1}/{args.aux_pretrain_epochs} "
                    f"optimisation_loss={mean_loss:.4f}",
                    flush=True,
                )
        target_validation = evaluate_cached(
            model, validation_cache, weights, args, device
        )
        source_validation = (
            evaluate_cached(model, auxiliary_eval_cache, weights, args, device)
            if auxiliary_eval_cache is not None
            else None
        )
        auxiliary_pretrain = {
            "epochs": args.aux_pretrain_epochs,
            "garments": len(set(auxiliary_gids)),
            "last_member_optimisation": auxiliary_last,
            "target_validation": target_validation,
            "source_validation": source_validation,
        }
        (output_dir / "auxiliary_pretrain_metrics.json").write_text(
            json.dumps(auxiliary_pretrain, indent=2)
        )
        print(
            f"auxiliary pretrain complete target_val_loss={target_validation['loss']:.4f}"
            + (
                ""
                if source_validation is None
                else f" source_val_loss={source_validation['loss']:.4f}"
            ),
            flush=True,
        )

    for member in model.members:
        set_route_trainable(member, True)

    def checkpoint_score(metrics: dict[str, float]) -> float:
        return (
            float(metrics["primary_loss"])
            + 0.35 * (1.0 - float(metrics["root_exact"]))
            + 0.35 * (1.0 - float(metrics["root_macro_accuracy"]))
            + 0.25 * (1.0 - float(metrics["detail_categorical_macro_accuracy"]))
            + 0.10 * float(metrics["numeric_mae"])
        )

    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_validation_loss = float("inf")
    best_validation_score: float | None = None
    best_member_states: list[dict[str, Tensor]] | None = None
    best_checkpoint_source = "raw"
    ema_member_states: list[dict[str, Tensor]] | None = None
    last_member_metrics: list[dict[str, float]] = []
    validation_every = max(int(args.validation_every), 1)
    joint_auxiliary = auxiliary_cache is not None and bool(auxiliary_gids)

    for epoch in range(args.epochs):
        route_trainable = epoch < args.route_freeze_epoch
        for member in model.members:
            set_route_trainable(member, route_trainable)
        if epoch == args.route_freeze_epoch:
            ema_member_states = [
                clone_member_state_on_device(member) for member in model.members
            ]
            print(
                f"route branch frozen after {args.route_freeze_epoch} target epochs; "
                f"parameter-head EMA started with decay={args.ema_decay:.5f}",
                flush=True,
            )
        last_member_metrics = []
        for member_index in range(len(model.members)):
            seed_everything(member_seeds[member_index] + epoch)
            metrics = train_member_epoch(
                model,
                member_index,
                train_cache,
                member_gids[member_index],
                epoch,
                args.epochs,
                weights,
                args,
                device,
                member_seeds[member_index],
                optimizers[member_index],
                scalers[member_index],
                auxiliary_cache=auxiliary_cache if joint_auxiliary else None,
                auxiliary_gids=auxiliary_gids,
                auxiliary_fraction=(
                    args.aux_batch_fraction if joint_auxiliary else 0.0
                ),
                router_loss_weight=(
                    args.lambda_router if route_trainable else 0.0
                ),
                auxiliary_router_weight=(
                    args.aux_router_weight if route_trainable else 0.0
                ),
                ema_state=(
                    None
                    if ema_member_states is None
                    else ema_member_states[member_index]
                ),
            )
            last_member_metrics.append(
                {"member": float(member_index), **metrics}
            )
        mean_optimisation_loss = sum(
            item["loss"] for item in last_member_metrics
        ) / max(len(last_member_metrics), 1)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(
                f"final ensemble epoch={epoch + 1}/{args.epochs} "
                f"optimisation_loss={mean_optimisation_loss:.4f}",
                flush=True,
            )

        should_validate = should_validate_epoch(
            epoch,
            args.epochs,
            validation_every,
            args.early_validation_epochs,
        )
        if not should_validate:
            continue
        raw_train_metrics = evaluate_cached(
            model, train_eval_cache, weights, args, device
        )
        raw_validation_metrics = evaluate_cached(
            model, validation_cache, weights, args, device
        )
        ema_train_metrics: dict[str, float] | None = None
        ema_validation_metrics: dict[str, float] | None = None
        if ema_member_states is not None:
            raw_member_states = [
                clone_member_state_on_device(member) for member in model.members
            ]
            load_member_states(model, ema_member_states)
            try:
                ema_train_metrics = evaluate_cached(
                    model, train_eval_cache, weights, args, device
                )
                ema_validation_metrics = evaluate_cached(
                    model, validation_cache, weights, args, device
                )
            finally:
                load_member_states(model, raw_member_states)

        checkpoint_source = "raw"
        train_metrics = raw_train_metrics
        validation_metrics = raw_validation_metrics
        if (
            ema_validation_metrics is not None
            and checkpoint_score(ema_validation_metrics)
            < checkpoint_score(raw_validation_metrics)
        ):
            checkpoint_source = "ema"
            train_metrics = ema_train_metrics
            validation_metrics = ema_validation_metrics
        record = {
            "epoch": epoch + 1,
            "optimisation_loss": mean_optimisation_loss,
            "checkpoint_source": checkpoint_source,
            "target_train": train_metrics,
            "target_validation": validation_metrics,
            "raw_target_train": raw_train_metrics,
            "raw_target_validation": raw_validation_metrics,
            "ema_target_train": ema_train_metrics,
            "ema_target_validation": ema_validation_metrics,
            "auxiliary_batch_fraction": (
                args.aux_batch_fraction if joint_auxiliary else 0.0
            ),
            "route_trainable": route_trainable,
        }
        history.append(record)
        (output_dir / "training_curve.json").write_text(
            json.dumps(history, indent=2)
        )
        print(
            f"ensemble checkpoint epoch={epoch + 1} source={checkpoint_source} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_primary={validation_metrics['primary_loss']:.4f} "
            f"raw_val_loss={raw_validation_metrics['loss']:.4f} "
            + (
                ""
                if ema_validation_metrics is None
                else f"ema_val_loss={ema_validation_metrics['loss']:.4f} "
            )
            + f"val_route_nll={validation_metrics['route_nll']:.4f} "
            f"val_root_macro={validation_metrics['root_macro_accuracy']:.4f} "
            f"val_root_exact={validation_metrics['root_exact']:.4f}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train_loss": train_metrics["loss"],
                    "val_loss": validation_metrics["loss"],
                    "val_primary_loss": validation_metrics["primary_loss"],
                    "val_loss_numeric": validation_metrics["loss_numeric"],
                    "val_loss_category": validation_metrics["loss_category"],
                    "val_loss_detail_category": validation_metrics[
                        "loss_detail_category"
                    ],
                    "val_loss_ordinal": validation_metrics["loss_ordinal"],
                    "val_loss_activity": validation_metrics["loss_activity"],
                    "val_loss_uncertainty": validation_metrics["loss_uncertainty"],
                    "val_predicted_route_block_rate": validation_metrics[
                        "predicted_route_block_rate"
                    ],
                },
                step=epoch + 1,
            )
        validation_score = checkpoint_score(validation_metrics)
        checkpoint_eligible = True
        if checkpoint_eligible and (
            best_validation_score is None or validation_score < best_validation_score
        ):
            best_epoch = epoch + 1
            best_validation_loss = validation_metrics["loss"]
            best_validation_score = validation_score
            best_checkpoint_source = checkpoint_source
            best_member_states = (
                clone_states_to_cpu(ema_member_states)
                if checkpoint_source == "ema" and ema_member_states is not None
                else clone_member_states(model)
            )
            atomic_save(
                {
                    "format": "GarmentTreeEnsemble/member-snapshot-v1",
                    "epoch": best_epoch,
                    "checkpoint_source": best_checkpoint_source,
                    "validation": validation_metrics,
                    "auxiliary_pretrain": auxiliary_pretrain,
                    "members": best_member_states,
                },
                output_dir / "best_members_during_training.pt",
            )

    atomic_save(
        {
            "format": "GarmentTreeEnsemble/member-snapshot-v1",
            "epoch": args.epochs,
            "checkpoint_source": "raw",
            "members": clone_member_states(model),
        },
        output_dir / "final_epoch_members.pt",
    )
    if ema_member_states is not None:
        atomic_save(
            {
                "format": "GarmentTreeEnsemble/member-snapshot-v1",
                "epoch": args.epochs,
                "checkpoint_source": "ema",
                "ema_decay": args.ema_decay,
                "members": clone_states_to_cpu(ema_member_states),
            },
            output_dir / "final_ema_members.pt",
        )
    if best_member_states is None:
        raise RuntimeError("Final ensemble training produced no validation checkpoint")
    for member, state in zip(model.members, best_member_states):
        member.load_state_dict(state)
        member.eval()
    train_metrics = evaluate_cached(model, train_eval_cache, weights, args, device)
    validation_metrics = evaluate_cached(
        model, validation_cache, weights, args, device
    )
    print(
        f"restored best ensemble epoch={best_epoch} source={best_checkpoint_source} "
        f"train_loss={train_metrics['loss']:.4f} "
        f"val_loss={validation_metrics['loss']:.4f}",
        flush=True,
    )
    return {
        "best_epoch": best_epoch,
        "best_checkpoint_source": best_checkpoint_source,
        "checkpoint_selection": "minimize primary_loss + 0.35*(1-root_exact) + 0.35*(1-root_macro_accuracy) + 0.25*(1-detail_categorical_macro_accuracy) + 0.10*numeric_mae",
        "train": train_metrics,
        "validation": validation_metrics,
        "last_member_optimisation": last_member_metrics,
        "auxiliary_pretrain": auxiliary_pretrain,
        "curve": history,
    }

def model_preflight(
    model: GarmentTreeEnsemble, args: argparse.Namespace, device: torch.device
) -> None:
    model.eval()
    image = torch.zeros((1, 3, args.image_size, args.image_size), device=device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=args.amp and device.type == "cuda",
    ):
        output = model(image)
    if output["numeric_mean"].shape != (1, len(model.numeric_paths)):
        raise RuntimeError("Ensemble numeric output contract failed")
    if output["route_logits"].shape != (
        1,
        len(model.constraints.valid_root_tuples),
    ):
        raise RuntimeError("Joint route output contract failed")
    if len(output["categorical_logits"]) != len(model.categorical_paths):
        raise RuntimeError("Ensemble categorical output contract failed")
    invalid = (
        (output["numeric_activity"].sigmoid().ge(0.5) & ~output["route_numeric_mask"]).any()
        or (
            output["categorical_activity"].sigmoid().ge(0.5)
            & ~output["route_categorical_mask"]
        ).any()
    )
    if bool(invalid):
        raise RuntimeError("Ensemble preflight emitted a route-forbidden path")
    print(
        f"preflight OK numeric={len(model.numeric_paths)} "
        f"categorical={len(model.categorical_paths)}",
        flush=True,
    )

def atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def init_wandb(args: argparse.Namespace, output_dir: Path, config: dict[str, Any]) -> Any:
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("--wandb was requested but wandb is not installed") from error
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=output_dir.name,
        dir=str(output_dir),
        mode=args.wandb_mode,
        tags=list(args.wandb_tags),
        config=config,
    )
def main() -> None:
    args = parse_args()
    positive_integer_options = {
        "--image-size": args.image_size,
        "--batch-size": args.batch_size,
        "--members": args.members,
        "--epochs": args.epochs,
        "--samples-per-garment": args.samples_per_garment,
        "--feature-cache-augmentations": args.feature_cache_augmentations,
        "--validation-every": args.validation_every,
        "--early-validation-epochs": args.early_validation_epochs,
        "--attention-heads": args.attention_heads,
        "--dimension": args.dimension,
        "--route-dimension": args.route_dimension,
        "--route-freeze-epoch": args.route_freeze_epoch,
        "--query-layers": args.query_layers,
    }
    invalid_positive = [
        name for name, value in positive_integer_options.items() if value < 1
    ]
    if invalid_positive:
        raise ValueError(f"These options must be positive: {invalid_positive}")
    if args.route_freeze_epoch < args.router_warmup_epochs:
        raise ValueError("--route-freeze-epoch must be at least --router-warmup-epochs")
    if args.route_freeze_epoch >= args.epochs:
        raise ValueError("--route-freeze-epoch must be smaller than --epochs")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if (
        args.router_warmup_epochs < 0
        or args.teacher_force_epochs < 0
        or args.early_validation_epochs < 0
    ):
        raise ValueError("Scheduling epoch counts cannot be negative")
    if not 0.0 <= args.teacher_force_start <= 1.0:
        raise ValueError("--teacher-force-start must be in [0, 1]")
    if args.lambda_router < 0.0:
        raise ValueError("--lambda-router cannot be negative")
    if args.root_balance_strength < 0.0:
        raise ValueError("--root-balance-strength cannot be negative")
    if args.aux_router_weight < 0.0:
        raise ValueError("--aux-router-weight cannot be negative")
    if args.lambda_pose_consistency < 0.0:
        raise ValueError("--lambda-pose-consistency cannot be negative")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if not 0.0 <= args.router_label_smoothing < 1.0:
        raise ValueError("--router-label-smoothing must be in [0, 1)")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be strictly between 0 and 1")
    if args.dimension % args.attention_heads:
        raise ValueError("--dimension must be divisible by --attention-heads")
    if not 0.0 <= args.feature_dropout < 1.0:
        raise ValueError("--feature-dropout must be in [0, 1)")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.feature_noise < 0.0:
        raise ValueError("--feature-noise cannot be negative")
    if args.lr <= 0.0 or not 0.0 <= args.min_lr <= args.lr:
        raise ValueError("--lr must be positive and --min-lr must be in [0, lr]")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay cannot be negative")
    if not math.isfinite(args.class_token_scale):
        raise ValueError("--class-token-scale must be finite")
    if (
        not math.isfinite(args.route_class_token_scale)
        or args.route_class_token_scale < 0.0
    ):
        raise ValueError("--route-class-token-scale must be finite and non-negative")
    if args.aux_pretrain_epochs < 0:
        raise ValueError("--aux-pretrain-epochs cannot be negative")
    if not 0.0 < args.bagging_fraction <= 1.0:
        raise ValueError("--bagging-fraction must be in (0, 1]")
    if args.aux_feature_cache_augmentations < 1:
        raise ValueError("--aux-feature-cache-augmentations must be positive")
    if args.aux_prepared_dir and not 0.0 < args.aux_batch_fraction < 1.0:
        raise ValueError("--aux-batch-fraction must be strictly between 0 and 1")
    seed_everything(args.seed)
    readiness_gate(args.prepared_dir, args.allow_unbalanced_data)
    device = choose_device(args.device)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_base = GarmentDataset(
        args.prepared_dir,
        split="train",
        mode="single",
        train=False,
        image_size=args.image_size,
        augmentation="none",
        aspect_pad=args.aspect_pad,
    )
    train_augmented_images = GarmentDataset(
        args.prepared_dir,
        split="train",
        mode="all_images",
        train=True,
        image_size=args.image_size,
        augmentation="domain",
        aspect_pad=args.aspect_pad,
    )
    train_evaluation_images = GarmentDataset(
        args.prepared_dir,
        split="train",
        mode="all_images",
        train=False,
        image_size=args.image_size,
        augmentation="none",
        aspect_pad=args.aspect_pad,
    )
    validation_images = GarmentDataset(
        args.prepared_dir,
        split="val",
        mode="all_images",
        train=False,
        image_size=args.image_size,
        augmentation="none",
        aspect_pad=args.aspect_pad,
    )
    auxiliary_base: GarmentDataset | None = None
    auxiliary_train_images: GarmentDataset | None = None
    auxiliary_validation_images: GarmentDataset | None = None
    if args.aux_prepared_dir:
        auxiliary_base = GarmentDataset(
            args.aux_prepared_dir,
            split="train",
            mode="single",
            train=False,
            image_size=args.image_size,
            augmentation="none",
            aspect_pad=args.aspect_pad,
        )
        if auxiliary_base.schema != train_base.schema:
            raise RuntimeError(
                "Auxiliary schema differs from the target schema. Rerun "
                "compile_chatgarment_auxiliary.py against this prepared target."
            )
        auxiliary_train_images = GarmentDataset(
            args.aux_prepared_dir,
            split="train",
            mode="all_images",
            train=True,
            image_size=args.image_size,
            augmentation="domain",
            aspect_pad=args.aspect_pad,
        )
        auxiliary_validation_images = GarmentDataset(
            args.aux_prepared_dir,
            split="val",
            mode="all_images",
            train=False,
            image_size=args.image_size,
            augmentation="none",
            aspect_pad=args.aspect_pad,
        )
    dataset_image_paths(train_augmented_images, "target train")
    dataset_image_paths(validation_images, "target validation")
    assert_image_disjoint(
        train_augmented_images,
        "target train",
        validation_images,
        "target validation",
    )
    if auxiliary_train_images is not None:
        dataset_image_paths(auxiliary_train_images, "auxiliary train")
        assert_image_disjoint(
            auxiliary_train_images,
            "auxiliary train",
            validation_images,
            "target validation",
        )
    if (
        auxiliary_train_images is not None
        and auxiliary_validation_images is not None
        and len(auxiliary_validation_images) > 0
    ):
        dataset_image_paths(auxiliary_validation_images, "auxiliary validation")
        assert_image_disjoint(
            auxiliary_train_images,
            "auxiliary train",
            auxiliary_validation_images,
            "auxiliary validation",
        )

    validate_dataset_contract(train_base, "target train")
    validate_root_component_coverage(train_base, "target train")
    validate_dataset_contract(validation_images, "target validation")
    if auxiliary_base is not None:
        validate_dataset_contract(auxiliary_base, "auxiliary train")
    if (
        auxiliary_validation_images is not None
        and len(auxiliary_validation_images) > 0
    ):
        validate_dataset_contract(
            auxiliary_validation_images, "auxiliary validation"
        )

    constraints = build_constraints(
        train_base,
        () if auxiliary_base is None else (auxiliary_base,),
    )
    validate_route_targets(train_base, constraints, "target train")
    validate_route_targets(validation_images, constraints, "target validation")
    if auxiliary_base is not None:
        validate_route_targets(auxiliary_base, constraints, "auxiliary train")
    if (
        auxiliary_validation_images is not None
        and len(auxiliary_validation_images) > 0
    ):
        validate_route_targets(
            auxiliary_validation_images, constraints, "auxiliary validation"
        )
    config = GarmentEnsembleConfig(
        dimension=args.dimension,
        separate_route_branch=True,
        route_dimension=args.route_dimension,
        route_class_token_scale=args.route_class_token_scale,
        query_layers=args.query_layers,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        member_count=args.members,
        backbone_name=args.backbone_name,
        backbone_repo=args.backbone_repo,
        freeze_backbone=True,
        class_token_scale=args.class_token_scale,
    )
    model = GarmentTreeEnsemble(train_base.schema, constraints, config).to(device)
    model_preflight(model, args, device)
    weights = objective_weights(train_base).to(device)
    run_config = {
        "args": vars(args),
        "model_config": config.to_dict(),
        "ensemble_aggregation": "uniform_mean",
        "routes": constraints.to_dict(),
        "train_garments": len(train_base.garment_ids),
        "validation_images": len(validation_images),
        "trainable_parameters": count_trainable_parameters(model),
        "auxiliary_train_garments": (
            0 if auxiliary_base is None else len(auxiliary_base.garment_ids)
        ),
        "auxiliary_validation_images": (
            0
            if auxiliary_validation_images is None
            else len(auxiliary_validation_images)
        ),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    wandb_run = init_wandb(args, output_dir, run_config)

    train_cache = cache_features(
        model,
        train_augmented_images,
        output_dir / "feature_cache_train_augmented.pt",
        device,
        args,
        variant="train-domain",
        repeats=args.feature_cache_augmentations,
    )
    train_eval_cache = cache_features(
        model,
        train_evaluation_images,
        output_dir / "feature_cache_train_eval.pt",
        device,
        args,
        variant="train-eval",
    )
    validation_cache = cache_features(
        model,
        validation_images,
        output_dir / "feature_cache_val.pt",
        device,
        args,
        variant="validation",
    )
    auxiliary_cache: dict[str, Any] | None = None
    auxiliary_eval_cache: dict[str, Any] | None = None
    auxiliary_gids: list[str] = []
    if auxiliary_train_images is not None and auxiliary_base is not None:
        auxiliary_cache = cache_features(
            model,
            auxiliary_train_images,
            output_dir / "feature_cache_auxiliary_train.pt",
            device,
            args,
            variant="auxiliary-train-domain",
            repeats=args.aux_feature_cache_augmentations,
        )
        auxiliary_gids = sorted(set(auxiliary_base.garment_ids))
        if (
            auxiliary_validation_images is not None
            and len(auxiliary_validation_images) > 0
        ):
            auxiliary_eval_cache = cache_features(
                model,
                auxiliary_validation_images,
                output_dir / "feature_cache_auxiliary_val.pt",
                device,
                args,
                variant="auxiliary-validation",
            )
    tokens_by_gid = gid_tokens(train_eval_cache)
    coverage_core = make_coverage_core(tokens_by_gid)
    all_gids = sorted(tokens_by_gid)
    token_universe = set().union(*tokens_by_gid.values())
    fold_report = {
        "format": "garment_ensemble_coverage/v2",
        "folds": [],
        "output_token_count": len(token_universe),
        "coverage_core_garments": coverage_core,
        "coverage_core_size": len(coverage_core),
        "oof_stacking": "disabled",
        "deprecated_oof_options": {
            "folds": args.folds,
            "oof_epochs": args.oof_epochs,
            "stacker_steps": args.stacker_steps,
            "stacker_lr": args.stacker_lr,
        },
        "valid_routes": constraints.to_dict(),
    }
    member_training_gids = coverage_bagging_subsets(
        all_gids,
        coverage_core,
        args.members,
        args.bagging_fraction,
        args.seed,
    )
    bagging_report = []
    for member_index, subset in enumerate(member_training_gids):
        observed = set().union(*(tokens_by_gid[gid] for gid in subset))
        missing = sorted(token_universe - observed)
        if missing:
            raise RuntimeError(
                f"Bagged member {member_index} misses output tokens: {missing[:20]}"
            )
        bagging_report.append(
            {
                "member": member_index,
                "garments": len(subset),
                "fraction": len(subset) / max(len(all_gids), 1),
                "missing_output_tokens": missing,
            }
        )
    fold_report["bagging"] = bagging_report
    fold_report["ensemble_aggregation"] = "uniform_mean"
    (output_dir / "ensemble_coverage_report.json").write_text(
        json.dumps(fold_report, indent=2)
    )
    print(
        f"device={device} train_garments={len(all_gids)} val_images={len(validation_images)} "
        f"routes={len(constraints.valid_root_tuples)} members={args.members} oof=disabled "
        f"coverage_core={len(coverage_core)} "
        f"bagged_member_garments={[len(gids) for gids in member_training_gids]} "
        f"trainable={count_trainable_parameters(model):,} "
        f"auxiliary_garments={len(auxiliary_gids)} "
        f"auxiliary_fraction={args.aux_batch_fraction if auxiliary_gids else 0.0:.2f}",
        flush=True,
    )

    # OOF stacking was empirically inert because all heads made correlated
    # errors. Keep the independently initialized heads, but aggregate them with
    # an untrained uniform mean so no validation-unstable stacker is introduced.
    for parameter in model.stacker_parameters():
        parameter.data.zero_()
        parameter.requires_grad_(False)
    print("ensemble aggregation=uniform_mean; OOF stacker disabled", flush=True)
    final_result = train_final_ensemble(
        model,
        train_cache,
        train_eval_cache,
        validation_cache,
        member_training_gids,
        weights,
        args,
        device,
        output_dir,
        wandb_run,
        auxiliary_cache=auxiliary_cache,
        auxiliary_eval_cache=auxiliary_eval_cache,
        auxiliary_gids=auxiliary_gids,
    )
    member_metrics = {
        str(index): evaluate_cached(
            model, validation_cache, weights, args, device, member_only=index
        )
        for index in range(args.members)
    }
    metrics = final_result["validation"]
    train_loss = final_result["train"]["loss"]
    final_train_metrics = final_result["last_member_optimisation"]
    if metrics["invalid_path_emission_rate"] != 0.0:
        raise RuntimeError("Hard route constraints allowed an invalid path emission")
    if metrics["ground_truth_route_block_rate"] != 0.0:
        raise RuntimeError(
            "Route constraints block active ground-truth fields; the schema or labels are inconsistent"
        )
    checkpoint = {
        "format": "GarmentTreeEnsemble/checkpoint-v1",
        "schema": train_base.schema,
        "model_config": config.to_dict(),
        "ensemble_aggregation": "uniform_mean",
        "route_constraints": constraints.to_dict(),
        "model": model.state_dict(),
        "train_args": vars(args),
        "validation": metrics,
        "final_train": final_result["train"],
        "best_epoch": final_result["best_epoch"],
        "best_checkpoint_source": final_result["best_checkpoint_source"],
        "checkpoint_selection": final_result["checkpoint_selection"],
        "training_curve": final_result["curve"],
        "auxiliary_pretrain": final_result["auxiliary_pretrain"],
        "member_validation": member_metrics,
        "fold_coverage": fold_report,
    }
    atomic_save(checkpoint, output_dir / "best.pt")
    (output_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "train_metrics.json").write_text(
        json.dumps(
            {
                "best_epoch": final_result["best_epoch"],
                "best_checkpoint_source": final_result["best_checkpoint_source"],
                "train": final_result["train"],
                "last_member_optimisation": final_train_metrics,
                "auxiliary_pretrain": final_result["auxiliary_pretrain"],
            },
            indent=2,
        )
    )
    print(f"train loss={train_loss:.4f}", flush=True)
    print("validation " + " ".join(f"{key}={value:.4f}" for key, value in metrics.items()), flush=True)
    print(f"wrote {output_dir / 'best.pt'}", flush=True)
    if wandb_run is not None:

        wandb_run.summary["train_loss"] = train_loss
        wandb_run.summary["val_loss"] = metrics["loss"]
        wandb_run.finish()


if __name__ == "__main__":
    main()











