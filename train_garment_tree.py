#!/usr/bin/env python3
"""Train GarmentTreeNet on one pose image at a time."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Sampler
import torch.nn.functional as F

from garment_tree_model import GarmentTreeConfig, GarmentTreeNet, count_parameters
from prepare_data import GarmentDataset


@dataclass
class ObjectiveWeights:
    numeric_field: Tensor
    numeric_activity_positive: Tensor
    categorical_class: list[Tensor]
    categorical_activity_positive: Tensor
    categorical_ordinal_values: list[Tensor | None]

    def to(self, device: torch.device) -> "ObjectiveWeights":
        return ObjectiveWeights(
            self.numeric_field.to(device),
            self.numeric_activity_positive.to(device),
            [weight.to(device) for weight in self.categorical_class],
            self.categorical_activity_positive.to(device),
            [
                None if values is None else values.to(device)
                for values in self.categorical_ordinal_values
            ],
        )


class PosePairDataset(Dataset):
    """One supervised image plus a different-pose consistency view."""

    def __init__(self, base: GarmentDataset) -> None:
        if base.mode != "single":
            raise ValueError("PosePairDataset requires a single-mode GarmentDataset")
        self.base = base

    def __len__(self) -> int:
        return len(self.base.garment_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        gid = self.base.garment_ids[index]
        _, views = self.base._pick_frame(self.base.images[gid]["frames"])
        present = [path for path in views if path is not None]
        if len(present) < 2:
            raise RuntimeError(f"{gid} needs at least two pose images for consistency training")
        order = torch.randperm(len(present))[:2].tolist()
        row = self.base.row_of[gid]
        return {
            "gid": gid,
            "image": self.base._load_image(present[order[0]]),
            "image_alt": self.base._load_image(present[order[1]]),
            "y_cont": torch.from_numpy(self.base.y_cont[row].copy()),
            "mask": torch.from_numpy(self.base.mask[row].copy()),
            "y_const": torch.from_numpy(self.base.y_const[row].copy()),
            "const_mask": torch.from_numpy(self.base.const_mask[row].copy()),
            "y_cat": torch.from_numpy(self.base.y_cat[row].copy()),
        }


def signature_for_gid(dataset: GarmentDataset, gid: str) -> tuple[str, ...]:
    metadata = dataset.images[gid].get("meta", ())
    return tuple("<null>" if value is None else str(value) for value in metadata)


class BalancedOutputSampler(Sampler[int]):
    """Draw every compiled output-value anchor exactly equally each epoch."""

    def __init__(
        self, dataset: GarmentDataset, prepared_dir: str, seed: int
    ) -> None:
        self.seed = seed
        self.epoch = 0
        plan_path = Path(prepared_dir) / "balanced_sampling.json"
        if not plan_path.is_file():
            raise RuntimeError(
                f"Missing equalized sampling plan: {plan_path}. Rerun the compiler."
            )
        plan = json.loads(plan_path.read_text())
        if plan.get("format") != "equal_output_token_anchors/v1":
            raise RuntimeError(
                f"Unsupported balanced sampling format in {plan_path}: "
                f"{plan.get('format')!r}"
            )
        index_by_gid = {gid: index for index, gid in enumerate(dataset.garment_ids)}
        self.groups: list[list[int]] = []
        self.tokens: list[tuple[str, str, str]] = []
        missing_groups = []
        for group in plan.get("groups", []):
            indices = [
                index_by_gid[gid]
                for gid in group.get("garment_ids", [])
                if gid in index_by_gid
            ]
            if not indices:
                missing_groups.append(group.get("token"))
                continue
            self.groups.append(indices)
            self.tokens.append(tuple(map(str, group["token"])))
        if missing_groups:
            raise RuntimeError(
                "Balanced sampling groups have no training garment: "
                + json.dumps(missing_groups[:20])
            )
        if not self.groups:
            raise RuntimeError("Balanced sampling plan contains no usable groups")
        self.anchor_draws_per_group = int(
            plan["anchor_draws_per_group_per_epoch"]
        )
        if self.anchor_draws_per_group < 1:
            raise RuntimeError("anchor_draws_per_group_per_epoch must be positive")
        self.length = self.anchor_draws_per_group * len(self.groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        output: list[int] = []
        quota = self.anchor_draws_per_group
        for group in self.groups:
            selected: list[int] = []
            while len(selected) < quota:
                order = torch.randperm(len(group), generator=generator).tolist()
                selected.extend(group[index] for index in order)
            output.extend(selected[:quota])
        permutation = torch.randperm(len(output), generator=generator).tolist()
        return iter(output[index] for index in permutation)

class ExponentialAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.trainable_names = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        for name, averaged in self.model.state_dict().items():
            if name not in self.trainable_names:
                continue
            value = current[name].detach()
            if averaged.is_floating_point():
                averaged.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                averaged.copy_(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--min-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-6)
    parser.add_argument("--warmup-epochs", type=int, default=12)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--lambda-category", type=float, default=0.35)
    parser.add_argument("--lambda-ordinal", type=float, default=0.15)
    parser.add_argument("--lambda-activity", type=float, default=0.08)
    parser.add_argument("--lambda-uncertainty", type=float, default=0.02)
    parser.add_argument("--lambda-consistency", type=float, default=0.12)
    parser.add_argument("--teacher-force-epochs", type=int, default=180)
    parser.add_argument("--no-pose-consistency", action="store_true")
    parser.add_argument(
        "--augmentation", choices=("none", "light", "domain"), default="light"
    )
    parser.add_argument(
        "--aspect-pad", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--encoder-kind", choices=("scratch", "foundation"), default="scratch"
    )
    parser.add_argument("--backbone-name", default="dinov2_vits14")
    parser.add_argument("--backbone-repo", default="DINOv2")
    parser.add_argument(
        "--freeze-backbone", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--hierarchical-categoricals",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--widths", type=int, nargs=4, default=(64, 128, 256, 384))
    parser.add_argument("--depths", type=int, nargs=4, default=(2, 2, 6, 2))
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--query-layers", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--drop-path", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--resume")
    parser.add_argument("--allow-unbalanced-data", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--train-eval-every",
        type=int,
        default=0,
        help=(
            "run deterministic eval-mode metrics on every training image every N "
            "epochs; 0 disables"
        ),
    )
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="ImageToGarment")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--wandb-tags", nargs="*", default=("garment-tree",))
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def readiness_gate(prepared_dir: str, override: bool) -> None:
    report_path = Path(prepared_dir) / "balance_report.json"
    if not report_path.is_file():
        raise SystemExit(f"Missing balance audit: {report_path}")
    report = json.loads(report_path.read_text())
    compiler = str(report.get("compiler", ""))
    if compiler.startswith("output_balanced_garmentcode_smplx/"):
        split = report.get("split_selection", {})
        if (
            compiler != "output_balanced_garmentcode_smplx/v4"
            or split.get("unseen_validation_categorical_classes") != 0
            or split.get("missing_train_garment_categories") != 0
            or split.get("unseen_train_output_tokens") != 0
            or not report.get("coverage", {}).get(
                "all_observed_garment_categories_in_train", False
            )
            or not report.get("coverage", {}).get(
                "all_observed_output_tokens_in_train", False
            )
        ):
            raise SystemExit(
                "The prepared data does not satisfy strict v4 coverage/equalization. "
                "Rerun: bash scripts/compile_output_balanced_smplx_data.sh"
            )
    if not report.get("ready", False) and not override:
        raise SystemExit(
            "The balance audit is not training-ready. Fix the listed deficits or "
            "pass --allow-unbalanced-data for an explicitly diagnostic run."
        )
    if not report.get("ready", False):
        print("WARNING: training with a failed balance audit", flush=True)


def is_numeric_select_vocab(vocab: list[Any]) -> bool:
    return (
        len(vocab) >= 3
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in vocab
        )
        and len({float(value) for value in vocab}) == len(vocab)
    )

def objective_weights(dataset: GarmentDataset) -> ObjectiveWeights:
    rows = np.asarray([dataset.row_of[gid] for gid in dataset.garment_ids])
    numeric_mask = np.concatenate((dataset.mask[rows], dataset.const_mask[rows]), axis=1)
    active = numeric_mask.sum(axis=0)
    numeric_field = np.sqrt(max(len(rows), 1) / np.maximum(active, 1.0))
    numeric_field /= max(float(numeric_field.mean()), 1e-8)
    numeric_field = np.clip(numeric_field, 0.5, 3.0)
    numeric_positive = (len(rows) - active) / np.maximum(active, 1.0)
    numeric_positive = np.clip(numeric_positive, 1.0, 8.0)

    categories = dataset.y_cat[rows]
    class_weights = []
    cat_active = []
    ordinal_values: list[Tensor | None] = []
    for index, vocab in enumerate(dataset.schema["cat_vocab"].values()):
        values = categories[:, index]
        values = values[values >= 0]
        counts = np.bincount(values, minlength=len(vocab)).astype(np.float32)
        weights = np.sqrt(max(len(values), 1) / np.maximum(counts, 1.0))
        weights /= max(float(weights.mean()), 1e-8)
        class_weights.append(torch.tensor(np.clip(weights, 0.33, 3.0)))
        cat_active.append(len(values))
        if is_numeric_select_vocab(vocab):
            ordered = torch.tensor([float(value) for value in vocab])
            span = ordered.max() - ordered.min()
            ordinal_values.append(
                (ordered - ordered.min()) / span.clamp_min(1e-8)
            )
        else:
            ordinal_values.append(None)
    cat_active_array = np.asarray(cat_active, dtype=np.float32)
    cat_positive = (len(rows) - cat_active_array) / np.maximum(cat_active_array, 1.0)
    cat_positive = np.clip(cat_positive, 1.0, 8.0)
    return ObjectiveWeights(
        torch.tensor(numeric_field, dtype=torch.float32),
        torch.tensor(numeric_positive, dtype=torch.float32),
        class_weights,
        torch.tensor(cat_positive, dtype=torch.float32),
        ordinal_values,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def repeat_targets(batch: dict[str, Any]) -> dict[str, Any]:
    repeated = dict(batch)
    for key in ("y_cont", "mask", "y_const", "const_mask", "y_cat"):
        repeated[key] = torch.cat((batch[key], batch[key]), dim=0)
    return repeated


def split_output(output: dict[str, Any], batch_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    first: dict[str, Any] = {}
    second: dict[str, Any] = {}
    for key, value in output.items():
        if isinstance(value, list):
            first[key] = [field[:batch_size] for field in value]
            second[key] = [field[batch_size:] for field in value]
        else:
            first[key] = value[:batch_size]
            second[key] = value[batch_size:]
    return first, second


def supervised_objective(
    output: dict[str, Any],
    batch: dict[str, Any],
    weights: ObjectiveWeights,
    args: argparse.Namespace,
) -> tuple[Tensor, dict[str, Tensor]]:
    target_numeric = torch.cat((batch["y_cont"], batch["y_const"]), dim=-1)
    numeric_mask = torch.cat(
        (batch["mask"], batch["const_mask"]), dim=-1
    ).to(output["numeric_mean"].dtype)
    elementwise = F.smooth_l1_loss(
        output["numeric_mean"], target_numeric, reduction="none", beta=0.04
    )
    weighted_mask = numeric_mask * weights.numeric_field
    loss_numeric = (elementwise * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)

    predicted_scale = output["numeric_log_scale"].exp()
    absolute_error = (output["numeric_mean"] - target_numeric).abs().detach()
    loss_uncertainty = (
        F.smooth_l1_loss(predicted_scale, absolute_error, reduction="none", beta=0.03)
        * weighted_mask
    ).sum() / weighted_mask.sum().clamp_min(1.0)

    numeric_activity_loss = F.binary_cross_entropy_with_logits(
        output["numeric_activity"],
        numeric_mask,
        pos_weight=weights.numeric_activity_positive,
    )
    categorical_active = batch["y_cat"].ge(0).to(output["numeric_mean"].dtype)
    categorical_activity_loss = F.binary_cross_entropy_with_logits(
        output["categorical_activity"],
        categorical_active,
        pos_weight=weights.categorical_activity_positive,
    )
    loss_activity = 0.5 * (numeric_activity_loss + categorical_activity_loss)

    categorical_losses = []
    ordinal_losses = []
    categorical_field_indices = getattr(
        args, "categorical_field_indices", range(len(output["categorical_logits"]))
    )
    for index in categorical_field_indices:
        logits = output["categorical_logits"][index]
        target = batch["y_cat"][:, index]
        active_field = target.ge(0)
        if active_field.any() and logits.shape[-1] > 1:
            active_logits = logits[active_field]
            active_target = target[active_field]
            log_probability = active_logits.log_softmax(dim=-1)
            target_nll = -log_probability.gather(
                1, active_target[:, None]
            ).squeeze(1)
            smooth_nll = -log_probability.mean(dim=-1)
            sample_loss = (
                (1.0 - args.label_smoothing) * target_nll
                + args.label_smoothing * smooth_nll
            )
            ordinal_values = weights.categorical_ordinal_values[index]
            sample_ordinal = None
            if ordinal_values is not None:
                target_values = ordinal_values[active_target]
                distances = (
                    ordinal_values.unsqueeze(0) - target_values.unsqueeze(1)
                ).abs()
                sample_ordinal = (
                    log_probability.exp().float() * distances.float()
                ).sum(dim=-1)
            # The compiler already guarantees every class is sampled. Apply the
            # precomputed, tempered global class weights once here instead of
            # re-equalizing whichever classes happen to occur in each batch.
            # Batch-local class averaging strongly over-amplified one-example
            # classes and made the loss depend on batch composition.
            class_weight = weights.categorical_class[index][active_target]
            denominator = class_weight.sum().clamp_min(1e-8)
            categorical_losses.append(
                (sample_loss * class_weight).sum() / denominator
            )
            if sample_ordinal is not None:
                ordinal_losses.append(
                    (sample_ordinal * class_weight).sum() / denominator
                )
    loss_category = (
        torch.stack(categorical_losses).mean()
        if categorical_losses
        else output["numeric_mean"].new_zeros(())
    )
    loss_ordinal = (
        torch.stack(ordinal_losses).mean()
        if ordinal_losses
        else output["numeric_mean"].new_zeros(())
    )
    total = (
        loss_numeric
        + args.lambda_category * loss_category
        + args.lambda_ordinal * loss_ordinal
        + args.lambda_activity * loss_activity
        + args.lambda_uncertainty * loss_uncertainty
    )
    return total, {
        "numeric": loss_numeric.detach(),
        "category": loss_category.detach(),
        "ordinal": loss_ordinal.detach(),
        "activity": loss_activity.detach(),
        "uncertainty": loss_uncertainty.detach(),
    }


def consistency_objective(
    first: dict[str, Any], second: dict[str, Any], batch: dict[str, Any]
) -> Tensor:
    numeric_mask = torch.cat((batch["mask"], batch["const_mask"]), dim=-1)
    numeric = (
        F.smooth_l1_loss(
            first["numeric_mean"], second["numeric_mean"], reduction="none", beta=0.02
        )
        * numeric_mask
    ).sum() / numeric_mask.sum().clamp_min(1.0)
    divergences = []
    for index, (left, right) in enumerate(
        zip(first["categorical_logits"], second["categorical_logits"])
    ):
        active = batch["y_cat"][:, index].ge(0)
        if not active.any() or left.shape[-1] == 1:
            continue
        left_log = left[active].log_softmax(dim=-1)
        right_log = right[active].log_softmax(dim=-1)
        left_prob = left_log.exp()
        right_prob = right_log.exp()
        middle_log = (0.5 * (left_prob + right_prob)).clamp_min(1e-8).log()
        divergences.append(
            0.5
            * (
                F.kl_div(middle_log, left_prob, reduction="batchmean")
                + F.kl_div(middle_log, right_prob, reduction="batchmean")
            )
        )
    categorical = torch.stack(divergences).mean() if divergences else numeric.new_zeros(())
    feature = 1.0 - F.cosine_similarity(
        first["part_features"], second["part_features"], dim=-1
    ).mean()
    return numeric + 0.25 * categorical + 0.05 * feature


def teacher_force_ratio(epoch: int, duration: int) -> float:
    if duration <= 0:
        return 0.0
    progress = min(max(epoch / duration, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def learning_rate(epoch: int, args: argparse.Namespace) -> float:
    if epoch < args.warmup_epochs:
        return args.lr * (epoch + 1) / max(args.warmup_epochs, 1)
    progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return args.min_lr + (args.lr - args.min_lr) * cosine


def train_epoch(
    model: GarmentTreeNet,
    ema: ExponentialAverage,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    weights: ObjectiveWeights,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    model.train()
    totals: Counter[str] = Counter()
    examples = 0
    ratio = teacher_force_ratio(epoch, args.teacher_force_epochs)
    use_amp = args.amp and device.type == "cuda"
    for step, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        batch_size = batch["image"].shape[0]
        optimizer.zero_grad(set_to_none=True)
        if "image_alt" in batch:
            images = torch.cat((batch["image"], batch["image_alt"]), dim=0)
            targets = repeat_targets(batch)
        else:
            images = batch["image"]
            targets = batch
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(images, targets["y_cat"], teacher_force=ratio)
            loss, components = supervised_objective(output, targets, weights, args)
            consistency = loss.new_zeros(())
            if "image_alt" in batch:
                first, second = split_output(output, batch_size)
                consistency = consistency_objective(first, second, batch)
                loss = loss + args.lambda_consistency * consistency
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        examples += batch_size
        totals["loss"] += float(loss.detach()) * batch_size
        totals["consistency"] += float(consistency.detach()) * batch_size
        for name, value in components.items():
            totals[name] += float(value) * batch_size
        if step % args.log_every == 0:
            print(
                f"  step={step:04d}/{len(loader):04d} loss={float(loss):.4f} "
                f"num={float(components['numeric']):.4f} "
                f"cat={float(components['category']):.4f} "
                f"ord={float(components['ordinal']):.4f} tf={ratio:.3f}",
                flush=True,
            )
    return {key: value / max(examples, 1) for key, value in totals.items()}


def macro_binary_f1(tp: Tensor, fp: Tensor, fn: Tensor) -> float:
    denominator = 2.0 * tp + fp + fn
    valid = denominator > 0
    if not valid.any():
        return 1.0
    return float((2.0 * tp[valid] / denominator[valid].clamp_min(1.0)).mean())


@torch.no_grad()
def validate(
    model: GarmentTreeNet,
    loader: DataLoader,
    device: torch.device,
    weights: ObjectiveWeights,
    args: argparse.Namespace,
    schema: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    evaluation_args = copy.copy(args)
    evaluation_args.label_smoothing = 0.0
    objective_outputs: list[dict[str, Any]] = []
    objective_batches: list[dict[str, Tensor]] = []
    examples = 0
    numeric_count = len(schema.get("cont_slots", {})) + len(schema.get("const_slots", {}))
    categorical_count = len(schema["cat_vocab"])
    numeric_error_by_field = torch.zeros(numeric_count, dtype=torch.float64)
    numeric_active_by_field = torch.zeros(numeric_count, dtype=torch.float64)
    numeric_tp = torch.zeros(numeric_count, dtype=torch.float64)
    numeric_fp = torch.zeros(numeric_count, dtype=torch.float64)
    numeric_fn = torch.zeros(numeric_count, dtype=torch.float64)
    categorical_correct_by_field = torch.zeros(categorical_count, dtype=torch.float64)
    categorical_active_by_field = torch.zeros(categorical_count, dtype=torch.float64)
    categorical_nll_sum_by_field = torch.zeros(categorical_count, dtype=torch.float64)
    ordinal_error_by_field = torch.zeros(categorical_count, dtype=torch.float64)
    ordinal_active_by_field = torch.zeros(categorical_count, dtype=torch.float64)
    categorical_tp = torch.zeros(categorical_count, dtype=torch.float64)
    categorical_fp = torch.zeros(categorical_count, dtype=torch.float64)
    categorical_fn = torch.zeros(categorical_count, dtype=torch.float64)
    root_correct = 0
    root_examples = 0
    by_gid: dict[str, list[tuple[Tensor, Tensor]]] = defaultdict(list)
    root_indices = [
        index
        for index, path in enumerate(schema["cat_vocab"])
        if path in {"meta.upper", "meta.wb", "meta.bottom"}
    ]
    use_amp = args.amp and device.type == "cuda"
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        batch_size = batch["image"].shape[0]
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(batch["image"])
        objective_outputs.append(
            {
                "numeric_mean": output["numeric_mean"].detach().float().cpu(),
                "numeric_log_scale": (
                    output["numeric_log_scale"].detach().float().cpu()
                ),
                "numeric_activity": (
                    output["numeric_activity"].detach().float().cpu()
                ),
                "categorical_logits": [
                    logits.detach().float().cpu()
                    for logits in output["categorical_logits"]
                ],
                "categorical_activity": (
                    output["categorical_activity"].detach().float().cpu()
                ),
            }
        )
        objective_batches.append(
            {
                key: batch[key].detach().cpu()
                for key in ("y_cont", "mask", "y_const", "const_mask", "y_cat")
            }
        )
        examples += batch_size

        target_numeric = torch.cat((batch["y_cont"], batch["y_const"]), dim=-1)
        numeric_mask = torch.cat((batch["mask"], batch["const_mask"]), dim=-1).bool()
        absolute_error = (output["numeric_mean"].float() - target_numeric.float()).abs()
        numeric_error_by_field += (
            absolute_error * numeric_mask
        ).sum(dim=0).detach().cpu().double()
        numeric_active_by_field += numeric_mask.sum(dim=0).detach().cpu().double()

        predicted_numeric_active = output["numeric_activity"].sigmoid().ge(0.5)
        numeric_tp += (
            predicted_numeric_active & numeric_mask
        ).sum(dim=0).detach().cpu().double()
        numeric_fp += (
            predicted_numeric_active & ~numeric_mask
        ).sum(dim=0).detach().cpu().double()
        numeric_fn += (
            ~predicted_numeric_active & numeric_mask
        ).sum(dim=0).detach().cpu().double()

        truth_categorical_active = batch["y_cat"].ge(0)
        predicted_categorical_active = output["categorical_activity"].sigmoid().ge(0.5)
        categorical_tp += (
            predicted_categorical_active & truth_categorical_active
        ).sum(dim=0).detach().cpu().double()
        categorical_fp += (
            predicted_categorical_active & ~truth_categorical_active
        ).sum(dim=0).detach().cpu().double()
        categorical_fn += (
            ~predicted_categorical_active & truth_categorical_active
        ).sum(dim=0).detach().cpu().double()

        root_batch = torch.ones(batch_size, dtype=torch.bool, device=device)
        for index, logits in enumerate(output["categorical_logits"]):
            target = batch["y_cat"][:, index]
            active = truth_categorical_active[:, index]
            predicted = logits.argmax(dim=-1)
            if active.any() and logits.shape[-1] > 1:
                categorical_nll_sum_by_field[index] += F.cross_entropy(
                    logits[active].float(),
                    target[active],
                    reduction="sum",
                ).detach().cpu().double()
                ordinal_values = weights.categorical_ordinal_values[index]
                if ordinal_values is not None:
                    expected_value = (
                        logits[active].float().softmax(dim=-1)
                        * ordinal_values.float().unsqueeze(0)
                    ).sum(dim=-1)
                    target_value = ordinal_values[target[active]].float()
                    ordinal_error_by_field[index] += (
                        expected_value - target_value
                    ).abs().sum().detach().cpu().double()
                    ordinal_active_by_field[index] += active.sum().detach().cpu().double()
            categorical_correct_by_field[index] += (
                predicted[active] == target[active]
            ).sum().detach().cpu().double()
            categorical_active_by_field[index] += active.sum().detach().cpu().double()
            if index in root_indices:
                field_correct = predicted_categorical_active[:, index].eq(active)
                field_correct &= ~active | predicted.eq(target.clamp_min(0))
                root_batch &= field_correct
        root_correct += int(root_batch.sum())
        root_examples += batch_size

        for gid, prediction, active_mask in zip(
            batch["gid"],
            output["numeric_mean"].detach().cpu().float(),
            numeric_mask.detach().cpu(),
        ):
            by_gid[str(gid)].append((prediction, active_mask))

    pose_stabilities = []
    for predictions_and_masks in by_gid.values():
        if len(predictions_and_masks) < 2:
            continue
        predictions = torch.stack([item[0] for item in predictions_and_masks])
        active = predictions_and_masks[0][1].bool()
        if active.any():
            pose_stabilities.append(
                predictions[:, active].std(dim=0, unbiased=False).mean()
            )

    if not objective_outputs:
        raise ValueError("Validation loader produced no batches")
    full_output = {
        key: torch.cat([item[key] for item in objective_outputs], dim=0)
        for key in (
            "numeric_mean",
            "numeric_log_scale",
            "numeric_activity",
            "categorical_activity",
        )
    }
    full_output["categorical_logits"] = [
        torch.cat(
            [item["categorical_logits"][field] for item in objective_outputs],
            dim=0,
        )
        for field in range(categorical_count)
    ]
    full_batch = {
        key: torch.cat([item[key] for item in objective_batches], dim=0)
        for key in ("y_cont", "mask", "y_const", "const_mask", "y_cat")
    }
    loss, components = supervised_objective(
        full_output, full_batch, weights.to(torch.device("cpu")), evaluation_args
    )
    metrics = {"loss": float(loss)}
    metrics.update({name: float(value) for name, value in components.items()})
    active_numeric_fields = numeric_active_by_field > 0
    per_field_numeric_mae = (
        numeric_error_by_field[active_numeric_fields]
        / numeric_active_by_field[active_numeric_fields].clamp_min(1.0)
    )
    metrics["numeric_mae"] = float(
        numeric_error_by_field.sum() / numeric_active_by_field.sum().clamp_min(1.0)
    )
    metrics["numeric_macro_mae"] = (
        float(per_field_numeric_mae.mean()) if per_field_numeric_mae.numel() else 0.0
    )
    active_categorical_fields = categorical_active_by_field > 0
    per_field_categorical_accuracy = (
        categorical_correct_by_field[active_categorical_fields]
        / categorical_active_by_field[active_categorical_fields].clamp_min(1.0)
    )
    metrics["categorical_accuracy"] = float(
        categorical_correct_by_field.sum()
        / categorical_active_by_field.sum().clamp_min(1.0)
    )
    per_field_categorical_nll = (
        categorical_nll_sum_by_field[active_categorical_fields]
        / categorical_active_by_field[active_categorical_fields].clamp_min(1.0)
    )
    metrics["categorical_macro_nll"] = (
        float(per_field_categorical_nll.mean())
        if per_field_categorical_nll.numel()
        else 0.0
    )
    for index, path in enumerate(schema["cat_vocab"]):
        if categorical_active_by_field[index] > 0:
            metrics[f"category_nll/{path}"] = float(
                categorical_nll_sum_by_field[index]
                / categorical_active_by_field[index]
            )
    active_ordinal_fields = ordinal_active_by_field > 0
    per_field_ordinal_mae = (
        ordinal_error_by_field[active_ordinal_fields]
        / ordinal_active_by_field[active_ordinal_fields].clamp_min(1.0)
    )
    metrics["ordinal_macro_mae"] = (
        float(per_field_ordinal_mae.mean())
        if per_field_ordinal_mae.numel()
        else 0.0
    )
    for index, path in enumerate(schema["cat_vocab"]):
        if ordinal_active_by_field[index] > 0:
            metrics[f"ordinal_mae/{path}"] = float(
                ordinal_error_by_field[index]
                / ordinal_active_by_field[index]
            )
    metrics["categorical_macro_accuracy"] = (
        float(per_field_categorical_accuracy.mean())
        if per_field_categorical_accuracy.numel()
        else 0.0
    )
    metrics["numeric_activity_macro_f1"] = macro_binary_f1(
        numeric_tp, numeric_fp, numeric_fn
    )
    metrics["categorical_activity_macro_f1"] = macro_binary_f1(
        categorical_tp, categorical_fp, categorical_fn
    )
    metrics["root_exact"] = root_correct / max(root_examples, 1)
    metrics["pose_numeric_std"] = (
        float(torch.stack(pose_stabilities).mean()) if pose_stabilities else 0.0
    )
    metrics["score"] = (
        metrics["numeric_macro_mae"]
        + 0.20 * (1.0 - metrics["categorical_macro_accuracy"])
        + 0.15 * (1.0 - metrics["root_exact"])
        + 0.10 * (1.0 - metrics["categorical_activity_macro_f1"])
        + 0.05 * (1.0 - metrics["numeric_activity_macro_f1"])
        + 0.50 * metrics["pose_numeric_std"]
    )
    return metrics

def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in state["cuda"]]
        )


def checkpoint_payload(
    model: GarmentTreeNet,
    ema: ExponentialAverage,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    schema: dict[str, Any],
    args: argparse.Namespace,
    epoch: int,
    best_score: float,
    stale_epochs: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "format": "GarmentTreeNet/checkpoint-v1",
        "epoch": epoch,
        "best_score": best_score,
        "model": model.state_dict(),
        "ema_model": ema.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "stale_epochs": stale_epochs,
        "rng_state": capture_rng_state(),
        "schema": schema,
        "model_config": model.config.to_dict(),
        "train_args": vars(args),
        "validation": metrics,
    }


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def init_wandb(
    args: argparse.Namespace,
    output_dir: Path,
    model: GarmentTreeNet,
    train_base: GarmentDataset,
    validation_dataset: GarmentDataset,
    sampler: Sampler[int],
) -> Any | None:
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "W&B logging was requested, but wandb is not installed. "
            "Install it with: pip install wandb"
        ) from exc

    private_config_keys = {"prepared_dir", "out_dir", "resume"}
    config = {
        key: value for key, value in vars(args).items()
        if key not in private_config_keys
    }
    config.update(
        {
            "train_garments": len(train_base.garment_ids),
            "train_items_per_epoch": len(sampler),
            "val_garments": len(validation_dataset.garment_ids),
            "val_images": len(validation_dataset),
            "parameters": count_parameters(model),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "numeric_outputs": len(model.numeric_paths),
            "categorical_outputs": len(model.categorical_paths),
        }
    )
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or output_dir.name,
        dir=str(output_dir),
        mode=args.wandb_mode,
        tags=list(args.wandb_tags),
        config=config,
    )
    wandb.define_metric("epoch")
    for prefix in (
        "train",
        "train_eval",
        "val",
        "gap",
        "eval_gap",
        "best",
        "optim",
        "schedule",
        "timing",
        "early_stopping",
    ):
        wandb.define_metric(f"{prefix}/*", step_metric="epoch")
    return run


def log_wandb_epoch(
    run: Any | None,
    epoch: int,
    train_metrics: dict[str, float],
    train_evaluation_metrics: dict[str, float] | None,
    validation_metrics: dict[str, float],
    lr: float,
    teacher_force: float,
    best_score: float,
    improved: bool,
    stale_epochs: int,
    seconds: float,
) -> None:
    if run is None:
        return
    payload: dict[str, float | int] = {
        "epoch": epoch,
        "optim/lr": lr,
        "schedule/teacher_force_ratio": teacher_force,
        "best/score": best_score,
        "best/improved": int(improved),
        "timing/epoch_seconds": seconds,
        "early_stopping/stale_epochs": stale_epochs,
    }
    payload.update({f"train/{key}": value for key, value in train_metrics.items()})
    if train_evaluation_metrics is not None:
        payload.update(
            {
                f"train_eval/{key}": value
                for key, value in train_evaluation_metrics.items()
            }
        )
        for key in ("loss", "category", "ordinal", "numeric"):
            payload[f"eval_gap/{key}"] = (
                validation_metrics[key] - train_evaluation_metrics[key]
            )
    payload.update({f"val/{key}": value for key, value in validation_metrics.items()})
    payload["gap/loss"] = validation_metrics["loss"] - train_metrics["loss"]
    payload["gap/category"] = (
        validation_metrics["category"] - train_metrics["category"]
    )
    payload["gap/ordinal"] = validation_metrics["ordinal"] - train_metrics["ordinal"]
    payload["gap/numeric"] = validation_metrics["numeric"] - train_metrics["numeric"]
    run.log(payload, step=epoch)
    run.summary["best/score"] = best_score


def main() -> None:
    args = parse_args()
    if args.dimension % args.attention_heads:
        raise SystemExit("--dimension must be divisible by --attention-heads")
    seed_everything(args.seed)
    readiness_gate(args.prepared_dir, args.allow_unbalanced_data)
    device = choose_device(args.device)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_base = GarmentDataset(
        args.prepared_dir,
        split="train",
        mode="single",
        train=True,
        image_size=args.image_size,
        augmentation=args.augmentation,
        aspect_pad=args.aspect_pad,
    )
    train_dataset: Dataset = (
        train_base if args.no_pose_consistency else PosePairDataset(train_base)
    )
    validation_dataset = GarmentDataset(
        args.prepared_dir,
        split="val",
        mode="all_images",
        train=False,
        image_size=args.image_size,
        augmentation="none",
        aspect_pad=args.aspect_pad,
    )
    train_evaluation_dataset = None
    if args.train_eval_every > 0:
        train_evaluation_dataset = GarmentDataset(
            args.prepared_dir,
            split="train",
            mode="all_images",
            train=False,
            image_size=args.image_size,
            augmentation="none",
            aspect_pad=args.aspect_pad,
        )
    sampler = BalancedOutputSampler(train_base, args.prepared_dir, args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, drop_last=True, **loader_options)
    val_loader = DataLoader(validation_dataset, shuffle=False, drop_last=False, **loader_options)
    train_evaluation_loader = (
        DataLoader(
            train_evaluation_dataset,
            shuffle=False,
            drop_last=False,
            **loader_options,
        )
        if train_evaluation_dataset is not None
        else None
    )

    config = GarmentTreeConfig(
        widths=tuple(args.widths),
        depths=tuple(args.depths),
        dimension=args.dimension,
        query_layers=args.query_layers,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        drop_path=args.drop_path,
        encoder_kind=args.encoder_kind,
        backbone_name=args.backbone_name,
        backbone_repo=args.backbone_repo,
        freeze_backbone=args.freeze_backbone,
        hierarchical_categoricals=args.hierarchical_categoricals,
    )
    model = GarmentTreeNet(train_base.schema, config).to(device)
    ema = ExponentialAverage(model, args.ema_decay)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    weights = objective_weights(train_base).to(device)
    start_epoch = 0
    best_score = math.inf
    stale_epochs = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get("schema") != train_base.schema:
            raise ValueError("Resume checkpoint schema differs from prepared data")
        if saved.get("model_config") != config.to_dict():
            raise ValueError("Resume checkpoint model configuration differs")
        model.load_state_dict(saved["model"])
        ema.model.load_state_dict(saved.get("ema_model", saved["model"]))
        optimizer.load_state_dict(saved["optimizer"])
        if "scaler" in saved:
            scaler.load_state_dict(saved["scaler"])
        start_epoch = int(saved["epoch"]) + 1
        best_score = float(saved.get("best_score", math.inf))
        stale_epochs = int(saved.get("stale_epochs", 0))
        restore_rng_state(saved.get("rng_state"))
        if start_epoch >= args.epochs:
            raise ValueError(
                f"Resume checkpoint already reached epoch {start_epoch}; "
                f"requested --epochs is {args.epochs}"
            )

    print(
        f"device={device} garments={len(train_base)} train_items={len(sampler)} "
        f"val_images={len(validation_dataset)} parameters={count_parameters(model):,} "
        f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        flush=True,
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "model_config": config.to_dict(),
                "schema": train_base.schema,
                "balanced_output_groups": len(sampler.groups),
                "anchor_draws_per_group": sampler.anchor_draws_per_group,
                "ordinal_categorical_paths": [
                    path
                    for path, vocab in train_base.schema["cat_vocab"].items()
                    if is_numeric_select_vocab(vocab)
                ],
            },
            indent=2,
        )
    )
    history_path = output_dir / "history.jsonl"
    wandb_run = init_wandb(
        args,
        output_dir,
        model,
        train_base,
        validation_dataset,
        sampler,
    )
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        sampler.set_epoch(epoch)
        lr = learning_rate(epoch, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        print(f"epoch={epoch + 1:04d}/{args.epochs} lr={lr:.3e}", flush=True)
        train_metrics = train_epoch(
            model, ema, train_loader, optimizer, scaler, device, weights, args, epoch
        )
        validation_metrics = validate(
            ema.model, val_loader, device, weights, args, train_base.schema
        )
        run_train_evaluation = (
            train_evaluation_loader is not None
            and (
                epoch == 0
                or (epoch + 1) % args.train_eval_every == 0
                or epoch + 1 == args.epochs
            )
        )
        train_evaluation_metrics = (
            validate(
                ema.model,
                train_evaluation_loader,
                device,
                weights,
                args,
                train_base.schema,
            )
            if run_train_evaluation
            else None
        )
        improved = validation_metrics["score"] < best_score - 1e-5
        if improved:
            best_score = validation_metrics["score"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        record = {
            "epoch": epoch + 1,
            "lr": lr,
            "seconds": time.time() - epoch_start,
            "train": train_metrics,
            "train_eval": train_evaluation_metrics,
            "val": validation_metrics,
            "best_score": best_score,
        }
        with history_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        log_wandb_epoch(
            wandb_run,
            epoch + 1,
            train_metrics,
            train_evaluation_metrics,
            validation_metrics,
            lr,
            teacher_force_ratio(epoch, args.teacher_force_epochs),
            best_score,
            improved,
            stale_epochs,
            record["seconds"],
        )
        if train_evaluation_metrics is not None:
            print(
                f"  train_eval_loss={train_evaluation_metrics['loss']:.4f} "
                f"train_eval_cat={train_evaluation_metrics['categorical_macro_accuracy']:.3f} "
                f"train_eval_root={train_evaluation_metrics['root_exact']:.3f}",
                flush=True,
            )
        print(
            f"  train_loss={train_metrics['loss']:.4f} "
            f"val_macro_mae={validation_metrics['numeric_macro_mae']:.4f} "
            f"val_macro_cat={validation_metrics['categorical_macro_accuracy']:.3f} "
            f"val_cat_act={validation_metrics['categorical_activity_macro_f1']:.3f} "
            f"val_num_act={validation_metrics['numeric_activity_macro_f1']:.3f} "
            f"val_root={validation_metrics['root_exact']:.3f} "
            f"pose_std={validation_metrics['pose_numeric_std']:.4f} "
            f"score={validation_metrics['score']:.4f} best={best_score:.4f}",
            flush=True,
        )
        payload = checkpoint_payload(
            model,
            ema,
            optimizer,
            scaler,
            train_base.schema,
            args,
            epoch,
            best_score,
            stale_epochs,
            validation_metrics,
        )
        atomic_save(payload, output_dir / "last.pt")
        if improved:
            atomic_save(payload, output_dir / "best.pt")
        if args.save_every and (epoch + 1) % args.save_every == 0:
            atomic_save(payload, output_dir / f"epoch_{epoch + 1:04d}.pt")
        if (
            not args.no_early_stopping
            and epoch + 1 >= args.min_epochs
            and stale_epochs >= args.patience
        ):
            print(f"early stopping after {stale_epochs} epochs without improvement", flush=True)
            break


    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    main()

