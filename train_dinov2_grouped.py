#!/usr/bin/env python
"""Train DINOv2 with upper, lower, and waistband component heads.

This is a standalone grouped-head experiment. It intentionally leaves
``train_dinov2.py`` unchanged as the shared-head baseline.

Architecture::

    DINOv2 feature
      -> shared trunk (LayerNorm -> Linear -> GELU -> Dropout)
      -> upper/lower/waistband component MLPs
      -> one linear regression projection and one categorical projection per group

Only the 152 union-schema ``[SEG]`` values are regression targets. Fixed numeric
literals in ``y_const`` are template values and are not predicted. Group outputs
are scattered back into the original schema order before loss computation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_dinov2 import cat_vocab_sizes, make_loaders, move_batch


GROUP_NAMES = ("upper", "lower", "waistband")
TOP_LEVEL_NAMES = {
    "upperbody_garment",
    "lowerbody_garment",
    "wholebody_garment",
}
UPPER_COMPONENTS = {"shirt", "bodice", "collar", "sleeve", "left"}
LOWER_COMPONENTS = {
    "skirt",
    "flare-skirt",
    "godet-skirt",
    "pencil-skirt",
    "pants",
    "levels-skirt",
}


def component_group(path: str) -> str:
    """Assign a schema path to one of the three semantic component groups."""
    parts = path.split(".")
    if parts and parts[0] == "design":
        parts = parts[1:]
    if parts and parts[0] in TOP_LEVEL_NAMES:
        parts = parts[1:]
    if not parts:
        raise ValueError(f"Cannot assign empty schema path {path!r}")

    component = parts[0]
    if component == "meta":
        if len(parts) < 2:
            raise ValueError(f"Meta path has no field name: {path!r}")
        group = {
            "upper": "upper",
            "bottom": "lower",
            "wb": "waistband",
        }.get(parts[1])
        if group is not None:
            return group
    elif component in UPPER_COMPONENTS:
        return "upper"
    elif component in LOWER_COMPONENTS:
        return "lower"
    elif component == "waistband":
        return "waistband"

    raise ValueError(
        f"Schema path {path!r} does not belong to upper, lower, or waistband. "
        "Assign new components explicitly rather than silently misrouting them."
    )


@dataclass(frozen=True)
class GroupLayout:
    name: str
    reg_indices: tuple[int, ...]
    reg_paths: tuple[str, ...]
    cat_field_indices: tuple[int, ...]
    cat_paths: tuple[str, ...]
    cat_logit_indices: tuple[int, ...]


def build_group_layout(schema: dict[str, Any]) -> tuple[GroupLayout, ...]:
    """Derive group membership without changing any global schema ordering."""
    reg_indices: dict[str, list[int]] = {name: [] for name in GROUP_NAMES}
    reg_paths: dict[str, list[str]] = {name: [] for name in GROUP_NAMES}
    cat_fields: dict[str, list[int]] = {name: [] for name in GROUP_NAMES}
    cat_paths: dict[str, list[str]] = {name: [] for name in GROUP_NAMES}
    cat_logits: dict[str, list[int]] = {name: [] for name in GROUP_NAMES}

    # Regression targets are the 152 [SEG] floats followed by the 32 normalised
    # constants, matching the baseline's ``y_reg = cat([y_cont, y_const])`` order.
    # Continuous slots occupy global indices ``[0, n_cont)``; constant slots
    # occupy ``[n_cont, n_cont + n_const)``.
    n_cont = int(schema["n_cont"])
    ordered_cont = sorted(schema["cont_slots"].items(), key=lambda item: item[1])
    for path, slot in ordered_cont:
        group = component_group(path)
        reg_indices[group].append(int(slot))
        reg_paths[group].append(path)

    ordered_const = sorted(schema["const_slots"].items(), key=lambda item: item[1])
    for path, slot in ordered_const:
        group = component_group(path)
        reg_indices[group].append(n_cont + int(slot))
        reg_paths[group].append(path)

    logit_offset = 0
    for field_idx, (path, vocab) in enumerate(schema["cat_vocab"].items()):
        group = component_group(path)
        cat_fields[group].append(field_idx)
        cat_paths[group].append(path)
        cat_logits[group].extend(range(logit_offset, logit_offset + len(vocab)))
        logit_offset += len(vocab)

    layouts = tuple(
        GroupLayout(
            name=name,
            reg_indices=tuple(reg_indices[name]),
            reg_paths=tuple(reg_paths[name]),
            cat_field_indices=tuple(cat_fields[name]),
            cat_paths=tuple(cat_paths[name]),
            cat_logit_indices=tuple(cat_logits[name]),
        )
        for name in GROUP_NAMES
    )

    all_reg_indices = sorted(index for group in layouts for index in group.reg_indices)
    expected_reg_indices = list(range(int(schema["n_cont"]) + int(schema["n_const"])))
    if all_reg_indices != expected_reg_indices:
        raise AssertionError("Each continuous and constant slot must belong to exactly one group")

    all_cat_fields = sorted(index for group in layouts for index in group.cat_field_indices)
    expected_cat_fields = list(range(int(schema["n_cat"])))
    if all_cat_fields != expected_cat_fields:
        raise AssertionError("Each categorical field must belong to exactly one group")

    expected_logits = sum(len(vocab) for vocab in schema["cat_vocab"].values())
    all_cat_logits = sorted(index for group in layouts for index in group.cat_logit_indices)
    if all_cat_logits != list(range(expected_logits)):
        raise AssertionError("Each packed categorical logit must belong to exactly one group")

    return layouts


class ComponentHead(nn.Module):
    """One nonlinear component representation with two output projections."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        reg_dim: int,
        cat_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reg_output = nn.Linear(hidden_dim, reg_dim)
        self.cat_output = nn.Linear(hidden_dim, cat_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)
        return self.reg_output(features), self.cat_output(features)


class GroupedGarmentDinoModel(nn.Module):
    """DINOv2 plus a shared trunk and three schema-derived component heads."""

    def __init__(
        self,
        schema: dict[str, Any],
        backbone_name: str,
        hidden_dim: int = 512,
        branch_hidden_dim: int = 128,
        waistband_hidden_dim: int = 64,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        self.freeze_backbone = freeze_backbone
        self.classification_frozen = False
        self.group_layout = build_group_layout(schema)
        self.reg_dim = int(schema["n_cont"]) + int(schema["n_const"])
        self.cat_vocab_sizes = cat_vocab_sizes(schema)
        self.cat_dim = sum(self.cat_vocab_sizes)

        embed_dim = self._infer_embed_dim()
        self.shared_trunk = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.component_heads = nn.ModuleDict()
        for layout in self.group_layout:
            group_hidden_dim = (
                waistband_hidden_dim
                if layout.name == "waistband"
                else branch_hidden_dim
            )
            self.component_heads[layout.name] = ComponentHead(
                in_dim=hidden_dim,
                hidden_dim=group_hidden_dim,
                reg_dim=len(layout.reg_indices),
                cat_dim=len(layout.cat_logit_indices),
                dropout=dropout,
            )
            self.register_buffer(
                f"_{layout.name}_reg_indices",
                torch.tensor(layout.reg_indices, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"_{layout.name}_cat_indices",
                torch.tensor(layout.cat_logit_indices, dtype=torch.long),
                persistent=False,
            )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def _infer_embed_dim(self) -> int:
        if hasattr(self.backbone, "embed_dim"):
            return int(self.backbone.embed_dim)
        with torch.inference_mode():
            was_training = self.backbone.training
            self.backbone.eval()
            output = self.backbone(torch.zeros(1, 3, 224, 224))
            self.backbone.train(was_training)
        return int(output.shape[-1])

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                return self.backbone(image)
        return self.backbone(image)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4:
            raise ValueError(f"Expected image batch [B,C,H,W], got {tuple(image.shape)}")

        shared = self.shared_trunk(self.encode(image))
        pred_reg = shared.new_zeros((shared.shape[0], self.reg_dim))
        pred_logits = shared.new_zeros((shared.shape[0], self.cat_dim))

        for layout in self.group_layout:
            group_reg, group_cat = self.component_heads[layout.name](shared)
            reg_indices = getattr(self, f"_{layout.name}_reg_indices")
            cat_indices = getattr(self, f"_{layout.name}_cat_indices")
            pred_reg = pred_reg.index_copy(1, reg_indices, group_reg)
            pred_logits = pred_logits.index_copy(1, cat_indices, group_cat)

        # Regression outputs stay linear during training. Clamp only in decode.
        return pred_reg, pred_logits

    def classification_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only task-specific categorical projections."""
        for head in self.component_heads.values():
            yield from head.cat_output.parameters()

    def freeze_classification(self) -> None:
        """Freeze cat projections; the loop separately disables categorical loss."""
        for parameter in self.classification_parameters():
            parameter.requires_grad = False
            parameter.grad = None
        self.classification_frozen = True


@dataclass
class EpochStats:
    loss: float = 0.0
    loss_reg: float = 0.0
    reg_mae: float = 0.0
    loss_cat: float = 0.0
    cat_acc: float = 0.0
    n: int = 0

    def update(self, values: dict[str, float], batch_size: int) -> None:
        self.loss += values["loss"] * batch_size
        self.loss_reg += values["loss_reg"] * batch_size
        self.reg_mae += values["reg_mae"] * batch_size
        self.loss_cat += values["loss_cat"] * batch_size
        self.cat_acc += values["cat_acc"] * batch_size
        self.n += batch_size

    def averages(self) -> dict[str, float]:
        denominator = max(self.n, 1)
        return {
            "loss": self.loss / denominator,
            "loss_reg": self.loss_reg / denominator,
            "reg_mae": self.reg_mae / denominator,
            "loss_cat": self.loss_cat / denominator,
            "cat_acc": self.cat_acc / denominator,
        }


@dataclass
class CatFreezeController:
    """Early-stop categorical optimization using validation categorical loss."""

    patience: int
    min_epochs: int
    min_delta: float
    best_loss: float = math.inf
    best_epoch: int = 0
    bad_epochs: int = 0
    frozen_epoch: int | None = None

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def update(
        self,
        epoch: int,
        val_loss: float,
        model: GroupedGarmentDinoModel,
    ) -> tuple[bool, bool]:
        """Return ``(improved, froze_now)`` after one validation epoch."""
        if not math.isfinite(val_loss):
            return False, False

        improved = val_loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.bad_epochs = 0
        elif epoch >= self.min_epochs and not model.classification_frozen:
            self.bad_epochs += 1

        should_freeze = (
            self.enabled
            and not model.classification_frozen
            and epoch >= self.min_epochs
            and self.bad_epochs >= self.patience
        )
        if should_freeze:
            model.freeze_classification()
            self.frozen_epoch = epoch
        return improved, should_freeze

    def state_dict(self) -> dict[str, Any]:
        return {
            "patience": self.patience,
            "min_epochs": self.min_epochs,
            "min_delta": self.min_delta,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
            "frozen_epoch": self.frozen_epoch,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", default="prepared_v2")
    parser.add_argument("--out-dir", default="runs/dinov2_grouped")
    parser.add_argument("--backbone", default="dinov2_vits14")
    parser.add_argument("--mode", choices=("single", "all_images"), default="single")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--branch-hidden-dim", type=int, default=128)
    parser.add_argument("--waistband-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--lambda-cat",
        type=float,
        default=0.5,
        help="weight on categorical loss; lowered from 1.0 so the categorical "
        "objective stops dragging the shared trunk toward overconfidence",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="categorical label smoothing (train only); curbs overconfident logits",
    )
    parser.add_argument("--reg-loss", choices=("huber", "mse"), default="huber")
    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--cat-freeze-patience",
        type=int,
        default=3,
        help="freeze cat projections after this many non-improving val epochs; 0 disables",
    )
    parser.add_argument(
        "--cat-freeze-min-epochs",
        type=int,
        default=5,
        help="do not freeze categorical optimization before this epoch",
    )
    parser.add_argument(
        "--cat-freeze-min-delta",
        type=float,
        default=1e-4,
        help="minimum categorical val-loss improvement that resets patience",
    )
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--amp", action="store_true", help="use CUDA mixed precision")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument(
        "--image-path-prefix",
        nargs=2,
        action="append",
        default=[],
        metavar=("OLD", "NEW"),
        help="rewrite image path prefixes; repeatable",
    )
    parser.add_argument(
        "--keep-missing-images",
        action="store_true",
        help="do not filter missing image paths after prefix rewrites",
    )
    parser.add_argument(
        "--print-group-paths",
        action="store_true",
        help="print every regression and categorical path assigned to each head",
    )
    return parser.parse_args()


def compute_losses(
    pred_reg: torch.Tensor,
    pred_logits: torch.Tensor,
    batch: dict[str, Any],
    vocab_sizes: list[int],
    lambda_cat: float,
    reg_loss_name: str,
    huber_delta: float,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_reg = torch.cat([batch["y_cont"], batch["y_const"]], dim=-1)
    mask_reg = torch.cat([batch["mask"], batch["const_mask"]], dim=-1)

    if reg_loss_name == "huber":
        elementwise_reg = F.smooth_l1_loss(
            pred_reg,
            target_reg,
            reduction="none",
            beta=huber_delta,
        )
    else:
        elementwise_reg = (pred_reg - target_reg).pow(2)

    active_reg = mask_reg.sum().clamp(min=1)
    loss_reg = (elementwise_reg * mask_reg).sum() / active_reg
    reg_mae = ((pred_reg - target_reg).abs() * mask_reg).sum() / active_reg

    loss_cat = pred_logits.new_zeros(())
    active_fields = 0
    correct = 0
    total = 0
    offset = 0
    for field_idx, vocab_size in enumerate(vocab_sizes):
        logits = pred_logits[:, offset:offset + vocab_size]
        target = batch["y_cat"][:, field_idx]
        valid = target.ne(-1)
        if valid.any():
            loss_cat = loss_cat + F.cross_entropy(
                logits, target, ignore_index=-1, label_smoothing=label_smoothing
            )
            active_fields += 1
            prediction = logits.argmax(dim=-1)
            correct += prediction[valid].eq(target[valid]).sum().item()
            total += valid.sum().item()
        offset += vocab_size

    if active_fields:
        loss_cat = loss_cat / active_fields

    # Once categorical training is frozen, keep reporting its validation metric
    # without retaining a zero-weight categorical graph for backpropagation.
    loss = loss_reg if lambda_cat == 0.0 else loss_reg + lambda_cat * loss_cat
    values = {
        "loss": float(loss.detach().cpu()),
        "loss_reg": float(loss_reg.detach().cpu()),
        "reg_mae": float(reg_mae.detach().cpu()),
        "loss_cat": float(loss_cat.detach().cpu()),
        "cat_acc": correct / total if total else 0.0,
    }
    return loss, values


def run_epoch(
    model: GroupedGarmentDinoModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    vocab_sizes: list[int],
    lambda_cat: float,
    reg_loss_name: str,
    huber_delta: float,
    label_smoothing: float,
    amp: bool,
    grad_clip: float,
    log_every: int,
    max_batches: int | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    if model.freeze_backbone:
        model.backbone.eval()

    stats = EpochStats()
    started = time.time()
    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for step, batch in enumerate(loader, start=1):
            if max_batches is not None and step > max_batches:
                break
            batch = move_batch(batch, device)

            with torch.autocast(
                device_type="cuda",
                enabled=amp and device.type == "cuda",
            ):
                pred_reg, pred_logits = model(batch["image"])
                loss, values = compute_losses(
                    pred_reg=pred_reg,
                    pred_logits=pred_logits,
                    batch=batch,
                    vocab_sizes=vocab_sizes,
                    lambda_cat=lambda_cat,
                    reg_loss_name=reg_loss_name,
                    huber_delta=huber_delta,
                    # Smoothing is a training regulariser only; validation keeps
                    # true cross-entropy so the reported val metric and the
                    # categorical freeze controller stay comparable across runs.
                    label_smoothing=label_smoothing if is_train else 0.0,
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        (
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ),
                        max_norm=grad_clip,
                    )
                scaler.step(optimizer)
                scaler.update()

            batch_size = int(batch["image"].shape[0])
            stats.update(values, batch_size)
            if is_train and log_every > 0 and step % log_every == 0:
                averages = stats.averages()
                print(
                    f"  step {step:04d} loss={averages['loss']:.4f} "
                    f"reg={averages['loss_reg']:.4f} mae={averages['reg_mae']:.4f} "
                    f"cat={averages['loss_cat']:.4f} acc={averages['cat_acc']:.3f} "
                    f"({time.time() - started:.1f}s)",
                    flush=True,
                )

    return stats.averages()


def make_optimizer(
    model: GroupedGarmentDinoModel,
    args: argparse.Namespace,
) -> torch.optim.AdamW:
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    ]
    parameter_groups: list[dict[str, Any]] = [
        {"params": head_parameters, "lr": args.lr, "name": "heads"}
    ]

    backbone_parameters = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": args.lr * args.backbone_lr_scale,
                "name": "backbone",
            }
        )

    return torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)


def save_checkpoint(
    path: Path,
    model: GroupedGarmentDinoModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cat_freeze: CatFreezeController,
    epoch: int,
    best_val_mae: float,
    args: argparse.Namespace,
    schema: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "dinov2_grouped_v1",
            "epoch": epoch,
            "best_val_mae": best_val_mae,
            "args": vars(args),
            "schema": schema,
            "group_layout": [
                {
                    "name": layout.name,
                    "reg_indices": list(layout.reg_indices),
                    "cat_field_indices": list(layout.cat_field_indices),
                    "cat_logit_indices": list(layout.cat_logit_indices),
                }
                for layout in model.group_layout
            ],
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "classification_frozen": model.classification_frozen,
            "cat_freeze": cat_freeze.state_dict(),
        },
        path,
    )


def print_layout(layouts: tuple[GroupLayout, ...], print_paths: bool) -> None:
    print("component heads:", flush=True)
    for layout in layouts:
        print(
            f"  {layout.name:<9} reg_slots={len(layout.reg_indices):3d} "
            f"cat_fields={len(layout.cat_field_indices):2d} "
            f"cat_logits={len(layout.cat_logit_indices):3d}",
            flush=True,
        )
        if print_paths:
            print("    regression:", flush=True)
            for index, path in zip(layout.reg_indices, layout.reg_paths):
                print(f"      [{index:3d}] {path}", flush=True)
            print("    categorical:", flush=True)
            for index, path in zip(layout.cat_field_indices, layout.cat_paths):
                print(f"      [{index:2d}] {path}", flush=True)


def validate_args(args: argparse.Namespace) -> None:
    if args.cat_freeze_patience < 0:
        raise SystemExit("--cat-freeze-patience must be >= 0")
    if args.cat_freeze_min_epochs < 1:
        raise SystemExit("--cat-freeze-min-epochs must be >= 1")
    if args.backbone_lr_scale <= 0:
        raise SystemExit("--backbone-lr-scale must be > 0")
    if args.huber_delta <= 0:
        raise SystemExit("--huber-delta must be > 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested, but CUDA is unavailable")


def main() -> None:
    args = parse_args()
    validate_args(args)

    root = Path(__file__).resolve().parent
    os.environ.setdefault("TORCH_HOME", str(root / ".cache" / "torch"))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_loader, val_loader, schema = make_loaders(args)
    vocab_sizes = cat_vocab_sizes(schema)
    model = GroupedGarmentDinoModel(
        schema=schema,
        backbone_name=args.backbone,
        hidden_dim=args.hidden_dim,
        branch_hidden_dim=args.branch_hidden_dim,
        waistband_hidden_dim=args.waistband_hidden_dim,
        dropout=args.dropout,
        freeze_backbone=not args.unfreeze_backbone,
    ).to(device)

    optimizer = make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )
    scaler = torch.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    cat_freeze = CatFreezeController(
        patience=args.cat_freeze_patience,
        min_epochs=args.cat_freeze_min_epochs,
        min_delta=args.cat_freeze_min_delta,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} mode={args.mode} backbone={args.backbone} architecture=grouped",
        flush=True,
    )
    print(
        f"train samples={len(train_loader.dataset)} "
        f"garments={len(train_loader.dataset.garment_ids)}; "
        f"val samples={len(val_loader.dataset)} "
        f"garments={len(val_loader.dataset.garment_ids)}",
        flush=True,
    )
    print(
        f"reg_dim={model.reg_dim} cat_fields={len(vocab_sizes)} "
        f"cat_logits={model.cat_dim}",
        flush=True,
    )
    print_layout(model.group_layout, args.print_group_paths)
    print(
        "optimizer LRs: "
        + ", ".join(
            f"{group.get('name', index)}={group['lr']:.2e}"
            for index, group in enumerate(optimizer.param_groups)
        ),
        flush=True,
    )
    if cat_freeze.enabled:
        print(
            "categorical early freeze: "
            f"patience={cat_freeze.patience} min_epoch={cat_freeze.min_epochs} "
            f"min_delta={cat_freeze.min_delta:g}",
            flush=True,
        )
    else:
        print("categorical early freeze: disabled", flush=True)
    print(f"parameters trainable={trainable:,} total={total:,}", flush=True)

    best_val_mae = math.inf
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        print(f"\nepoch {epoch}/{args.epochs}", flush=True)
        effective_lambda_cat = 0.0 if model.classification_frozen else args.lambda_cat

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            vocab_sizes=vocab_sizes,
            lambda_cat=effective_lambda_cat,
            reg_loss_name=args.reg_loss,
            huber_delta=args.huber_delta,
            label_smoothing=args.label_smoothing,
            amp=args.amp,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            scaler=scaler,
            device=device,
            vocab_sizes=vocab_sizes,
            lambda_cat=effective_lambda_cat,
            reg_loss_name=args.reg_loss,
            huber_delta=args.huber_delta,
            label_smoothing=args.label_smoothing,
            amp=args.amp,
            grad_clip=args.grad_clip,
            log_every=0,
            max_batches=args.max_val_batches,
        )

        print(
            "  train "
            f"loss={train_metrics['loss']:.4f} reg={train_metrics['loss_reg']:.4f} "
            f"mae={train_metrics['reg_mae']:.4f} "
            f"cat={train_metrics['loss_cat']:.4f} acc={train_metrics['cat_acc']:.3f}",
            flush=True,
        )
        print(
            "  val   "
            f"loss={val_metrics['loss']:.4f} reg={val_metrics['loss_reg']:.4f} "
            f"mae={val_metrics['reg_mae']:.4f} "
            f"cat={val_metrics['loss_cat']:.4f} acc={val_metrics['cat_acc']:.3f}",
            flush=True,
        )

        cat_improved, cat_froze_now = cat_freeze.update(
            epoch,
            val_metrics["loss_cat"],
            model,
        )
        if cat_froze_now:
            print(
                "  categorical projections frozen and categorical loss disabled "
                f"from epoch {epoch + 1}; best categorical val loss "
                f"was {cat_freeze.best_loss:.4f} at epoch {cat_freeze.best_epoch}",
                flush=True,
            )

        scheduler.step()
        if val_metrics["reg_mae"] < best_val_mae:
            best_val_mae = val_metrics["reg_mae"]
            save_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                cat_freeze,
                epoch,
                best_val_mae,
                args,
                schema,
            )
        if cat_improved:
            save_checkpoint(
                out_dir / "best_cat.pt",
                model,
                optimizer,
                scheduler,
                cat_freeze,
                epoch,
                best_val_mae,
                args,
                schema,
            )
        save_checkpoint(
            out_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            cat_freeze,
            epoch,
            best_val_mae,
            args,
            schema,
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lambda_cat_effective": effective_lambda_cat,
                "classification_frozen": model.classification_frozen,
                "cat_freeze": cat_freeze.state_dict(),
                "lrs_next_epoch": [group["lr"] for group in optimizer.param_groups],
            }
        )
        with open(out_dir / "history.json", "w") as handle:
            json.dump({"args": vars(args), "history": history}, handle, indent=2)

    with open(out_dir / "config.json", "w") as handle:
        json.dump(
            {
                "args": vars(args),
                "architecture": "grouped",
                "regression_targets": "y_cont+y_const",
                "schema_summary": {
                    "n_cont": schema["n_cont"],
                    "n_cat": schema["n_cat"],
                    "cat_logits": sum(
                        len(vocab) for vocab in schema["cat_vocab"].values()
                    ),
                },
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
