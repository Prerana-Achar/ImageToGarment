#!/usr/bin/env python
"""Train a DINOv2 image-to-GarmentCode baseline.

The data contract follows README.md:

* ``GarmentDataset`` returns either one sampled view per garment or every image.
* ``y_cont`` and normalized ``y_const`` are concatenated into one regression
  target.
* ``y_cat`` contains one class index per categorical field, with ``-1`` for
  inapplicable fields.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import UNSUPPORTED_GARMENTCODE_PARAMS
from prepare_data import GarmentDataset


def make_grad_scaler(enabled: bool) -> Any:
    scaler_cls = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if scaler_cls is None:
        scaler_cls = torch.cuda.amp.GradScaler
    return scaler_cls(enabled=enabled)

@dataclass
class BatchStats:
    loss: float = 0.0
    loss_reg: float = 0.0
    loss_cat: float = 0.0
    cat_acc: float = 0.0
    cat_correct: int = 0
    cat_total: int = 0
    n: int = 0

    def update(self, values: dict[str, float], batch_size: int) -> None:
        self.loss += values["loss"] * batch_size
        self.loss_reg += values["loss_reg"] * batch_size
        self.loss_cat += values["loss_cat"] * batch_size
        self.cat_acc += values["cat_acc"] * batch_size
        self.cat_correct += int(values.get("_cat_correct", 0))
        self.cat_total += int(values.get("_cat_total", 0))
        self.n += batch_size

    def averages(self) -> dict[str, float]:
        denom = max(self.n, 1)
        cat_acc = (
            self.cat_correct / self.cat_total
            if self.cat_total
            else self.cat_acc / denom
        )
        return {
            "loss": self.loss / denom,
            "loss_reg": self.loss_reg / denom,
            "loss_cat": self.loss_cat / denom,
            "cat_acc": cat_acc,
        }


class GarmentDinoModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        reg_dim: int,
        cat_vocab_sizes: list[int],
        hidden_dim: int,
        dropout: float,
        freeze_backbone: bool,
        bounded_regression: bool = True,
        dinov2_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        repo = Path(dinov2_dir).expanduser().resolve() if dinov2_dir else None
        if repo is not None and (repo / "hubconf.py").is_file():
            self.backbone = torch.hub.load(
                str(repo), backbone_name, source="local", pretrained=True
            )
        else:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", backbone_name, pretrained=True
            )
        self.freeze_backbone = freeze_backbone
        self.bounded_regression = bounded_regression
        self.cat_vocab_sizes = cat_vocab_sizes

        embed_dim = self._infer_embed_dim()
        self.reg_head = self._make_head(embed_dim, hidden_dim, reg_dim, dropout)
        self.cat_head = self._make_head(embed_dim, hidden_dim, sum(cat_vocab_sizes), dropout)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _infer_embed_dim(self) -> int:
        if hasattr(self.backbone, "embed_dim"):
            return int(self.backbone.embed_dim)
        with torch.inference_mode():
            was_training = self.backbone.training
            self.backbone.eval()
            y = self.backbone(torch.zeros(1, 3, 224, 224))
            self.backbone.train(was_training)
        return int(y.shape[-1])

    @staticmethod
    def _make_head(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4:
            raise ValueError(f"Expected image batch [B,C,H,W], got {tuple(image.shape)}")

        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                feat = self.backbone(image)
        else:
            feat = self.backbone(image)

        regression = self.reg_head(feat)
        if self.bounded_regression:
            regression = torch.sigmoid(regression)
        return regression, self.cat_head(feat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", default="prepared_v2")
    parser.add_argument("--out-dir", default="runs/dinov2_vits14")
    parser.add_argument("--backbone", default="dinov2_vitg14_reg")
    parser.add_argument("--dinov2-dir")
    parser.add_argument("--mode", choices=("single", "all_images"), default="single")
    parser.add_argument("--augmentation", choices=("none", "light"), default="light")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=100, help="save numbered checkpoints every N epochs; 0 disables")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda-cat", type=float, default=1.0)
    parser.add_argument(
        "--unbounded-regression",
        action="store_false",
        dest="bounded_regression",
        help="legacy mode: do not apply sigmoid to baseline regression outputs",
    )
    parser.set_defaults(bounded_regression=True)
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
        help="rewrite image path prefixes from a moved prepared manifest; repeatable",
    )
    parser.add_argument(
        "--keep-missing-images",
        action="store_true",
        help="do not filter missing image paths after prefix rewrites",
    )
    parser.add_argument("--allow-unbalanced-data", action="store_true",
                        help="override a failed balance_report.json readiness gate")
    return parser.parse_args()


def cat_vocab_sizes(schema: dict[str, Any]) -> list[int]:
    return [len(vocab) for vocab in schema["cat_vocab"].values()]


def strip_target_prefix(path: str) -> str:
    parts = path.split(".")
    if parts and parts[0] in {
        "wholebody_garment",
        "upperbody_garment",
        "lowerbody_garment",
    }:
        return ".".join(parts[1:])
    return path


def disable_unsupported_targets(ds: GarmentDataset) -> list[str]:
    """Remove legacy targets from baseline loss masks without changing tensor shape."""
    ignored: list[str] = []
    for path, index in ds.schema["cont_slots"].items():
        if strip_target_prefix(path) in UNSUPPORTED_GARMENTCODE_PARAMS:
            ds.mask[:, int(index)] = 0
            ignored.append(path)
    for path, index in ds.schema["const_slots"].items():
        if strip_target_prefix(path) in UNSUPPORTED_GARMENTCODE_PARAMS:
            ds.const_mask[:, int(index)] = 0
            ignored.append(path)
    for index, path in enumerate(ds.schema["cat_vocab"]):
        if strip_target_prefix(path) in UNSUPPORTED_GARMENTCODE_PARAMS:
            ds.y_cat[:, index] = -1
            ignored.append(path)
    return ignored


def rewrite_and_filter_images(ds: GarmentDataset, rewrites: list[list[str]], keep_missing: bool) -> None:
    kept_ids = []
    for gid in ds.garment_ids:
        frames = ds.images[gid]["frames"]
        kept_frames = {}
        for frame_key, views in frames.items():
            new_views = []
            for path in views:
                if path is None:
                    new_views.append(None)
                    continue
                new_path = path
                for old, new in rewrites:
                    if new_path.startswith(old):
                        new_path = new + new_path[len(old):]
                if keep_missing or os.path.isfile(new_path):
                    new_views.append(new_path)
                else:
                    new_views.append(None)
            if any(v is not None for v in new_views):
                kept_frames[frame_key] = new_views
        ds.images[gid]["frames"] = kept_frames
        if kept_frames:
            kept_ids.append(gid)
    dropped = len(ds.garment_ids) - len(kept_ids)
    ds.garment_ids = kept_ids
    ds.refresh_samples()
    if dropped:
        print(f"[data] {ds.split}: dropped {dropped} garments with no readable images", flush=True)


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    report_path = Path(args.prepared_dir) / "balance_report.json"
    if report_path.is_file():
        with open(report_path) as handle:
            balance_report = json.load(handle)
        if not balance_report.get("ready", False) and not getattr(args, "allow_unbalanced_data", False):
            raise RuntimeError(
                f"Prepared dataset is not training-ready: {report_path}. "
                "Fix the reported deficits or explicitly pass --allow-unbalanced-data."
            )
    with open(Path(args.prepared_dir) / "schema.json") as f:
        schema = json.load(f)

    train_ds = GarmentDataset(
        args.prepared_dir,
        split="train",
        mode=args.mode,
        train=True,
        image_size=args.image_size,
        augmentation=args.augmentation,
    )
    val_ds = GarmentDataset(
        args.prepared_dir,
        split="val",
        mode=args.mode,
        train=False,
        image_size=args.image_size,
        augmentation="none",
    )
    ignored_targets = disable_unsupported_targets(train_ds)
    disable_unsupported_targets(val_ds)
    if ignored_targets:
        print(
            f"[data] disabled unsupported targets: {', '.join(sorted(ignored_targets))}",
            flush=True,
        )

    if args.image_path_prefix or not args.keep_missing_images:
        rewrite_and_filter_images(train_ds, args.image_path_prefix, args.keep_missing_images)
        rewrite_and_filter_images(val_ds, args.image_path_prefix, args.keep_missing_images)

    if len(train_ds) == 0:
        raise RuntimeError("No train samples have readable images after path filtering.")
    if len(val_ds) == 0:
        raise RuntimeError("No val samples have readable images after path filtering.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, schema


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def compute_losses(
    pred_reg: torch.Tensor,
    pred_logits: torch.Tensor,
    batch: dict[str, Any],
    vocab_sizes: list[int],
    lambda_cat: float,
    label_smoothing: float = 0.0,
    class_weights: list[list[float] | None] | None = None,
    reg_loss: str = "mse",
    smooth_l1_beta: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    y_reg = torch.cat([batch["y_cont"], batch["y_const"]], dim=-1)
    m_reg = torch.cat([batch["mask"], batch["const_mask"]], dim=-1)
    if reg_loss == "smooth_l1":
        per_item_reg = F.smooth_l1_loss(
            pred_reg, y_reg, reduction="none", beta=smooth_l1_beta
        )
    else:
        per_item_reg = (pred_reg - y_reg).pow(2)
    loss_reg = (per_item_reg * m_reg).sum() / m_reg.sum().clamp(min=1)

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
            weights = None
            if class_weights is not None and class_weights[field_idx] is not None:
                weights = torch.as_tensor(
                    class_weights[field_idx], device=logits.device, dtype=logits.dtype
                )
            loss_cat = loss_cat + F.cross_entropy(
                logits,
                target,
                ignore_index=-1,
                label_smoothing=label_smoothing,
                weight=weights,
            )
            active_fields += 1
            pred = logits.argmax(dim=-1)
            correct += pred[valid].eq(target[valid]).sum().item()
            total += valid.sum().item()
        offset += vocab_size

    if active_fields:
        loss_cat = loss_cat / active_fields

    loss = loss_reg + lambda_cat * loss_cat
    stats = {
        "loss": float(loss.detach().cpu()),
        "loss_reg": float(loss_reg.detach().cpu()),
        "loss_cat": float(loss_cat.detach().cpu()),
        "cat_acc": correct / total if total else 0.0,
        "_cat_correct": correct,
        "_cat_total": total,
    }
    return loss, stats

def run_epoch(
    model: GarmentDinoModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any,
    device: torch.device,
    vocab_sizes: list[int],
    lambda_cat: float,
    amp: bool,
    log_every: int,
    max_batches: int | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    if model.freeze_backbone:
        model.backbone.eval()

    stats = BatchStats()
    started = time.time()
    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for step, batch in enumerate(loader, start=1):
            if max_batches is not None and step > max_batches:
                break
            batch = move_batch(batch, device)

            with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                pred_reg, pred_logits = model(batch["image"])
                loss, values = compute_losses(
                    pred_reg,
                    pred_logits,
                    batch,
                    vocab_sizes,
                    lambda_cat,
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_size = int(batch["image"].shape[0])
            stats.update(values, batch_size)
            if is_train and log_every > 0 and step % log_every == 0:
                avg = stats.averages()
                elapsed = time.time() - started
                print(
                    f"  step {step:04d} "
                    f"loss={avg['loss']:.4f} reg={avg['loss_reg']:.4f} "
                    f"cat={avg['loss_cat']:.4f} acc={avg['cat_acc']:.3f} "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )

    return stats.averages()


def save_checkpoint(
    path: Path,
    model: GarmentDinoModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    args: argparse.Namespace,
    schema: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val": best_val,
            "args": vars(args),
            "schema": schema,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    os.environ.setdefault("TORCH_HOME", str(root / ".cache" / "torch"))

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_loader, val_loader, schema = make_loaders(args)
    vocab_sizes = cat_vocab_sizes(schema)
    reg_dim = int(schema["n_cont"]) + int(schema["n_const"])

    model = GarmentDinoModel(
        backbone_name=args.backbone,
        reg_dim=reg_dim,
        cat_vocab_sizes=vocab_sizes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_backbone=not args.unfreeze_backbone,
        bounded_regression=args.bounded_regression,
        dinov2_dir=args.dinov2_dir,
    ).to(device)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"device={device} mode={args.mode} backbone={args.backbone}", flush=True)
    print(
        f"train samples={len(train_loader.dataset)} "
        f"garments={len(train_loader.dataset.garment_ids)}; "
        f"val samples={len(val_loader.dataset)} "
        f"garments={len(val_loader.dataset.garment_ids)}",
        flush=True,
    )
    print(f"reg_dim={reg_dim} cat_fields={len(vocab_sizes)} cat_logits={sum(vocab_sizes)}", flush=True)
    print(f"parameters trainable={trainable:,} total={total:,}", flush=True)

    best_val = math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        print(f"\nepoch {epoch}/{args.epochs}", flush=True)
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            vocab_sizes,
            args.lambda_cat,
            args.amp,
            args.log_every,
            args.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            scaler,
            device,
            vocab_sizes,
            args.lambda_cat,
            args.amp,
            0,
            args.max_val_batches,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            "  train "
            f"loss={train_metrics['loss']:.4f} reg={train_metrics['loss_reg']:.4f} "
            f"cat={train_metrics['loss_cat']:.4f} acc={train_metrics['cat_acc']:.3f}",
            flush=True,
        )
        print(
            "  val   "
            f"loss={val_metrics['loss']:.4f} reg={val_metrics['loss_reg']:.4f} "
            f"cat={val_metrics['loss_cat']:.4f} acc={val_metrics['cat_acc']:.3f}",
            flush=True,
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, best_val, args, schema)
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, best_val, args, schema)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(out_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, best_val, args, schema)

        with open(out_dir / "history.json", "w") as f:
            json.dump({"args": vars(args), "history": history}, f, indent=2)

    with open(out_dir / "config.json", "w") as f:
        json.dump({"args": vars(args), "schema_summary": asdict_summary(schema)}, f, indent=2)


def asdict_summary(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_cont": schema["n_cont"],
        "n_const": schema["n_const"],
        "n_cat": schema["n_cat"],
        "cat_logits": sum(len(vocab) for vocab in schema["cat_vocab"].values()),
    }


if __name__ == "__main__":
    main()
