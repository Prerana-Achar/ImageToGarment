#!/usr/bin/env python
"""Train or run inference with either DINOv2 garment MLP implementation.

Architectures:

* ``baseline`` uses the two flat heads from ``train_dinov2.py``.
* ``modelpy`` uses the per-parameter heads from ``model.py``.

Inference detects the architecture stored in new checkpoints. Checkpoints made by
the original ``train_dinov2.py`` are treated as ``baseline`` checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from infer_dinov2 import (
    build_model as build_baseline_inference_model,
    decode_prediction as decode_baseline_prediction,
    load_image,
    path_from_gid,
    write_yaml as write_baseline_yaml,
)
from model import UNSUPPORTED_GARMENTCODE_PARAMS
from train_dinov2 import (
    BatchStats,
    GarmentDinoModel,
    asdict_summary,
    cat_vocab_sizes,
    compute_losses as compute_baseline_losses,
    make_loaders,
    move_batch,
)


ROOT = Path(__file__).resolve().parent
TOP_PREFIXES = {"wholebody_garment", "upperbody_garment", "lowerbody_garment"}
DEFAULT_MODEL_SCHEMA = ROOT / "GarmentCodeRC" / "assets" / "design_params" / "default_new.yaml"
_RELU_WARNING_EMITTED = False


def strip_top_prefix(path: str) -> str:
    parts = path.split(".")
    if parts and parts[0] in TOP_PREFIXES:
        parts = parts[1:]
    return ".".join(parts)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested, but CUDA is not available")
    return torch.device(name)


def configure_torch_home() -> None:
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

def make_grad_scaler(enabled: bool) -> Any:
    scaler_cls = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if scaler_cls is None:
        scaler_cls = torch.cuda.amp.GradScaler
    return scaler_cls(enabled=enabled)


def checkpoint_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: value for key, value in vars(args).items() if key != "func"}



def init_wandb(
    args: argparse.Namespace,
    out_dir: Path,
    schema: dict[str, Any],
    train_loader: Any,
    val_loader: Any,
    trainable_params: int,
    total_params: int,
    adapter: Any | None,
) -> Any | None:
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "--wandb was requested, but wandb is not installed. Install it with: pip install wandb"
        ) from exc

    run_name = args.wandb_run_name or out_dir.name
    config = checkpoint_args(args)
    config.update(
        {
            "train_samples": len(train_loader.dataset),
            "train_garments": len(train_loader.dataset.garment_ids),
            "val_samples": len(val_loader.dataset),
            "val_garments": len(val_loader.dataset.garment_ids),
            "trainable_params": trainable_params,
            "total_params": total_params,
            "n_cont": schema.get("n_cont"),
            "n_const": schema.get("n_const"),
            "n_cat": schema.get("n_cat"),
            "cat_logits": sum(len(vocab) for vocab in schema.get("cat_vocab", {}).values()),
        }
    )
    if adapter is not None:
        config["target_adapter"] = adapter.summary()

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        dir=str(out_dir),
        mode=args.wandb_mode,
        tags=args.wandb_tags,
        config=config,
    )
    wandb.define_metric("epoch")
    for prefix in ("train", "val", "gap", "best"):
        wandb.define_metric(f"{prefix}/*", step_metric="epoch")
    wandb.define_metric("optim/*", step_metric="epoch")
    return run


def log_wandb_epoch(
    wandb_run: Any | None,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_val: float,
    optimizer: torch.optim.Optimizer,
) -> None:
    if wandb_run is None:
        return
    payload: dict[str, float | int] = {"epoch": epoch}
    for split, metrics in (("train", train_metrics), ("val", val_metrics)):
        payload.update(
            {
                f"{split}/loss": metrics["loss"],
                f"{split}/loss_reg": metrics["loss_reg"],
                f"{split}/loss_cat": metrics["loss_cat"],
                f"{split}/cat_acc": metrics["cat_acc"],
            }
        )
    payload.update(
        {
            "gap/loss": val_metrics["loss"] - train_metrics["loss"],
            "gap/loss_reg": val_metrics["loss_reg"] - train_metrics["loss_reg"],
            "gap/loss_cat": val_metrics["loss_cat"] - train_metrics["loss_cat"],
            "best/val_loss": best_val,
            "optim/lr": float(optimizer.param_groups[0]["lr"]),
        }
    )
    wandb_run.log(payload, step=epoch)


def ensure_modelpy_relu_compatibility() -> None:
    """Work around model.py's nn.RELU typo without modifying model.py."""
    global _RELU_WARNING_EMITTED
    if not hasattr(nn, "RELU"):
        setattr(nn, "RELU", nn.ReLU)
        if not _RELU_WARNING_EMITTED:
            warnings.warn(
                "model.py uses nn.RELU(), which does not exist; using nn.ReLU() "
                "through a runner-local compatibility alias.",
                stacklevel=2,
            )
            _RELU_WARNING_EMITTED = True


def resolve_dinov2_dir(value: str | None) -> Path:
    candidates = []
    if value:
        candidates.append(Path(value).expanduser())
    candidates.extend(
        [
            ROOT / "DINOv2",
            Path(os.environ["TORCH_HOME"]) / "hub" / "facebookresearch_dinov2_main",
        ]
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_dir() and (candidate / "hubconf.py").is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find a local DINOv2 repository. Searched: {searched}")


class PreparedTargetAdapter:
    """Map the prepared prefixed slots to model.py's unprefixed parameter heads."""

    def __init__(self, schema: dict[str, Any], param_specs: list[Any]) -> None:
        self.schema = schema
        self.spec_by_name = {spec.name: spec for spec in param_specs}
        self.reg_sources: dict[str, list[dict[str, Any]]] = {
            spec.name: [] for spec in param_specs if spec.is_regression
        }
        self.cat_sources: dict[str, list[dict[str, Any]]] = {
            spec.name: [] for spec in param_specs if spec.is_classification
        }
        problems: list[str] = []
        self.range_expansions: list[dict[str, Any]] = []
        self.ignored_prepared_targets: list[str] = []

        for path, index in schema["cont_slots"].items():
            name = strip_top_prefix(path)
            if name in UNSUPPORTED_GARMENTCODE_PARAMS:
                self.ignored_prepared_targets.append(path)
                continue
            spec = self.spec_by_name.get(name)
            if spec is None or not spec.is_regression:
                problems.append(f"continuous target {path!r} has no regression head")
                continue
            self.reg_sources[name].append(
                {"kind": "cont", "path": path, "index": int(index)}
            )

        for path, index in schema["const_slots"].items():
            name = strip_top_prefix(path)
            if name in UNSUPPORTED_GARMENTCODE_PARAMS:
                self.ignored_prepared_targets.append(path)
                continue
            spec = self.spec_by_name.get(name)
            if spec is None:
                problems.append(f"constant target {path!r} has no model.py head")
                continue
            data_lo, data_hi = schema["const_ranges"][path]
            if spec.is_classification:
                if not spec.choices or not all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in spec.choices
                ):
                    problems.append(
                        f"numeric constant target {path!r} cannot map to choices {spec.choices!r}"
                    )
                    continue
                self.cat_sources[name].append(
                    {
                        "kind": "const_cat",
                        "path": path,
                        "index": int(index),
                        "data_lo": float(data_lo),
                        "data_hi": float(data_hi),
                        "choices": [float(value) for value in spec.choices],
                    }
                )
                continue
            if not spec.is_regression:
                problems.append(f"constant target {path!r} has no compatible head")
                continue
            model_lo = float(spec.min_value)
            model_hi = float(spec.max_value)
            if float(data_lo) < model_lo or float(data_hi) > model_hi:
                old_lo, old_hi = model_lo, model_hi
                model_lo = min(float(data_lo), model_lo)
                model_hi = max(float(data_hi), model_hi)
                object.__setattr__(spec, "min_value", model_lo)
                object.__setattr__(spec, "max_value", model_hi)
                self.range_expansions.append(
                    {
                        "path": path,
                        "old_range": [old_lo, old_hi],
                        "expanded_range": [model_lo, model_hi],
                    }
                )
                warnings.warn(
                    f"Expanded model range for {path!r} from {[old_lo, old_hi]} "
                    f"to {[model_lo, model_hi]} to cover prepared data range {[data_lo, data_hi]}",
                    stacklevel=2,
                )
            self.reg_sources[name].append(
                {
                    "kind": "const",
                    "path": path,
                    "index": int(index),
                    "data_lo": float(data_lo),
                    "data_hi": float(data_hi),
                    "model_lo": model_lo,
                    "model_hi": model_hi,
                }
            )

        for index, (path, vocab) in enumerate(schema["cat_vocab"].items()):
            name = strip_top_prefix(path)
            if name in UNSUPPORTED_GARMENTCODE_PARAMS:
                self.ignored_prepared_targets.append(path)
                continue
            spec = self.spec_by_name.get(name)
            if spec is None or not spec.is_classification:
                problems.append(f"categorical target {path!r} has no classification head")
                continue
            missing = [value for value in vocab if value not in spec.choices]
            if missing:
                problems.append(f"categorical target {path!r} has unknown choices {missing!r}")
                continue
            class_map = [spec.choices.index(value) for value in vocab]
            self.cat_sources[name].append(
                {
                    "kind": "cat",
                    "path": path,
                    "index": index,
                    "class_map": class_map,
                }
            )

        if problems:
            details = "\n  - ".join(problems)
            raise ValueError(f"Prepared data is incompatible with model.py:\n  - {details}")

        self.unsupervised_regression = [
            name for name, sources in self.reg_sources.items() if not sources
        ]
        self.unsupervised_classification = [
            name for name, sources in self.cat_sources.items() if not sources
        ]
        self.class_weighting = "none"

    def configure_class_weights(
        self,
        dataset: Any,
        method: str,
        beta: float = 0.999,
        max_weight: float = 5.0,
    ) -> None:
        """Estimate categorical weights from unique training garments."""
        self.class_weighting = method
        if method == "none":
            return
        if not 0.0 <= beta < 1.0:
            raise ValueError("--class-weight-beta must be in [0, 1)")
        if max_weight <= 0:
            raise ValueError("--class-weight-max must be positive")

        rows = torch.tensor(
            [dataset.row_of[gid] for gid in dataset.garment_ids], dtype=torch.long
        )
        for name, sources in self.cat_sources.items():
            num_classes = len(self.spec_by_name[name].choices)
            for source in sources:
                if source["kind"] == "cat":
                    target = torch.from_numpy(
                        dataset.y_cat[:, source["index"]].copy()
                    )[rows]
                    valid = target.ne(-1)
                    lookup = torch.tensor(source["class_map"], dtype=torch.long)
                    mapped = lookup[target[valid].long()]
                else:
                    target = torch.from_numpy(
                        dataset.y_const_raw[:, source["index"]].copy()
                    )[rows]
                    valid = torch.from_numpy(
                        dataset.const_mask[:, source["index"]].copy()
                    )[rows].bool()
                    choices = torch.tensor(source["choices"], dtype=target.dtype)
                    mapped = (
                        target[valid].unsqueeze(-1) - choices
                    ).abs().argmin(dim=-1)

                counts = torch.bincount(mapped, minlength=num_classes).float()
                observed = counts.gt(0)
                weights = torch.ones_like(counts)
                if method == "balanced":
                    weights[observed] = counts[observed].sum() / (
                        observed.sum() * counts[observed]
                    )
                else:
                    weights[observed] = (1.0 - beta) / (
                        1.0 - beta ** counts[observed]
                    )
                if observed.any():
                    weights[observed] /= weights[observed].mean()
                weights.clamp_(max=max_weight)
                source["class_weights"] = weights.tolist()

    def is_supervised(self, name: str) -> bool:
        return bool(self.reg_sources.get(name) or self.cat_sources.get(name))

    def summary(self) -> dict[str, Any]:
        return {
            "prepared_numeric_slots": int(self.schema["n_cont"])
            + int(self.schema["n_const"]),
            "prepared_categorical_fields": int(self.schema["n_cat"]),
            "supervised_regression_heads": sum(bool(v) for v in self.reg_sources.values()),
            "supervised_categorical_heads": sum(bool(v) for v in self.cat_sources.values()),
            "unsupervised_regression_heads": self.unsupervised_regression,
            "unsupervised_categorical_heads": self.unsupervised_classification,
            "range_expansions": self.range_expansions,
            "ignored_prepared_targets": self.ignored_prepared_targets,
            "class_weighting": self.class_weighting,
        }


def forward_modelpy(model: nn.Module, images: torch.Tensor) -> dict[str, Any]:
    if images.ndim != 4:
        raise ValueError(f"Expected image batch [B,C,H,W], got {tuple(images.shape)}")
    return model(images)


def compute_modelpy_losses(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    adapter: PreparedTargetAdapter,
    lambda_cat: float,
    label_smoothing: float = 0.0,
    reg_loss: str = "mse",
    smooth_l1_beta: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    first_param = next(iter(outputs["params"].values()))
    first_tensor = next(iter(first_param.values()))
    regression_sum = first_tensor.new_zeros(())
    regression_count = first_tensor.new_zeros(())

    for name, sources in adapter.reg_sources.items():
        prediction = outputs["params"][name]["normalized"]
        for source in sources:
            index = source["index"]
            if source["kind"] == "cont":
                target = batch["y_cont"][:, index]
                mask = batch["mask"][:, index]
            else:
                target = batch["y_const"][:, index]
                data_span = max(source["data_hi"] - source["data_lo"], 1e-8)
                raw_target = source["data_lo"] + target * data_span
                model_span = max(source["model_hi"] - source["model_lo"], 1e-8)
                target = (raw_target - source["model_lo"]) / model_span
                mask = batch["const_mask"][:, index]
            if reg_loss == "smooth_l1":
                per_item = F.smooth_l1_loss(
                    prediction,
                    target,
                    reduction="none",
                    beta=smooth_l1_beta,
                )
            else:
                per_item = (prediction - target).pow(2)
            regression_sum = regression_sum + (per_item * mask).sum()
            regression_count = regression_count + mask.sum()

    loss_reg = regression_sum / regression_count.clamp(min=1)
    loss_cat = first_tensor.new_zeros(())
    active_fields = 0
    correct = 0
    total = 0
    for name, sources in adapter.cat_sources.items():
        logits = outputs["params"][name]["logits"]
        for source in sources:
            if source["kind"] == "cat":
                target = batch["y_cat"][:, source["index"]]
                valid = target.ne(-1)
                if not valid.any():
                    continue
                lookup = torch.tensor(
                    source["class_map"], device=target.device, dtype=target.dtype
                )
                mapped_target = target.clone()
                mapped_target[valid] = lookup[target[valid]]
            else:
                target_norm = batch["y_const"][:, source["index"]]
                valid = batch["const_mask"][:, source["index"]].bool()
                if not valid.any():
                    continue
                data_span = max(source["data_hi"] - source["data_lo"], 1e-8)
                raw_target = source["data_lo"] + target_norm * data_span
                choices = torch.tensor(
                    source["choices"], device=raw_target.device, dtype=raw_target.dtype
                )
                distances = (raw_target.unsqueeze(-1) - choices).abs()
                min_distance, mapped_target = distances.min(dim=-1)
                if (min_distance[valid] > 1e-4).any():
                    bad_value = float(raw_target[valid][min_distance[valid].argmax()].detach().cpu())
                    raise ValueError(
                        f"Constant target {source['path']!r} value {bad_value} is not "
                        f"one of model.py choices {source['choices']!r}"
                    )
                mapped_target = mapped_target.long()
                mapped_target[~valid] = -1
            class_weights = source.get("class_weights")
            loss_cat = loss_cat + F.cross_entropy(
                logits,
                mapped_target,
                ignore_index=-1,
                label_smoothing=label_smoothing,
                weight=(
                    torch.as_tensor(
                        class_weights, device=logits.device, dtype=logits.dtype
                    )
                    if class_weights is not None
                    else None
                ),
            )
            active_fields += 1
            prediction = logits.argmax(dim=-1)
            correct += prediction[valid].eq(mapped_target[valid]).sum().item()
            total += valid.sum().item()

    if active_fields:
        loss_cat = loss_cat / active_fields
    loss = loss_reg + lambda_cat * loss_cat
    values = {
        "loss": float(loss.detach().cpu()),
        "loss_reg": float(loss_reg.detach().cpu()),
        "loss_cat": float(loss_cat.detach().cpu()),
        "cat_acc": correct / total if total else 0.0,
    }
    return loss, values


def run_epoch(
    model: nn.Module,
    architecture: str,
    loader: Any,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any,
    device: torch.device,
    vocab_sizes: list[int],
    adapter: PreparedTargetAdapter | None,
    lambda_cat: float,
    label_smoothing: float,
    reg_loss: str,
    smooth_l1_beta: float,
    grad_clip_norm: float,
    amp: bool,
    log_every: int,
    max_batches: int | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    if architecture == "baseline" and model.freeze_backbone:
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
                if architecture == "baseline":
                    pred_reg, pred_logits = model(batch["image"])
                    loss, values = compute_baseline_losses(
                        pred_reg, pred_logits, batch, vocab_sizes, lambda_cat
                    )
                else:
                    assert adapter is not None
                    outputs = forward_modelpy(model, batch["image"])
                    loss, values = compute_modelpy_losses(
                        outputs,
                        batch,
                        adapter,
                        lambda_cat,
                        label_smoothing=label_smoothing,
                        reg_loss=reg_loss,
                        smooth_l1_beta=smooth_l1_beta,
                    )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        (parameter for parameter in model.parameters() if parameter.requires_grad),
                        grad_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()

            batch_size = int(batch["image"].shape[0])
            stats.update(values, batch_size)
            if is_train and log_every > 0 and step % log_every == 0:
                avg = stats.averages()
                print(
                    f"  step {step:04d} loss={avg['loss']:.4f} "
                    f"reg={avg['loss_reg']:.4f} cat={avg['loss_cat']:.4f} "
                    f"acc={avg['cat_acc']:.3f} ({time.time() - started:.1f}s)",
                    flush=True,
                )
    return stats.averages()


def build_training_model(
    args: argparse.Namespace,
    schema: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, PreparedTargetAdapter | None, dict[str, Any] | None]:
    if args.architecture == "baseline":
        model = GarmentDinoModel(
            backbone_name=args.backbone,
            reg_dim=int(schema["n_cont"]) + int(schema["n_const"]),
            cat_vocab_sizes=cat_vocab_sizes(schema),
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            freeze_backbone=not args.unfreeze_backbone,
            bounded_regression=args.bounded_regression,
        )
        return model.to(device), None, None

    ensure_modelpy_relu_compatibility()
    from model import GarmentCodeDINOv2MLP

    schema_path = Path(args.model_schema).expanduser().resolve()
    if not schema_path.is_file():
        raise FileNotFoundError(f"model.py schema not found: {schema_path}")
    dinov2_dir = resolve_dinov2_dir(args.dinov2_dir)
    args.model_schema = str(schema_path)
    args.dinov2_dir = str(dinov2_dir)
    with open(schema_path) as handle:
        model_schema = yaml.safe_load(handle)

    model = GarmentCodeDINOv2MLP(
        model_name=args.backbone,
        head_hidden_dims=tuple(args.head_hidden_dims),
        shared_hidden_dim=args.shared_hidden_dim,
        dropout=args.dropout,
        head_layer_norm=args.head_layer_norm,
        exclude_unsupported_params=args.exclude_unsupported_params,
        pretrained=True,
        freeze_encoder=not args.unfreeze_backbone,
        normalize_images=False,
        schema_path=schema_path,
        dinov2_dir=dinov2_dir,
    ).to(device)
    adapter = PreparedTargetAdapter(schema, model.param_specs)
    return model, adapter, model_schema


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    args: argparse.Namespace,
    schema: dict[str, Any],
    model_schema: dict[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "architecture": args.architecture,
        "epoch": epoch,
        "best_val": best_val,
        "args": checkpoint_args(args),
        "schema": schema,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if model_schema is not None:
        payload["model_schema"] = model_schema
    torch.save(payload, path)


def train(args: argparse.Namespace) -> None:
    configure_torch_home()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    if args.out_dir is None:
        args.out_dir = f"runs/{args.architecture}_{args.backbone}"

    train_loader, val_loader, schema = make_loaders(args)
    vocab_sizes = cat_vocab_sizes(schema)
    model, adapter, model_schema = build_training_model(args, schema, device)
    if adapter is not None:
        adapter.configure_class_weights(
            train_loader.dataset,
            args.class_weighting,
            beta=args.class_weight_beta,
            max_weight=args.class_weight_max,
        )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.lr_plateau_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience,
            min_lr=args.min_lr,
        )
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"device={device} architecture={args.architecture} mode={args.mode} "
        f"backbone={args.backbone}",
        flush=True,
    )
    print(
        f"train samples={len(train_loader.dataset)} "
        f"garments={len(train_loader.dataset.garment_ids)}; "
        f"val samples={len(val_loader.dataset)} "
        f"garments={len(val_loader.dataset.garment_ids)}",
        flush=True,
    )
    print(f"parameters trainable={trainable:,} total={total:,}", flush=True)
    if adapter is not None:
        print(f"target adapter={json.dumps(adapter.summary(), sort_keys=True)}", flush=True)

    wandb_run = init_wandb(args, out_dir, schema, train_loader, val_loader, trainable, total, adapter)

    best_val = math.inf
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        print(f"\nepoch {epoch}/{args.epochs}", flush=True)
        train_metrics = run_epoch(
            model,
            args.architecture,
            train_loader,
            optimizer,
            scaler,
            device,
            vocab_sizes,
            adapter,
            args.lambda_cat,
            args.label_smoothing,
            args.reg_loss,
            args.smooth_l1_beta,
            args.grad_clip_norm,
            args.amp,
            args.log_every,
            args.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            args.architecture,
            val_loader,
            None,
            scaler,
            device,
            vocab_sizes,
            adapter,
            args.lambda_cat,
            args.label_smoothing,
            args.reg_loss,
            args.smooth_l1_beta,
            0.0,
            args.amp,
            0,
            args.max_val_batches,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"  train loss={train_metrics['loss']:.4f} reg={train_metrics['loss_reg']:.4f} "
            f"cat={train_metrics['loss_cat']:.4f} acc={train_metrics['cat_acc']:.3f}",
            flush=True,
        )
        print(
            f"  val   loss={val_metrics['loss']:.4f} reg={val_metrics['loss_reg']:.4f} "
            f"cat={val_metrics['loss_cat']:.4f} acc={val_metrics['cat_acc']:.3f}",
            flush=True,
        )

        improved = val_metrics["loss"] < best_val - args.min_delta
        if improved:
            best_val = val_metrics["loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step(val_metrics["loss"])
        log_wandb_epoch(wandb_run, epoch, train_metrics, val_metrics, best_val, optimizer)
        if improved:
            save_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_val,
                args,
                schema,
                model_schema,
            )
        save_checkpoint(
            out_dir / "last.pt",
            model,
            optimizer,
            epoch,
            best_val,
            args,
            schema,
            model_schema,
        )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                out_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                best_val,
                args,
                schema,
                model_schema,
            )
        with open(out_dir / "history.json", "w") as handle:
            json.dump({"args": checkpoint_args(args), "history": history}, handle, indent=2)
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"early stopping after {epoch} epochs; best val loss={best_val:.4f}",
                flush=True,
            )
            break

    config = {
        "architecture": args.architecture,
        "args": checkpoint_args(args),
        "schema_summary": asdict_summary(schema),
    }
    if adapter is not None:
        config["target_adapter"] = adapter.summary()
    with open(out_dir / "config.json", "w") as handle:
        json.dump(config, handle, indent=2)
    if wandb_run is not None:
        wandb_run.summary["best/val_loss"] = best_val
        wandb_run.finish()


def checkpoint_architecture(checkpoint: dict[str, Any]) -> str:
    architecture = checkpoint.get("architecture")
    if architecture is None:
        architecture = checkpoint.get("args", {}).get("architecture", "baseline")
    if architecture not in {"baseline", "modelpy"}:
        raise ValueError(f"Unknown checkpoint architecture: {architecture!r}")
    return architecture


def embedded_schema_file(checkpoint: dict[str, Any]) -> str:
    model_schema = checkpoint.get("model_schema")
    if model_schema is None:
        raise FileNotFoundError(
            "The model.py schema path stored in this checkpoint is missing, and the "
            "checkpoint has no embedded model_schema. Pass --model-schema."
        )
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    try:
        yaml.safe_dump(model_schema, handle, sort_keys=False)
        return handle.name
    finally:
        handle.close()


def build_modelpy_inference_model(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, PreparedTargetAdapter]:
    ensure_modelpy_relu_compatibility()
    from model import GarmentCodeDINOv2MLP

    train_args = checkpoint["args"]
    temporary_schema: str | None = None
    # Old checkpoints predate the unsupported-parameter contract. Their embedded
    # schema is the only version compatible with the saved MLP head names.
    legacy_checkpoint = "exclude_unsupported_params" not in train_args
    requested_schema = None if legacy_checkpoint else (
        args.model_schema or train_args.get("model_schema")
    )
    if requested_schema and Path(requested_schema).expanduser().is_file():
        schema_path = str(Path(requested_schema).expanduser().resolve())
    else:
        temporary_schema = embedded_schema_file(checkpoint)
        schema_path = temporary_schema

    try:
        model = GarmentCodeDINOv2MLP(
            model_name=train_args.get("backbone", "dinov2_vits14"),
            head_hidden_dims=tuple(train_args.get("head_hidden_dims", (256, 128))),
            shared_hidden_dim=train_args.get("shared_hidden_dim"),
            dropout=float(train_args.get("dropout", 0.1)),
            head_layer_norm=bool(train_args.get("head_layer_norm", False)),
            # Older checkpoints include the legacy heads and need strict state-dict compatibility.
            exclude_unsupported_params=bool(train_args.get("exclude_unsupported_params", False)),
            pretrained=False,
            freeze_encoder=not bool(train_args.get("unfreeze_backbone", False)),
            normalize_images=False,
            schema_path=schema_path,
            dinov2_dir=resolve_dinov2_dir(args.dinov2_dir or train_args.get("dinov2_dir")),
        )
        model.load_state_dict(checkpoint["model"])
    finally:
        if temporary_schema is not None:
            Path(temporary_schema).unlink(missing_ok=True)
    model = model.to(device).eval()
    return model, PreparedTargetAdapter(checkpoint["schema"], model.param_specs)


def modelpy_prediction_json(
    model: nn.Module,
    outputs: dict[str, Any],
    adapter: PreparedTargetAdapter,
) -> dict[str, Any]:
    regression = {}
    categoricals = {}
    for spec in model.param_specs:
        prediction = outputs["params"][spec.name]
        supervised = adapter.is_supervised(spec.name)
        if spec.is_regression:
            regression[spec.name] = {
                "normalized": float(prediction["normalized"][0].detach().cpu()),
                "value": float(prediction["value"][0].detach().cpu()),
                "supervised": supervised,
            }
        else:
            probs = prediction["probs"][0].detach().cpu()
            index = int(probs.argmax())
            categoricals[spec.name] = {
                "value": spec.choices[index],
                "index": index,
                "confidence": float(probs[index]),
                "supervised": supervised,
            }
    return {"regression": regression, "categoricals": categoricals}


def set_design_value(design: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = design
    for key in path[:-1]:
        node = node[key]
    node[path[-1]]["v"] = value


def decode_modelpy_design(
    model: nn.Module,
    outputs: dict[str, Any],
    adapter: PreparedTargetAdapter,
) -> dict[str, Any]:
    design = model.decode(outputs)
    for spec in model.param_specs:
        if not adapter.is_supervised(spec.name):
            set_design_value(design["design"], spec.path, copy.deepcopy(spec.default))
    return design


def infer(args: argparse.Namespace) -> None:
    if bool(args.gid) == bool(args.image):
        raise SystemExit("Provide exactly one of --gid or --image")
    configure_torch_home()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    detected = checkpoint_architecture(checkpoint)
    if args.architecture != "auto" and args.architecture != detected:
        raise SystemExit(
            f"Checkpoint architecture is {detected!r}, not requested {args.architecture!r}"
        )
    architecture = detected

    if architecture == "baseline":
        model = build_baseline_inference_model(checkpoint, device)
        adapter = None
    else:
        model, adapter = build_modelpy_inference_model(checkpoint, args, device)

    image_path = args.image
    source: dict[str, Any] = {"image": args.image}
    if args.gid:
        image_path = path_from_gid(args.prepared_dir, args.gid, args.frame, args.view)
        source = {
            "gid": args.gid,
            "frame": args.frame,
            "view": args.view,
            "image": image_path,
        }
    image_size = int(checkpoint["args"].get("image_size", 224))
    image = load_image(image_path, image_size, device)

    with torch.inference_mode():
        if architecture == "baseline":
            pred_reg, pred_logits = model(image)
            prediction = decode_baseline_prediction(
                pred_reg, pred_logits, checkpoint["schema"]
            )
            design = None
        else:
            assert adapter is not None
            outputs = forward_modelpy(model, image)
            prediction = modelpy_prediction_json(model, outputs, adapter)
            design = decode_modelpy_design(model, outputs, adapter)

    result = {
        "architecture": architecture,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_val": checkpoint.get("best_val"),
        "source": source,
        "prediction": prediction,
    }
    stem = f"infer_{args.gid}" if args.gid else f"infer_{Path(args.image).stem}"
    out_dir = Path(args.checkpoint).parent / "inference"
    out_json = Path(args.out) if args.out else out_dir / f"{stem}.json"
    out_yaml = Path(args.yaml_out) if args.yaml_out else out_dir / f"{stem}.yaml"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"wrote {out_json}")

    if architecture == "baseline":
        write_baseline_yaml(result, args.template, args.source_mode, str(out_yaml))
    else:
        out_yaml.parent.mkdir(parents=True, exist_ok=True)
        with open(out_yaml, "w") as handle:
            yaml.safe_dump(design, handle, sort_keys=False, default_flow_style=False)
        print(f"wrote {out_yaml}")

    if args.render_3d:
        render_python = ROOT / ".envs" / "garmentcode" / "bin" / "python"
        render_script = ROOT / "render_garmentcode.py"
        if not render_python.is_file():
            raise FileNotFoundError(f"GarmentCode Python environment not found: {render_python}")
        command = [
            str(render_python),
            str(render_script),
            "--design",
            str(out_yaml.resolve()),
            "--resolution-scale",
            str(args.render_resolution_scale),
        ]
        if args.render_out_dir:
            command.extend(["--out-dir", args.render_out_dir])
        for field in ("upper", "wb", "bottom"):
            value = getattr(args, f"render_{field}")
            if value is not None:
                command.extend([f"--override-{field}", value])
        print("starting GarmentCode drape and render", flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"Inference outputs were written successfully, but GarmentCode rendering failed "
                f"with exit code {exc.returncode}. See the renderer message above."
            ) from None


def add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="train either architecture", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    train_parser.add_argument("--architecture", choices=("baseline", "modelpy"), default="baseline")
    train_parser.add_argument("--prepared-dir", default="prepared_v2")
    train_parser.add_argument("--out-dir")
    train_parser.add_argument("--backbone", default="dinov2_vitl14")
    train_parser.add_argument("--mode", choices=("single", "all_images"), default="single")
    train_parser.add_argument("--augmentation", choices=("none", "light"), default="light")
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--epochs", type=int, default=1000)
    train_parser.add_argument("--save-every", type=int, default=100, help="save numbered checkpoints every N epochs; 0 disables")
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--hidden-dim", type=int, default=512, help="baseline head width")
    train_parser.add_argument(
        "--head-hidden-dims", type=int, nargs="+", default=[64, 32], help="model.py per-head widths"
    )
    train_parser.add_argument(
        "--shared-hidden-dim", type=int, help="model.py shared projection width"
    )
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--head-layer-norm", action="store_true", help="add LayerNorm inside each model.py MLP head")
    train_parser.add_argument(
        "--include-unsupported-params",
        action="store_false",
        dest="exclude_unsupported_params",
        help="legacy compatibility mode: include shirt.openfront and waistband.height heads",
    )
    train_parser.set_defaults(exclude_unsupported_params=True)
    train_parser.add_argument("--label-smoothing", type=float, default=0.05, help="categorical label smoothing")
    train_parser.add_argument(
        "--class-weighting",
        choices=("none", "balanced", "effective"),
        default="effective",
        help="categorical class weighting estimated from unique training garments",
    )
    train_parser.add_argument("--class-weight-beta", type=float, default=0.999)
    train_parser.add_argument("--class-weight-max", type=float, default=5.0)
    train_parser.add_argument("--reg-loss", choices=("mse", "smooth_l1"), default="smooth_l1", help="regression loss for normalized numeric targets")
    train_parser.add_argument("--smooth-l1-beta", type=float, default=0.05, help="beta for SmoothL1 regression loss")
    train_parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="clip trainable gradient norm; 0 disables")
    train_parser.add_argument("--lr-plateau-patience", type=int, default=10, help="ReduceLROnPlateau patience; 0 disables")
    train_parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    train_parser.add_argument("--min-lr", type=float, default=1e-6)
    train_parser.add_argument("--early-stopping-patience", type=int, default=30, help="stop after this many unimproved epochs; 0 disables")
    train_parser.add_argument("--min-delta", type=float, default=0.0, help="minimum val loss improvement for best/early stopping")
    train_parser.add_argument("--lambda-cat", type=float, default=1.0)
    train_parser.add_argument(
        "--unbounded-regression",
        action="store_false",
        dest="bounded_regression",
        help="legacy mode: do not apply sigmoid to baseline regression outputs",
    )
    train_parser.set_defaults(bounded_regression=True)
    train_parser.add_argument("--unfreeze-backbone", action="store_true")
    train_parser.add_argument("--amp", action="store_true", help="use CUDA mixed precision")
    add_device_argument(train_parser)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--log-every", type=int, default=25)
    train_parser.add_argument("--max-train-batches", type=int)
    train_parser.add_argument("--max-val-batches", type=int)
    train_parser.add_argument(
        "--image-path-prefix", nargs=2, action="append", default=[], metavar=("OLD", "NEW")
    )
    train_parser.add_argument("--keep-missing-images", action="store_true")
    train_parser.add_argument("--allow-unbalanced-data", action="store_true",
                              help="override a failed balance_report.json readiness gate")
    train_parser.add_argument("--model-schema", default=str(DEFAULT_MODEL_SCHEMA))
    train_parser.add_argument("--dinov2-dir", help="local DINOv2 repository for model.py")
    train_parser.add_argument("--wandb", action="store_true", help="log training and validation curves to Weights & Biases")
    train_parser.add_argument("--wandb-project", default="ImageToGarment")
    train_parser.add_argument("--wandb-entity")
    train_parser.add_argument("--wandb-run-name")
    train_parser.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    train_parser.add_argument("--wandb-tags", nargs="*", default=[])
    train_parser.set_defaults(func=train)

    infer_parser = subparsers.add_parser(
        "infer", help="infer with either checkpoint", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    infer_parser.add_argument("--architecture", choices=("auto", "baseline", "modelpy"), default="auto")
    infer_parser.add_argument("--checkpoint", default="runs/dinov2_vits14/best.pt")
    infer_parser.add_argument("--prepared-dir", default="prepared_v2")
    infer_parser.add_argument("--gid")
    infer_parser.add_argument("--image")
    infer_parser.add_argument("--frame", default="0")
    infer_parser.add_argument("--view", type=int, choices=(0, 1, 2, 3), default=0)
    infer_parser.add_argument("--out")
    infer_parser.add_argument("--yaml-out")
    infer_parser.add_argument("--template", default=str(DEFAULT_MODEL_SCHEMA))
    infer_parser.add_argument("--source-mode", choices=("split", "wholebody", "all"), default="split")
    infer_parser.add_argument("--model-schema", help="override the model.py YAML schema")
    infer_parser.add_argument("--dinov2-dir", help="override the local DINOv2 repository")
    infer_parser.add_argument(
        "--render-3d", action="store_true", help="drape and render the inferred YAML with GarmentCode"
    )
    infer_parser.add_argument(
        "--render-resolution-scale",
        type=float,
        default=3.0,
        help="GarmentCode mesh edge length in cm; use 1 for the official high-resolution setting",
    )
    infer_parser.add_argument(
        "--render-out-dir", help="GarmentCode output directory (defaults to runs/garmentcode_reconstructions)"
    )
    infer_parser.add_argument(
        "--render-upper",
        choices=("none", "FittedShirt", "Shirt"),
        help="override the inferred upper topology for rendering; 'none' removes it",
    )
    infer_parser.add_argument(
        "--render-wb",
        choices=("none", "StraightWB", "FittedWB"),
        help="override the inferred waistband topology for rendering; 'none' removes it",
    )
    infer_parser.add_argument(
        "--render-bottom",
        choices=("none", "SkirtCircle", "AsymmSkirtCircle", "GodetSkirt", "PencilSkirt", "Skirt2", "Pants"),
        help="override the inferred bottom topology for rendering; 'none' removes it",
    )
    add_device_argument(infer_parser)
    infer_parser.set_defaults(func=infer)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
