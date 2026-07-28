"""Train a template-deformation GNN against meshes with arbitrary topology."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mesh_fitting_gnn import (
    Mesh,
    TemplateDeformationGNN,
    load_mesh,
    mesh_fitting_loss,
    save_obj,
)


class MeshPairDataset(Dataset):
    def __init__(
        self,
        manifest: Path | None,
        split: str,
        template: Path | None = None,
        target: Path | None = None,
    ) -> None:
        if manifest is None:
            if template is None or target is None:
                raise ValueError("Pass --manifest or both --template and --target")
            records = [{"id": template.stem, "template": str(template), "target": str(target)}]
            base_dir = Path.cwd()
        else:
            with manifest.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            records = payload["pairs"] if isinstance(payload, dict) else payload
            base_dir = manifest.parent
            selected = [record for record in records if record.get("split", "train") == split]
            if selected:
                records = selected
            elif split != "train":
                records = []
        self.records = []
        for index, record in enumerate(records):
            template_path = Path(record["template"])
            target_path = Path(record["target"])
            self.records.append(
                {
                    "id": record.get("id", f"pair_{index:06d}"),
                    "template": template_path if template_path.is_absolute() else base_dir / template_path,
                    "target": target_path if target_path.is_absolute() else base_dir / target_path,
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        return {
            "id": record["id"],
            "template": load_mesh(record["template"]),
            "target": load_mesh(record["target"]),
        }


def collate_mesh_pairs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return items


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: TemplateDeformationGNN,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "chamfer": 0.0, "edge": 0.0, "laplacian": 0.0, "displacement": 0.0}
    sample_count = 0
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + epoch + (0 if training else 1_000_000))

    for batch in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        batch_loss = torch.zeros((), device=device)
        for item in batch:
            template: Mesh = item["template"].to(device)
            target: Mesh = item["target"].to(device)
            with torch.set_grad_enabled(training):
                predicted, _ = model.deform(template.vertices, template.faces)
                loss, terms = mesh_fitting_loss(
                    predicted,
                    template.vertices,
                    template.faces,
                    target.vertices,
                    target.faces,
                    num_surface_samples=args.surface_samples,
                    chamfer_weight=args.chamfer_weight,
                    edge_weight=args.edge_weight,
                    laplacian_weight=args.laplacian_weight,
                    displacement_weight=args.displacement_weight,
                    chunk_size=args.chamfer_chunk_size,
                    generator=generator,
                )
            batch_loss = batch_loss + loss / len(batch)
            totals["loss"] += loss.detach().item()
            for name, value in terms.items():
                totals[name] += value.detach().item()
            sample_count += 1
        if training:
            batch_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
    return {key: value / max(sample_count, 1) for key, value in totals.items()}


@torch.no_grad()
def save_previews(
    model: TemplateDeformationGNN,
    dataset: MeshPairDataset,
    output_dir: Path,
    device: torch.device,
    limit: int,
) -> None:
    model.eval()
    for index in range(min(limit, len(dataset))):
        item = dataset[index]
        template = item["template"].to(device)
        predicted, _ = model.deform(template.vertices, template.faces)
        save_obj(output_dir / f"{item['id']}.obj", predicted, template.faces)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="JSON list (or {'pairs': list}) of mesh pairs")
    source.add_argument("--template", type=Path, help="Template OBJ/NPZ for a single training pair")
    parser.add_argument("--target", type=Path, help="Target OBJ/NPZ used with --template")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--message-passing-steps", type=int, default=8)
    parser.add_argument("--max-displacement-ratio", type=float)
    parser.add_argument("--surface-samples", type=int, default=4096)
    parser.add_argument("--chamfer-chunk-size", type=int, default=2048)
    parser.add_argument("--chamfer-weight", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--laplacian-weight", type=float, default=0.05)
    parser.add_argument("--displacement-weight", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.template is not None and args.target is None:
        raise SystemExit("--target is required with --template")
    if args.epochs < 1 or args.batch_size < 1 or args.surface_samples < 1:
        raise SystemExit("--epochs, --batch-size, and --surface-samples must be positive")
    checkpoint_path = args.out_dir / "last.pt"
    if checkpoint_path.exists() and not args.overwrite:
        raise SystemExit(f"{checkpoint_path} exists; pass --overwrite to replace it")
    set_seed(args.seed)
    device = torch.device(args.device)
    train_dataset = MeshPairDataset(args.manifest, "train", args.template, args.target)
    if not train_dataset:
        raise SystemExit("No training pairs found")
    val_dataset = (
        MeshPairDataset(args.manifest, "val")
        if args.manifest is not None
        else train_dataset
    )
    if len(val_dataset) == 0:
        print("No validation split found; evaluating on the training pairs")
        val_dataset = train_dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_mesh_pairs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_mesh_pairs,
    )
    model = TemplateDeformationGNN(
        latent_dim=args.latent_dim,
        message_passing_steps=args.message_passing_steps,
        max_displacement_ratio=args.max_displacement_ratio,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({key: str(value) for key, value in config.items() if isinstance(value, Path)})
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history, best_loss = [], float("inf")
    for epoch in range(args.epochs):
        train_metrics = _run_epoch(model, train_loader, device, args, optimizer, epoch)
        val_metrics = _run_epoch(model, val_loader, device, args, None, epoch)
        record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(
            f"epoch={epoch + 1:04d} train={train_metrics['loss']:.6g} "
            f"val={val_metrics['loss']:.6g} chamfer={val_metrics['chamfer']:.6g}"
        )
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": {
                "latent_dim": args.latent_dim,
                "message_passing_steps": args.message_passing_steps,
                "max_displacement_ratio": args.max_displacement_ratio,
            },
            "metrics": record,
        }
        torch.save(checkpoint, checkpoint_path)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save(checkpoint, args.out_dir / "best.pt")
            save_previews(
                model, val_dataset, args.out_dir / "previews", device, args.preview_count
            )
        (args.out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
