#!/usr/bin/env python
"""Plot train/val curves from a training run's history.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/dinov2_grouped")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    hist = json.load(open(run_dir / "history.json"))["history"]
    ep = [h["epoch"] for h in hist]

    def series(split, key):
        return [h[split][key] for h in hist]

    frozen = [h["epoch"] for h in hist if h.get("classification_frozen")]
    freeze_epoch = min(frozen) if frozen else None

    panels = [
        ("loss_reg", "Regression loss (Huber)"),
        ("reg_mae", "Regression MAE"),
        ("loss_cat", "Categorical loss (CE)"),
        ("cat_acc", "Categorical accuracy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, title) in zip(axes.ravel(), panels):
        ax.plot(ep, series("train", key), "-o", ms=3, label="train", color="#2563eb")
        ax.plot(ep, series("val", key), "-o", ms=3, label="val", color="#dc2626")
        if freeze_epoch is not None:
            ax.axvline(freeze_epoch, ls="--", lw=1, color="#6b7280",
                       label=f"cat frozen (ep {freeze_epoch})")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{run_dir.name} — train vs. val", fontsize=13)
    fig.tight_layout()

    out = Path(args.out) if args.out else run_dir / "curves.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
