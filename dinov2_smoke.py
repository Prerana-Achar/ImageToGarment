#!/usr/bin/env python
"""Quick DINOv2 setup check for this project."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load DINOv2 and run one dummy image through it.")
    parser.add_argument("--model", default="dinov2_vits14", help="torch.hub DINOv2 model name")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="device for the smoke test",
    )
    return parser.parse_args()


def main() -> None:
    root = Path(__file__).resolve().parent
    os.environ.setdefault("TORCH_HOME", str(root / ".cache" / "torch"))

    import torch

    args = parse_args()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = torch.hub.load("facebookresearch/dinov2", args.model)
    model = model.to(device).eval()

    x = torch.zeros(1, 3, 224, 224, device=device)
    with torch.inference_mode():
        y = model(x)

    print(f"loaded {args.model} on {device}")
    print(f"output shape: {tuple(y.shape)}")


if __name__ == "__main__":
    main()
