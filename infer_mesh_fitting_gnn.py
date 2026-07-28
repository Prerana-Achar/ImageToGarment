"""Run a trained template-deformation GNN and export fitted vertices and delta xyz."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mesh_fitting_gnn import TemplateDeformationGNN, load_mesh, save_obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out-obj", type=Path, required=True)
    parser.add_argument("--out-delta", type=Path, help="Optional NPZ containing delta_x")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = TemplateDeformationGNN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    template = load_mesh(args.template).to(device)
    with torch.no_grad():
        fitted_vertices, delta_x = model.deform(template.vertices, template.faces)
    save_obj(args.out_obj, fitted_vertices, template.faces)
    if args.out_delta is not None:
        args.out_delta.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.out_delta,
            delta_x=delta_x.cpu().numpy(),
            vertices=fitted_vertices.cpu().numpy(),
            faces=template.faces.cpu().numpy(),
        )


if __name__ == "__main__":
    main()
