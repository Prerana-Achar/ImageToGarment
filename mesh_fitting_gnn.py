"""MeshGraphNets-style template deformation with topology-agnostic losses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn


@dataclass
class Mesh:
    vertices: Tensor
    faces: Tensor

    def to(self, device: torch.device | str) -> "Mesh":
        return Mesh(self.vertices.to(device), self.faces.to(device))


def load_mesh(path: str | Path) -> Mesh:
    """Load a triangular OBJ or an NPZ containing vertices/verts and faces."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".obj":
        vertices, faces = [], []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                fields = raw_line.strip().split()
                if not fields:
                    continue
                if fields[0] == "v" and len(fields) >= 4:
                    vertices.append([float(value) for value in fields[1:4]])
                elif fields[0] == "f" and len(fields) >= 4:
                    polygon = []
                    for field in fields[1:]:
                        vertex_index = int(field.split("/")[0])
                        polygon.append(
                            vertex_index - 1 if vertex_index > 0 else len(vertices) + vertex_index
                        )
                    for index in range(1, len(polygon) - 1):
                        faces.append([polygon[0], polygon[index], polygon[index + 1]])
        mesh = Mesh(
            torch.tensor(vertices, dtype=torch.float32),
            torch.tensor(faces, dtype=torch.long).reshape(-1, 3),
        )
    elif suffix == ".npz":
        with np.load(path) as archive:
            vertex_key = "vertices" if "vertices" in archive else "verts"
            if vertex_key not in archive or "faces" not in archive:
                raise ValueError(f"{path} must contain vertices (or verts) and faces")
            mesh = Mesh(
                torch.as_tensor(archive[vertex_key], dtype=torch.float32),
                torch.as_tensor(archive["faces"], dtype=torch.long),
            )
    else:
        raise ValueError(f"Unsupported mesh format {suffix!r}; expected .obj or .npz")
    _validate_mesh(mesh, path)
    return mesh


def save_obj(path: str | Path, vertices: Tensor, faces: Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = vertices.detach().cpu()
    faces = faces.detach().cpu()
    with path.open("w", encoding="utf-8") as handle:
        for vertex in vertices.tolist():
            handle.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        for face in faces.tolist():
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def _validate_mesh(mesh: Mesh, path: Path) -> None:
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3:
        raise ValueError(f"{path}: vertices must have shape [V, 3]")
    if mesh.vertices.shape[0] == 0:
        raise ValueError(f"{path}: mesh has no vertices")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError(f"{path}: faces must have shape [F, 3]")
    if mesh.faces.numel() and (
        mesh.faces.min().item() < 0 or mesh.faces.max().item() >= mesh.vertices.shape[0]
    ):
        raise ValueError(f"{path}: a face references an invalid vertex")


def faces_to_edge_index(faces: Tensor) -> Tensor:
    """Return unique directed mesh edges as a [2, E] tensor."""
    if faces.numel() == 0:
        raise ValueError("The template needs faces to define its graph")
    edges = torch.cat(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), dim=0
    )
    edges = torch.cat((edges, edges.flip(1)), dim=0)
    return torch.unique(edges, dim=0).t().contiguous()


def vertex_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    face_normals = torch.linalg.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
        dim=-1,
    )
    normals = torch.zeros_like(vertices)
    for corner in range(3):
        normals.index_add_(0, faces[:, corner], face_normals)
    return normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    layer_norm: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    ]
    if layer_norm:
        layers.append(nn.LayerNorm(output_dim))
    return nn.Sequential(*layers)


class TemplateDeformationGNN(nn.Module):
    """Predict one xyz displacement for every vertex of a triangular template."""

    def __init__(
        self,
        latent_dim: int = 128,
        message_passing_steps: int = 8,
        max_displacement_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.message_passing_steps = message_passing_steps
        self.max_displacement_ratio = max_displacement_ratio
        self.node_encoder = _mlp(6, latent_dim, latent_dim)
        self.edge_encoder = _mlp(4, latent_dim, latent_dim)
        self.edge_blocks = nn.ModuleList(
            [_mlp(3 * latent_dim, latent_dim, latent_dim) for _ in range(message_passing_steps)]
        )
        self.node_blocks = nn.ModuleList(
            [_mlp(2 * latent_dim, latent_dim, latent_dim) for _ in range(message_passing_steps)]
        )
        self.decoder = _mlp(2 * latent_dim, latent_dim, 3, layer_norm=False)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, vertices: Tensor, faces: Tensor) -> Tensor:
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape [V, 3]")
        edge_index = faces_to_edge_index(faces)
        source, destination = edge_index

        center = vertices.mean(dim=0, keepdim=True)
        centered = vertices - center
        scale = centered.square().sum(dim=-1).mean().sqrt().clamp_min(1e-8)
        normalized = centered / scale
        normals = vertex_normals(vertices, faces)
        node_latent = self.node_encoder(torch.cat((normalized, normals), dim=-1))

        relative = normalized[source] - normalized[destination]
        edge_latent = self.edge_encoder(
            torch.cat((relative, relative.norm(dim=-1, keepdim=True)), dim=-1)
        )
        for edge_block, node_block in zip(self.edge_blocks, self.node_blocks):
            edge_latent = edge_latent + edge_block(
                torch.cat(
                    (edge_latent, node_latent[source], node_latent[destination]), dim=-1
                )
            )
            aggregate = torch.zeros_like(node_latent)
            aggregate.index_add_(0, destination, edge_latent)
            degree = torch.zeros(
                vertices.shape[0], 1, dtype=vertices.dtype, device=vertices.device
            )
            degree.index_add_(
                0,
                destination,
                torch.ones(
                    destination.shape[0], 1, dtype=vertices.dtype, device=vertices.device
                ),
            )
            aggregate = aggregate / degree.clamp_min(1)
            node_latent = node_latent + node_block(
                torch.cat((node_latent, aggregate), dim=-1)
            )

        global_latent = node_latent.mean(dim=0, keepdim=True).expand_as(node_latent)
        normalized_delta = self.decoder(torch.cat((node_latent, global_latent), dim=-1))
        if self.max_displacement_ratio is not None:
            normalized_delta = (
                torch.tanh(normalized_delta) * self.max_displacement_ratio
            )
        return normalized_delta * scale

    def deform(self, vertices: Tensor, faces: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the fitted vertices and the per-vertex `delta_x`."""
        delta_x = self(vertices, faces)
        return vertices + delta_x, delta_x


def sample_mesh_surface(
    vertices: Tensor,
    faces: Tensor,
    num_samples: int,
    *,
    area_vertices: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Sample points by face area; gradients flow to ``vertices``."""
    if faces.numel() == 0:
        indices = torch.randint(
            vertices.shape[0],
            (num_samples,),
            device=vertices.device,
            generator=generator,
        )
        return vertices[indices]
    area_source = vertices if area_vertices is None else area_vertices
    triangles = area_source[faces]
    twice_area = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=-1,
    ).norm(dim=-1)
    probabilities = twice_area.clamp_min(1e-12)
    probabilities = probabilities / probabilities.sum()
    face_indices = torch.multinomial(
        probabilities, num_samples, replacement=True, generator=generator
    )
    selected = vertices[faces[face_indices]]
    random_values = torch.rand(
        num_samples, 2, device=vertices.device, dtype=vertices.dtype, generator=generator
    )
    root = random_values[:, :1].sqrt()
    barycentric = torch.cat(
        (1 - root, root * (1 - random_values[:, 1:]), root * random_values[:, 1:]),
        dim=-1,
    )
    return (selected * barycentric.unsqueeze(-1)).sum(dim=1)


def _nearest_squared_distance(
    query: Tensor, reference: Tensor, chunk_size: int
) -> Tensor:
    minima = []
    for start in range(0, query.shape[0], chunk_size):
        distances = torch.cdist(query[start : start + chunk_size], reference)
        minima.append(distances.square().min(dim=1).values)
    return torch.cat(minima)


def symmetric_chamfer_distance(
    first: Tensor, second: Tensor, chunk_size: int = 2048
) -> Tensor:
    return 0.5 * (
        _nearest_squared_distance(first, second, chunk_size).mean()
        + _nearest_squared_distance(second, first, chunk_size).mean()
    )


def mesh_fitting_loss(
    predicted_vertices: Tensor,
    template_vertices: Tensor,
    template_faces: Tensor,
    target_vertices: Tensor,
    target_faces: Optional[Tensor] = None,
    *,
    num_surface_samples: int = 4096,
    chamfer_weight: float = 1.0,
    edge_weight: float = 0.1,
    laplacian_weight: float = 0.05,
    displacement_weight: float = 1e-4,
    chunk_size: int = 2048,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compare meshes without requiring vertex correspondence or equal topology."""
    if target_faces is None:
        target_faces = torch.empty(0, 3, dtype=torch.long, device=target_vertices.device)
    predicted_points = sample_mesh_surface(
        predicted_vertices,
        template_faces,
        num_surface_samples,
        area_vertices=template_vertices,
        generator=generator,
    )
    target_points = sample_mesh_surface(
        target_vertices, target_faces, num_surface_samples, generator=generator
    )
    chamfer = symmetric_chamfer_distance(predicted_points, target_points, chunk_size)

    edge_index = faces_to_edge_index(template_faces)
    source, destination = edge_index
    template_edges = template_vertices[source] - template_vertices[destination]
    predicted_edges = predicted_vertices[source] - predicted_vertices[destination]
    edge = (
        predicted_edges.norm(dim=-1) - template_edges.norm(dim=-1)
    ).square().mean()
    delta = predicted_vertices - template_vertices
    laplacian = (delta[source] - delta[destination]).square().sum(dim=-1).mean()
    displacement = delta.square().sum(dim=-1).mean()

    terms = {
        "chamfer": chamfer,
        "edge": edge,
        "laplacian": laplacian,
        "displacement": displacement,
    }
    total = (
        chamfer_weight * chamfer
        + edge_weight * edge
        + laplacian_weight * laplacian
        + displacement_weight * displacement
    )
    return total, terms
