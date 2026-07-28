"""Single-image network for GarmentCode parameter reconstruction.

The network deliberately mirrors the GarmentCode parameter tree. It can use
either the original compact visual encoder or a locally cached, broadly
pretrained patch encoder. Learned garment-part queries extract semantic
evidence, root topology is decoded before component choices, and numeric heads
are conditioned on the resulting soft topology prediction.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


SEMANTIC_GROUPS = (
    "global",
    "waistband",
    "shirt",
    "collar",
    "sleeve",
    "left",
    "skirt",
    "flare-skirt",
    "godet-skirt",
    "pencil-skirt",
    "levels-skirt",
    "pants",
)


def parameter_group(path: str) -> str:
    """Map a dotted GarmentCode path to its semantic query."""
    first = path.split(".", 1)[0]
    if first == "meta":
        return "global"
    return first if first in SEMANTIC_GROUPS else "global"


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization for NCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class SpatialGatedBlock(nn.Module):
    """Large-context residual block with channel gating."""

    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        drop_path: float = 0.0,
        layer_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.project = nn.Conv2d(hidden, channels, kernel_size=1)
        self.scale = nn.Parameter(torch.full((channels, 1, 1), layer_scale))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        value, gate = self.expand(x).chunk(2, dim=1)
        x = F.gelu(value) * torch.sigmoid(gate)
        x = self.project(x) * self.scale
        return residual + self.drop_path(x)


class VisualEncoder(nn.Module):
    def __init__(
        self,
        widths: tuple[int, ...],
        depths: tuple[int, ...],
        drop_path: float,
    ) -> None:
        super().__init__()
        if len(widths) != 4 or len(depths) != 4:
            raise ValueError("VisualEncoder expects four widths and four depths")
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=4, stride=4),
            LayerNorm2d(widths[0]),
        )
        total_blocks = sum(depths)
        rates = torch.linspace(0.0, drop_path, total_blocks).tolist()
        cursor = 0
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for stage_index, (width, depth) in enumerate(zip(widths, depths)):
            blocks = [
                SpatialGatedBlock(width, drop_path=rates[cursor + block_index])
                for block_index in range(depth)
            ]
            cursor += depth
            self.stages.append(nn.Sequential(*blocks))
            if stage_index < len(widths) - 1:
                self.downsamples.append(
                    nn.Sequential(
                        LayerNorm2d(width),
                        nn.Conv2d(width, widths[stage_index + 1], 2, stride=2),
                    )
                )

    def forward(self, image: Tensor) -> list[Tensor]:
        x = self.stem(image)
        features = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        return features


class FoundationVisualEncoder(nn.Module):
    """Convert cached pretrained patch tokens into a garment feature map."""

    def __init__(
        self,
        backbone_name: str,
        backbone_repo: str,
        dimension: int,
        freeze_backbone: bool,
    ) -> None:
        super().__init__()
        repo = Path(backbone_repo).expanduser()
        if not repo.is_absolute():
            repo = Path(__file__).resolve().parent / repo
        if not (repo / "hubconf.py").is_file():
            raise FileNotFoundError(
                f"Local pretrained encoder repository is missing: {repo}"
            )
        self.backbone = torch.hub.load(
            str(repo), backbone_name, source="local", pretrained=True
        )
        self.freeze_backbone = bool(freeze_backbone)
        embed_dim = int(getattr(self.backbone, "embed_dim"))
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, dimension),
            nn.GELU(),
            nn.LayerNorm(dimension),
        )
        self.class_projection = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, dimension)
        )
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> "FoundationVisualEncoder":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, image: Tensor) -> Tensor:
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                encoded = self.backbone.forward_features(image)
        else:
            encoded = self.backbone.forward_features(image)
        patches = encoded["x_norm_patchtokens"]
        class_token = encoded["x_norm_clstoken"]
        patches = self.patch_projection(patches)
        patches = patches + self.class_projection(class_token).unsqueeze(1)
        side = int(round(patches.shape[1] ** 0.5))
        if side * side != patches.shape[1]:
            raise RuntimeError(
                "Pretrained encoder returned a non-square patch grid: "
                f"{patches.shape[1]} tokens"
            )
        return patches.transpose(1, 2).reshape(
            image.shape[0], patches.shape[-1], side, side
        )


class MultiScaleFusion(nn.Module):
    """Fuse fine contours and coarse semantic context at 1/8 resolution."""

    def __init__(self, widths: Iterable[int], dimension: int) -> None:
        super().__init__()
        widths = tuple(widths)
        self.lateral = nn.ModuleList(
            nn.Conv2d(width, dimension, kernel_size=1) for width in widths
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(dimension * len(widths), dimension, kernel_size=1),
            LayerNorm2d(dimension),
            nn.GELU(),
            nn.Conv2d(dimension, dimension, kernel_size=3, padding=1),
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        target_size = features[1].shape[-2:]
        aligned = []
        for feature, projection in zip(features, self.lateral):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature, size=target_size, mode="bilinear", align_corners=False
                )
            aligned.append(feature)
        return self.fuse(torch.cat(aligned, dim=1))


class QueryDecoderLayer(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dimension)
        self.self_attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.cross_query_norm = nn.LayerNorm(dimension)
        self.memory_norm = nn.LayerNorm(dimension)
        self.cross_attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 4, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, queries: Tensor, memory: Tensor) -> Tensor:
        normalized = self.query_norm(queries)
        queries = queries + self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        queries = queries + self.cross_attention(
            self.cross_query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )[0]
        return queries + self.ffn(self.ffn_norm(queries))


class PartQueryDecoder(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(len(SEMANTIC_GROUPS), dimension))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.coordinate_embedding = nn.Sequential(
            nn.Linear(2, dimension), nn.GELU(), nn.Linear(dimension, dimension)
        )
        self.global_projection = nn.Linear(dimension, dimension)
        self.layers = nn.ModuleList(
            QueryDecoderLayer(dimension, heads, dropout) for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(dimension)

    @staticmethod
    def coordinates(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2)

    def forward(self, feature: Tensor) -> Tensor:
        batch, channels, height, width = feature.shape
        memory = feature.flatten(2).transpose(1, 2)
        coords = self.coordinates(height, width, feature.device, feature.dtype)
        memory = memory + self.coordinate_embedding(coords)
        global_context = self.global_projection(memory.mean(dim=1, keepdim=True))
        queries = self.query.unsqueeze(0).expand(batch, -1, -1) + global_context
        for layer in self.layers:
            queries = layer(queries, memory)
        return self.output_norm(queries)


class CategoricalDecoder(nn.Module):
    def __init__(self, dimension: int, classes: int, dropout: float) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(dimension * 2),
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classes = nn.Linear(dimension, classes)
        self.activity = nn.Linear(dimension, 1)

    def forward(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        feature = self.trunk(feature)
        return self.classes(feature), self.activity(feature).squeeze(-1)


class NumericGroupDecoder(nn.Module):
    def __init__(self, dimension: int, outputs: int, dropout: float) -> None:
        super().__init__()
        self.outputs = outputs
        self.trunk = nn.Sequential(
            nn.LayerNorm(dimension * 3),
            nn.Linear(dimension * 3, dimension * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
        )
        self.output = nn.Linear(dimension, outputs * 3)
        with torch.no_grad():
            self.output.weight.mul_(0.02)
            self.output.bias[:outputs].zero_()
            self.output.bias[outputs : outputs * 2].fill_(-1.5)
            self.output.bias[outputs * 2 :].zero_()

    def forward(self, feature: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        output = self.output(self.trunk(feature))
        mean, log_scale, activity = output.split(self.outputs, dim=-1)
        return torch.sigmoid(mean), log_scale.clamp(-4.0, 2.0), activity


@dataclass(frozen=True)
class GarmentTreeConfig:
    widths: tuple[int, int, int, int] = (64, 128, 256, 384)
    depths: tuple[int, int, int, int] = (2, 2, 6, 2)
    dimension: int = 256
    query_layers: int = 3
    attention_heads: int = 8
    dropout: float = 0.15
    drop_path: float = 0.15
    encoder_kind: str = "scratch"
    backbone_name: str = "dinov2_vits14"
    backbone_repo: str = "DINOv2"
    freeze_backbone: bool = True
    hierarchical_categoricals: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GarmentTreeConfig":
        values = dict(values)
        if "widths" in values:
            values["widths"] = tuple(values["widths"])
        if "depths" in values:
            values["depths"] = tuple(values["depths"])
        return cls(**values)


class GarmentTreeNet(nn.Module):
    """Predict a complete prepared-data target from one RGB image."""

    def __init__(self, schema: dict[str, Any], config: GarmentTreeConfig | None = None) -> None:
        super().__init__()
        self.schema = schema
        self.config = config or GarmentTreeConfig()
        self.numeric_paths = [
            *schema.get("cont_slots", {}).keys(), *schema.get("const_slots", {}).keys()
        ]
        self.categorical_paths = list(schema["cat_vocab"])
        self.group_index = {name: index for index, name in enumerate(SEMANTIC_GROUPS)}

        if self.config.encoder_kind == "scratch":
            self.encoder = VisualEncoder(
                self.config.widths, self.config.depths, self.config.drop_path
            )
            self.fusion: nn.Module | None = MultiScaleFusion(
                self.config.widths, self.config.dimension
            )
        elif self.config.encoder_kind == "foundation":
            self.encoder = FoundationVisualEncoder(
                self.config.backbone_name,
                self.config.backbone_repo,
                self.config.dimension,
                self.config.freeze_backbone,
            )
            self.fusion = None
        else:
            raise ValueError(f"Unknown encoder_kind {self.config.encoder_kind!r}")
        self.part_decoder = PartQueryDecoder(
            self.config.dimension,
            self.config.attention_heads,
            self.config.query_layers,
            self.config.dropout,
        )

        self.categorical_heads = nn.ModuleList(
            CategoricalDecoder(
                self.config.dimension,
                len(schema["cat_vocab"][path]),
                self.config.dropout,
            )
            for path in self.categorical_paths
        )
        self.total_classes = sum(len(schema["cat_vocab"][path]) for path in self.categorical_paths)
        self.root_paths = tuple(
            path
            for path in ("meta.upper", "meta.wb", "meta.bottom")
            if path in self.categorical_paths
        )
        self.root_indices = tuple(self.categorical_paths.index(path) for path in self.root_paths)
        root_classes = sum(
            len(schema["cat_vocab"][self.categorical_paths[index]])
            for index in self.root_indices
        )
        self.root_topology_projection = (
            nn.Sequential(
                nn.LayerNorm(root_classes),
                nn.Linear(root_classes, self.config.dimension),
                nn.GELU(),
                nn.Linear(self.config.dimension, self.config.dimension),
            )
            if self.config.hierarchical_categoricals and root_classes
            else None
        )
        self.topology_projection = nn.Sequential(
            nn.LayerNorm(self.total_classes),
            nn.Linear(self.total_classes, self.config.dimension),
            nn.GELU(),
            nn.Linear(self.config.dimension, self.config.dimension),
        )

        grouped: OrderedDict[str, list[int]] = OrderedDict(
            (group, []) for group in SEMANTIC_GROUPS
        )
        for index, path in enumerate(self.numeric_paths):
            grouped[parameter_group(path)].append(index)
        self.numeric_layout = {group: indices for group, indices in grouped.items() if indices}
        self.numeric_heads = nn.ModuleDict(
            {
                group: NumericGroupDecoder(
                    self.config.dimension, len(indices), self.config.dropout
                )
                for group, indices in self.numeric_layout.items()
            }
        )

    @staticmethod
    def _scheduled_probability(
        logits: Tensor,
        target: Tensor | None,
        teacher_force: float,
    ) -> Tensor:
        probability = logits.softmax(dim=-1)
        if target is None or teacher_force <= 0.0:
            return probability
        active = target.ge(0)
        truth = F.one_hot(target.clamp_min(0), logits.shape[-1]).to(probability.dtype)
        use_truth = torch.rand(
            (logits.shape[0], 1), device=logits.device
        ).lt(teacher_force)
        return torch.where(use_truth & active[:, None], truth, probability)

    def _categoricals(
        self,
        parts: Tensor,
        target_categories: Tensor | None,
        teacher_force: float,
    ) -> tuple[list[Tensor], Tensor]:
        global_feature = parts[:, self.group_index["global"]]
        logits: list[Tensor | None] = [None] * len(self.categorical_paths)
        activities: list[Tensor | None] = [None] * len(self.categorical_paths)

        def decode(index: int, conditioned_global: Tensor) -> None:
            path = self.categorical_paths[index]
            head = self.categorical_heads[index]
            local = parts[:, self.group_index[parameter_group(path)]]
            field_logits, field_activity = head(
                torch.cat((local, conditioned_global), dim=-1)
            )
            logits[index] = field_logits
            activities[index] = field_activity

        root_set = set(self.root_indices)
        for index in self.root_indices:
            decode(index, global_feature)

        conditioned_global = global_feature
        if self.root_topology_projection is not None:
            root_probabilities = []
            for index in self.root_indices:
                field_logits = logits[index]
                assert field_logits is not None
                target = (
                    None
                    if target_categories is None
                    else target_categories[:, index]
                )
                root_probabilities.append(
                    self._scheduled_probability(
                        field_logits, target, teacher_force
                    ).detach()
                )
            root_context = self.root_topology_projection(
                torch.cat(root_probabilities, dim=-1)
            )
            conditioned_global = global_feature + root_context

        for index in range(len(self.categorical_paths)):
            if index not in root_set:
                decode(index, conditioned_global)
        assert all(value is not None for value in logits)
        assert all(value is not None for value in activities)
        return (
            [value for value in logits if value is not None],
            torch.stack(
                [value for value in activities if value is not None], dim=-1
            ),
        )

    def _topology_vector(
        self,
        logits: list[Tensor],
        activity_logits: Tensor,
        target_categories: Tensor | None,
        teacher_force: float,
    ) -> Tensor:
        """Create a stable topology code for numeric decoding.

        Inactive fields contribute zero instead of an arbitrary probability
        simplex. Scheduled sampling uses either a complete prediction or the
        target for each field, rather than a train-only probability blend.
        The code is detached so regression cannot distort categorical logits.
        """
        probabilities = []
        for index, field_logits in enumerate(logits):
            predicted = field_logits.softmax(dim=-1)
            activity = activity_logits[:, index : index + 1].sigmoid()
            if target_categories is not None and teacher_force > 0.0:
                target = target_categories[:, index]
                active = target.ge(0)
                safe_target = target.clamp_min(0)
                truth = F.one_hot(safe_target, field_logits.shape[-1]).to(predicted.dtype)
                truth_activity = active[:, None].to(activity.dtype)
                use_truth = torch.rand_like(activity).lt(teacher_force)
                predicted = torch.where(use_truth & active[:, None], truth, predicted)
                activity = torch.where(use_truth, truth_activity, activity)
            probabilities.append(predicted.detach() * activity.detach())
        return torch.cat(probabilities, dim=-1)

    def forward(
        self,
        image: Tensor,
        target_categories: Tensor | None = None,
        teacher_force: float = 0.0,
    ) -> dict[str, Any]:
        if not 0.0 <= teacher_force <= 1.0:
            raise ValueError("teacher_force must be in [0, 1]")
        if self.config.encoder_kind == "scratch":
            features = self.encoder(image)
            assert self.fusion is not None
            fused = self.fusion(features)
        else:
            fused = self.encoder(image)
        parts = self.part_decoder(fused)
        categorical_logits, categorical_activity = self._categoricals(
            parts, target_categories, teacher_force
        )
        topology_vector = self._topology_vector(
            categorical_logits,
            categorical_activity,
            target_categories,
            teacher_force,
        )
        topology = self.topology_projection(topology_vector)

        batch = image.shape[0]
        numeric_count = len(self.numeric_paths)
        # The input remains float32 while autocast may make head outputs
        # float16/bfloat16. Initialize from the first head output so indexed
        # assignment always uses a matching dtype under AMP.
        numeric_mean = None
        numeric_log_scale = None
        numeric_activity = None
        global_feature = parts[:, self.group_index["global"]]
        for group, indices in self.numeric_layout.items():
            local = parts[:, self.group_index[group]]
            feature = torch.cat((local, global_feature, topology), dim=-1)
            mean, log_scale, activity = self.numeric_heads[group](feature)
            if numeric_mean is None:
                numeric_mean = mean.new_zeros((batch, numeric_count))
                numeric_log_scale = log_scale.new_zeros((batch, numeric_count))
                numeric_activity = activity.new_zeros((batch, numeric_count))
            numeric_mean[:, indices] = mean
            numeric_log_scale[:, indices] = log_scale
            numeric_activity[:, indices] = activity

        assert numeric_mean is not None
        assert numeric_log_scale is not None
        assert numeric_activity is not None

        return {
            "numeric_mean": numeric_mean,
            "numeric_log_scale": numeric_log_scale,
            "numeric_activity": numeric_activity,
            "categorical_logits": categorical_logits,
            "categorical_activity": categorical_activity,
            "part_features": parts,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

