from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn
import yaml


DINOV2_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitl14_reg": 1024,
    "dinov2_vitg14": 1536,
    "dinov2_vitg14_reg": 1536,
}

DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
CLASSIFICATION_TYPES = {"bool", "select", "select_null"}
REGRESSION_TYPES = {"float", "int"}

# These legacy targets exist in prepared data but are intentionally omitted from
# new models because the active GarmentCode design contract does not use them.
UNSUPPORTED_GARMENTCODE_PARAMS = frozenset({
    "shirt.openfront",
    "waistband.height",
})


@dataclass(frozen=True)
class GarmentParamSpec:
    path: tuple[str, ...]
    param_type: str
    default: Any
    choices: tuple[Any, ...] = ()
    min_value: float | None = None
    max_value: float | None = None

    @property
    def name(self) -> str:
        return ".".join(self.path)

    @property
    def module_name(self) -> str:
        return "__".join(self.path).replace("-", "_")

    @property
    def is_regression(self) -> bool:
        return self.param_type in REGRESSION_TYPES

    @property
    def is_classification(self) -> bool:
        return self.param_type in CLASSIFICATION_TYPES


class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int] = (256, 128),
        dropout: float = 0.1,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    return tuple(value)


def _flatten_specs(node: dict[str, Any], path: tuple[str, ...] = ()) -> list[GarmentParamSpec]:
    specs: list[GarmentParamSpec] = []
    if "v" in node and "type" in node:
        param_type = node["type"]
        value_range = node.get("range")
        if param_type in CLASSIFICATION_TYPES:
            choices = _as_tuple(value_range)
            if param_type == "select_null" and None not in choices:
                choices = choices + (None,)
            specs.append(
                GarmentParamSpec(
                    path=path,
                    param_type=param_type,
                    default=node.get("v"),
                    choices=choices,
                )
            )
        elif param_type in REGRESSION_TYPES:
            if not isinstance(value_range, list) or len(value_range) != 2:
                raise ValueError(f"Regression parameter {'.'.join(path)} must have [min, max] range")
            specs.append(
                GarmentParamSpec(
                    path=path,
                    param_type=param_type,
                    default=node.get("v"),
                    min_value=float(value_range[0]),
                    max_value=float(value_range[1]),
                )
            )
        else:
            raise ValueError(f"Unsupported parameter type {param_type!r} at {'.'.join(path)}")
        return specs

    for key, child in node.items():
        if isinstance(child, dict):
            specs.extend(_flatten_specs(child, path + (key,)))
    return specs


def load_garmentcode_schema(
    schema_path: str | Path | None = None,
    exclude_unsupported_params: bool = True,
) -> tuple[dict[str, Any], list[GarmentParamSpec]]:
    if schema_path is None:
        schema_path = Path(__file__).resolve().parent / "GarmentCodeRC" / "assets" / "design_params" / "default_new.yaml"
    schema_path = Path(schema_path)
    with open(schema_path, "r") as f:
        schema = yaml.safe_load(f)
    design_schema = schema["design"]
    specs = _flatten_specs(design_schema)
    if exclude_unsupported_params:
        specs = [
            spec for spec in specs
            if spec.name not in UNSUPPORTED_GARMENTCODE_PARAMS
        ]
    return design_schema, specs


def _set_nested(root: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = root
    for key in path[:-1]:
        node = node[key]
    node[path[-1]]["v"] = value


def _tensor_to_scalar(value: Tensor) -> Any:
    if value.numel() != 1:
        return value.detach().cpu().tolist()
    return value.detach().cpu().item()


class GarmentCodeDINOv2MLP(nn.Module):
    """Predict every GarmentCode design parameter from an image"""

    def __init__(
        self,
        model_name: str = "dinov2_vitl14",
        head_hidden_dims: Iterable[int] = (256, 128),
        shared_hidden_dim: int | None = None,
        dropout: float = 0.1,
        head_layer_norm: bool = False,
        exclude_unsupported_params: bool = True,
        pretrained: bool = True,
        freeze_encoder: bool = True,
        normalize_images: bool = True,
        schema_path: str | Path | None = None,
        dinov2_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        if model_name not in DINOV2_DIMS:
            valid = ", ".join(sorted(DINOV2_DIMS))
            raise ValueError(f"Unknown DINOv2 model '{model_name}'. Valid options: {valid}")

        repo_dir = Path(dinov2_dir) if dinov2_dir is not None else Path(__file__).resolve().parent / "DINOv2"
        if not repo_dir.exists():
            raise FileNotFoundError(f"DINOv2 directory not found: {repo_dir}")

        self.model_name = model_name
        self.feature_dim = DINOV2_DIMS[model_name]
        self.freeze_encoder_flag = freeze_encoder
        self.normalize_images = normalize_images
        self.exclude_unsupported_params = exclude_unsupported_params
        self.design_schema, self.param_specs = load_garmentcode_schema(
            schema_path,
            exclude_unsupported_params=exclude_unsupported_params,
        )

        self.register_buffer("image_mean", torch.tensor(DINO_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(DINO_STD).view(1, 3, 1, 1), persistent=False)

        self.encoder = torch.hub.load(str(repo_dir), model_name, source="local", pretrained=pretrained)
        if freeze_encoder:
            self.freeze_encoder()

        shared_dim = shared_hidden_dim or self.feature_dim
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        self.shared = nn.Sequential(
            nn.Linear(self.feature_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(shared_dim),
        )

        self.heads = nn.ModuleDict()
        self.spec_by_module: dict[str, GarmentParamSpec] = {}
        for spec in self.param_specs:
            output_dim = 1 if spec.is_regression else len(spec.choices)
            self.heads[spec.module_name] = MLPHead(
                shared_dim,
                output_dim,
                head_hidden_dims,
                dropout,
                layer_norm=head_layer_norm,
            )
            self.spec_by_module[spec.module_name] = spec

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder_flag:
            self.encoder.eval()
        return self

    def freeze_encoder(self) -> None:
        self.freeze_encoder_flag = True
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        self.freeze_encoder_flag = False
        for param in self.encoder.parameters():
            param.requires_grad = True

    def normalize(self, images: Tensor) -> Tensor:
        if not self.normalize_images:
            return images
        return (images - self.image_mean.to(images.dtype)) / self.image_std.to(images.dtype)

    def encode(self, images: Tensor) -> Tensor:
        images = self.normalize(images)
        if self.freeze_encoder_flag:
            with torch.no_grad():
                features = self.encoder.forward_features(images)
        else:
            features = self.encoder.forward_features(images)
        return features["x_norm_clstoken"]

    def _regression_output(self, raw: Tensor, spec: GarmentParamSpec) -> dict[str, Tensor]:
        normalized = torch.sigmoid(raw).squeeze(-1)
        min_value = torch.as_tensor(spec.min_value, device=raw.device, dtype=raw.dtype)
        max_value = torch.as_tensor(spec.max_value, device=raw.device, dtype=raw.dtype)
        scaled = min_value + normalized * (max_value - min_value)
        return {"raw": raw.squeeze(-1), "normalized": normalized, "value": scaled}

    def _classification_output(self, logits: Tensor, spec: GarmentParamSpec) -> dict[str, Tensor]:
        return {"logits": logits, "probs": torch.softmax(logits, dim=-1)}

    def forward(self, images: Tensor, return_encoding: bool = False) -> dict[str, Any]:
        encoding = self.encode(images)
        shared = self.shared(self.feature_norm(encoding))

        params: dict[str, dict[str, Tensor]] = {}
        for module_name, head in self.heads.items():
            spec = self.spec_by_module[module_name]
            raw = head(shared)
            if spec.is_regression:
                params[spec.name] = self._regression_output(raw, spec)
            else:
                params[spec.name] = self._classification_output(raw, spec)

        output: dict[str, Any] = {"params": params}
        if return_encoding:
            output["encoding"] = encoding
        return output

    # Use for inference
    def decode(self, outputs: dict[str, Any], batch_index: int = 0) -> dict[str, Any]:
        design = deepcopy(self.design_schema)
        params = outputs["params"]

        for spec in self.param_specs:
            pred = params[spec.name]
            if spec.is_regression:
                value = pred["value"][batch_index]
                decoded = _tensor_to_scalar(value)
                if spec.param_type == "int":
                    decoded = int(round(decoded))
                    decoded = max(int(spec.min_value), min(int(spec.max_value), decoded))
            else:
                class_idx = int(torch.argmax(pred["logits"][batch_index]).detach().cpu().item())
                decoded = spec.choices[class_idx]
            _set_nested(design, spec.path, decoded)

        return {"design": design}

    def decode_batch(self, outputs: dict[str, Any]) -> list[dict[str, Any]]:
        first = next(iter(outputs["params"].values()))
        batch_size = next(iter(first.values())).shape[0]
        return [self.decode(outputs, i) for i in range(batch_size)]

    def parameter_groups(self) -> dict[str, list[str]]:
        groups = {"regression": [], "classification": []}
        for spec in self.param_specs:
            key = "regression" if spec.is_regression else "classification"
            groups[key].append(spec.name)
        return groups


def build_model(**kwargs) -> GarmentCodeDINOv2MLP:
    return GarmentCodeDINOv2MLP(**kwargs)
