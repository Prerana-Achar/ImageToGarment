"""Hierarchical, route-constrained ensemble for image-to-GarmentCode."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from garment_tree_model import (
    CategoricalDecoder,
    NumericGroupDecoder,
    PartQueryDecoder,
    SEMANTIC_GROUPS,
    parameter_group,
)

ROOT_PATHS = ("meta.upper", "meta.wb", "meta.bottom")
UPPER_GROUPS = frozenset(("shirt", "collar", "sleeve", "left"))
SKIRT_GROUPS = frozenset(
    ("skirt", "flare-skirt", "godet-skirt", "pencil-skirt", "levels-skirt")
)


def route_marginal_logits(
    route_logits: Tensor,
    routes: Tensor,
    class_counts: Sequence[int],
) -> list[Tensor]:
    """Marginalize a constrained joint route distribution into root logits."""
    if route_logits.ndim != 2 or routes.ndim != 2:
        raise ValueError("Route logits and route tuples must both be rank two")
    if route_logits.shape[1] != routes.shape[0]:
        raise ValueError("Route-logit width does not match the route table")
    if routes.shape[1] != len(class_counts):
        raise ValueError("Route tuples and root class counts disagree")

    route_probability = route_logits.softmax(dim=-1)
    marginals: list[Tensor] = []
    for position, classes in enumerate(class_counts):
        marginal = route_probability.new_zeros(
            (route_probability.shape[0], classes)
        )
        selected = routes[:, position].unsqueeze(0).expand(
            route_probability.shape[0], -1
        )
        marginal.scatter_add_(1, selected, route_probability)
        marginals.append(marginal.clamp_min(1e-8).log())
    return marginals


def path_region(path: str) -> str:
    """Return the structural body region controlling a GarmentCode path."""
    group = parameter_group(path)
    if group in UPPER_GROUPS:
        return "upper"
    if group == "waistband":
        return "waist"
    if group in SKIRT_GROUPS:
        return "skirt"
    if group == "pants":
        return "pants"
    return "global"


@dataclass(frozen=True)
class RouteConstraints:
    """Semantic valid root tuples and route-conditioned active-path masks."""

    root_paths: tuple[str, ...]
    root_indices: tuple[int, ...]
    valid_root_tuples: tuple[tuple[int, ...], ...]
    observed_root_tuples: tuple[tuple[int, ...], ...]
    numeric_masks: tuple[tuple[bool, ...], ...]
    categorical_masks: tuple[tuple[bool, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "RouteConstraints":
        return cls(
            root_paths=tuple(values["root_paths"]),
            root_indices=tuple(map(int, values["root_indices"])),
            valid_root_tuples=tuple(
                tuple(map(int, route)) for route in values["valid_root_tuples"]
            ),
            observed_root_tuples=tuple(
                tuple(map(int, route))
                for route in values.get(
                    "observed_root_tuples", values["valid_root_tuples"]
                )
            ),
            numeric_masks=tuple(
                tuple(map(bool, mask)) for mask in values["numeric_masks"]
            ),
            categorical_masks=tuple(
                tuple(map(bool, mask)) for mask in values["categorical_masks"]
            ),
        )

    @classmethod
    def from_targets(
        cls,
        schema: dict[str, Any],
        y_cat: Any,
        numeric_mask: Any,
        *,
        support_y_cat: Any | None = None,
        support_numeric_mask: Any | None = None,
    ) -> "RouteConstraints":
        categorical_paths = list(schema["cat_vocab"])
        numeric_paths = [
            *schema.get("cont_slots", {}).keys(),
            *schema.get("const_slots", {}).keys(),
        ]
        missing = [path for path in ROOT_PATHS if path not in categorical_paths]
        if missing:
            raise ValueError(f"Schema is missing required topology roots: {missing}")
        root_indices = tuple(categorical_paths.index(path) for path in ROOT_PATHS)
        categories = torch.as_tensor(y_cat, dtype=torch.long).cpu()
        numeric = torch.as_tensor(numeric_mask, dtype=torch.bool).cpu()
        if categories.ndim != 2 or numeric.ndim != 2:
            raise ValueError("Route target arrays must be rank two")
        if categories.shape[0] != numeric.shape[0]:
            raise ValueError("Categorical and numeric route targets disagree in length")
        if categories.shape[1] != len(categorical_paths):
            raise ValueError(
                "Categorical target width does not match schema.cat_vocab"
            )
        if numeric.shape[1] != len(numeric_paths):
            raise ValueError(
                "Numeric activity-mask width does not match the schema"
            )
        support_categories = (
            categories
            if support_y_cat is None
            else torch.as_tensor(support_y_cat, dtype=torch.long).cpu()
        )
        support_numeric = (
            numeric
            if support_numeric_mask is None
            else torch.as_tensor(support_numeric_mask, dtype=torch.bool).cpu()
        )
        if support_categories.ndim != 2 or support_numeric.ndim != 2:
            raise ValueError("Route support arrays must be rank two")
        if support_categories.shape[0] != support_numeric.shape[0]:
            raise ValueError("Categorical and numeric route support disagree in length")
        if support_categories.shape[1] != len(categorical_paths):
            raise ValueError(
                "Categorical route support width does not match the schema"
            )
        if support_numeric.shape[1] != len(numeric_paths):
            raise ValueError("Numeric route support width does not match the schema")

        routes: dict[tuple[int, ...], tuple[Tensor, Tensor]] = {}
        for row in range(categories.shape[0]):
            route = tuple(int(categories[row, index]) for index in root_indices)
            if any(value < 0 for value in route):
                raise ValueError(f"Topology roots must always be active, got {route}")
            for position, value in enumerate(route):
                classes = len(schema["cat_vocab"][ROOT_PATHS[position]])
                if value >= classes:
                    raise ValueError(
                        f"Topology root {ROOT_PATHS[position]} has out-of-range "
                        f"class {value}; expected [0, {classes})"
                    )
            cat_mask = categories[row].ge(0)
            if route in routes:
                previous_numeric, previous_cat = routes[route]
                routes[route] = (previous_numeric | numeric[row], previous_cat | cat_mask)
            else:
                routes[route] = (numeric[row].clone(), cat_mask.clone())
        if not routes:
            raise ValueError("No valid training routes were found")
        observed_routes = tuple(sorted(routes))
        empirical = cls(
            root_paths=ROOT_PATHS,
            root_indices=root_indices,
            valid_root_tuples=observed_routes,
            observed_root_tuples=observed_routes,
            numeric_masks=tuple(
                tuple(bool(value) for value in routes[route][0].tolist())
                for route in observed_routes
            ),
            categorical_masks=tuple(
                tuple(bool(value) for value in routes[route][1].tolist())
                for route in observed_routes
            ),
        )
        empirical.assert_exclusive_lower_body(
            schema, numeric_paths, categorical_paths
        )

        # Learn which fields belong to each semantic component independently of
        # the complete tuple. This transfers a component's full field support
        # to unseen valid combinations without enabling unrelated subtypes.
        root_vocabs = [schema["cat_vocab"][path] for path in ROOT_PATHS]
        numeric_regions = [path_region(path) for path in numeric_paths]
        categorical_regions = [path_region(path) for path in categorical_paths]
        global_numeric = support_numeric.any(dim=0) & torch.tensor(
            [region == "global" for region in numeric_regions],
            dtype=torch.bool,
        )
        global_categorical = support_categories.ge(0).any(dim=0) & torch.tensor(
            [region == "global" for region in categorical_regions],
            dtype=torch.bool,
        )
        global_categorical[list(root_indices)] = True
        component_numeric: dict[tuple[int, int], Tensor] = {}
        component_categorical: dict[tuple[int, int], Tensor] = {}
        for root_position, (root_index, vocabulary) in enumerate(
            zip(root_indices, root_vocabs)
        ):
            for value, semantic in enumerate(vocabulary):
                selected_rows = support_categories[:, root_index].eq(value)
                if root_position == 0:
                    controlled_region = "upper"
                elif root_position == 1:
                    controlled_region = "waist"
                else:
                    controlled_region = "pants" if semantic == "Pants" else "skirt"
                numeric_support = (
                    support_numeric[selected_rows].any(dim=0)
                    if bool(selected_rows.any())
                    else torch.zeros(
                        support_numeric.shape[1], dtype=torch.bool
                    )
                )
                categorical_support = (
                    support_categories[selected_rows].ge(0).any(dim=0)
                    if bool(selected_rows.any())
                    else torch.zeros(
                        support_categories.shape[1], dtype=torch.bool
                    )
                )
                component_numeric[(root_position, value)] = numeric_support & torch.tensor(
                    [region == controlled_region for region in numeric_regions],
                    dtype=torch.bool,
                )
                component_categorical[(root_position, value)] = categorical_support & torch.tensor(
                    [region == controlled_region for region in categorical_regions],
                    dtype=torch.bool,
                )

        # The semantic grammar defines legal combinations, not the finite set
        # of complete tuples observed in this split.
        grammar_routes: dict[tuple[int, ...], tuple[Tensor, Tensor]] = {}
        for candidate in product(*(range(len(vocab)) for vocab in root_vocabs)):
            upper, waistband, bottom = (
                root_vocabs[position][candidate[position]]
                for position in range(3)
            )
            if upper is None and bottom is None:
                continue
            if bottom is None and waistband is not None:
                continue

            numeric_route_mask = global_numeric.clone()
            categorical_route_mask = global_categorical.clone()
            for root_position, (value, semantic) in enumerate(
                zip(candidate, (upper, waistband, bottom))
            ):
                if semantic is None:
                    continue
                numeric_route_mask |= component_numeric[(root_position, value)]
                categorical_route_mask |= component_categorical[
                    (root_position, value)
                ]
            grammar_routes[candidate] = (
                numeric_route_mask,
                categorical_route_mask,
            )

        invalid_observed = sorted(set(observed_routes) - set(grammar_routes))
        if invalid_observed:
            raise ValueError(
                "Observed topology roots violate the semantic garment grammar: "
                f"{invalid_observed[:5]}"
            )
        routes = grammar_routes
        ordered_routes = tuple(sorted(routes))
        constraints = cls(
            root_paths=ROOT_PATHS,
            root_indices=root_indices,
            valid_root_tuples=ordered_routes,
            observed_root_tuples=observed_routes,
            numeric_masks=tuple(
                tuple(bool(value) for value in routes[route][0].tolist())
                for route in ordered_routes
            ),
            categorical_masks=tuple(
                tuple(bool(value) for value in routes[route][1].tolist())
                for route in ordered_routes
            ),
        )
        constraints.validate(schema, numeric_paths, categorical_paths)
        return constraints

    def validate(
        self,
        schema: dict[str, Any],
        numeric_paths: Sequence[str] | None = None,
        categorical_paths: Sequence[str] | None = None,
    ) -> None:
        numeric_paths = list(numeric_paths or [
            *schema.get("cont_slots", {}).keys(),
            *schema.get("const_slots", {}).keys(),
        ])
        categorical_paths = list(categorical_paths or schema["cat_vocab"])
        if tuple(self.root_paths) != ROOT_PATHS:
            raise ValueError(
                f"Route root order must be {ROOT_PATHS}, got {self.root_paths}"
            )
        expected_indices = tuple(
            categorical_paths.index(path) for path in ROOT_PATHS
        )
        if tuple(self.root_indices) != expected_indices:
            raise ValueError(
                f"Route root indices {self.root_indices} do not match schema "
                f"indices {expected_indices}"
            )
        route_count = len(self.valid_root_tuples)
        if route_count == 0:
            raise ValueError("Route constraints contain no valid topology")
        if len(set(self.valid_root_tuples)) != route_count:
            raise ValueError("Route constraints contain duplicate topologies")
        if not self.observed_root_tuples:
            raise ValueError("Route constraints contain no observed topology")
        if not set(self.observed_root_tuples).issubset(self.valid_root_tuples):
            raise ValueError("Observed topologies must be a subset of valid topologies")
        if len(self.numeric_masks) != route_count or len(self.categorical_masks) != route_count:
            raise ValueError("Route mask counts do not match the valid-route count")
        root_vocabs = [schema["cat_vocab"][path] for path in ROOT_PATHS]
        for index, route in enumerate(self.valid_root_tuples):
            if len(route) != len(ROOT_PATHS):
                raise ValueError(f"Route {route} has the wrong number of roots")
            for position, value in enumerate(route):
                if value < 0 or value >= len(root_vocabs[position]):
                    raise ValueError(
                        f"Route {route} has an invalid {ROOT_PATHS[position]} class"
                    )
            if len(self.numeric_masks[index]) != len(numeric_paths):
                raise ValueError(
                    f"Route {route} numeric mask has the wrong width"
                )
            if len(self.categorical_masks[index]) != len(categorical_paths):
                raise ValueError(
                    f"Route {route} categorical mask has the wrong width"
                )
            if not all(
                self.categorical_masks[index][root_index]
                for root_index in self.root_indices
            ):
                raise ValueError(f"Route {route} disables a topology root")
        self.assert_exclusive_lower_body(
            schema, numeric_paths, categorical_paths
        )

    def assert_exclusive_lower_body(
        self,
        schema: dict[str, Any],
        numeric_paths: Sequence[str] | None = None,
        categorical_paths: Sequence[str] | None = None,
    ) -> None:
        numeric_paths = list(numeric_paths or [
            *schema.get("cont_slots", {}).keys(),
            *schema.get("const_slots", {}).keys(),
        ])
        categorical_paths = list(categorical_paths or schema["cat_vocab"])
        upper_position = self.root_paths.index("meta.upper")
        waistband_position = self.root_paths.index("meta.wb")
        bottom_position = self.root_paths.index("meta.bottom")
        upper_vocab = schema["cat_vocab"]["meta.upper"]
        waistband_vocab = schema["cat_vocab"]["meta.wb"]
        bottom_vocab = schema["cat_vocab"]["meta.bottom"]
        for route_index, route in enumerate(self.valid_root_tuples):
            bottom = bottom_vocab[route[bottom_position]]
            active_paths = [
                path
                for path, allowed in zip(numeric_paths, self.numeric_masks[route_index])
                if allowed
            ] + [
                path
                for path, allowed in zip(
                    categorical_paths, self.categorical_masks[route_index]
                )
                if allowed
            ]
            regions = {path_region(path) for path in active_paths}
            upper = upper_vocab[route[upper_position]]
            waistband = waistband_vocab[route[waistband_position]]
            if bottom == "Pants" and "skirt" in regions:
                raise ValueError(f"Pants route {route} activates skirt fields")
            if bottom not in (None, "Pants") and "pants" in regions:
                raise ValueError(f"Skirt route {route} activates pants fields")
            if bottom is None and ({"pants", "skirt"} & regions):
                raise ValueError(f"No-bottom route {route} activates lower-body fields")
            if upper is None and "upper" in regions:
                raise ValueError(f"No-upper route {route} activates upper fields")
            if waistband is None and "waist" in regions:
                raise ValueError(f"No-waistband route {route} activates waistband fields")
            if upper is None and bottom is None:
                raise ValueError(f"Empty-garment route {route} is not valid")
            if bottom is None and waistband is not None:
                raise ValueError(
                    f"Waistband-without-bottom route {route} is not valid"
                )

    def tensors(self, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        return (
            torch.tensor(self.valid_root_tuples, dtype=torch.long, device=device),
            torch.tensor(self.numeric_masks, dtype=torch.bool, device=device),
            torch.tensor(self.categorical_masks, dtype=torch.bool, device=device),
        )

    def select(self, root_logits: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        """Constrained MAP selection over root tuples observed in training."""
        if len(root_logits) != len(self.root_paths):
            raise ValueError("Router produced the wrong number of topology roots")
        routes, _, _ = self.tensors(root_logits[0].device)
        joint = root_logits[0].new_zeros((root_logits[0].shape[0], routes.shape[0]))
        for root_position, logits in enumerate(root_logits):
            selected = routes[:, root_position]
            joint = joint + logits.log_softmax(dim=-1)[:, selected]
        route_id = joint.argmax(dim=-1)
        return route_id, routes[route_id]


@dataclass(frozen=True)
class GarmentEnsembleConfig:
    dimension: int = 64
    query_layers: int = 1
    attention_heads: int = 4
    dropout: float = 0.25
    member_count: int = 5
    backbone_name: str = "dinov2_vits14"
    backbone_repo: str = "DINOv2"
    freeze_backbone: bool = True
    class_token_scale: float = 0.0
    joint_route_router: bool = True
    separate_route_branch: bool = True
    route_dimension: int = 32
    route_class_token_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.dimension < 1:
            raise ValueError("Ensemble dimension must be positive")
        if self.query_layers < 1:
            raise ValueError("Ensemble query_layers must be positive")
        if self.attention_heads < 1:
            raise ValueError("Ensemble attention_heads must be positive")
        if self.dimension % self.attention_heads:
            raise ValueError(
                "Ensemble dimension must be divisible by attention_heads"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Ensemble dropout must be in [0, 1)")
        if self.member_count < 1:
            raise ValueError("Ensemble member_count must be positive")
        if self.route_dimension < 1:
            raise ValueError("Ensemble route_dimension must be positive")
        if not (
            float("-inf") < self.route_class_token_scale < float("inf")
        ) or self.route_class_token_scale < 0.0:
            raise ValueError(
                "Ensemble route_class_token_scale must be finite and non-negative"
            )
        if not self.backbone_name:
            raise ValueError("Ensemble backbone_name cannot be empty")
        if not self.backbone_repo:
            raise ValueError("Ensemble backbone_repo cannot be empty")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GarmentEnsembleConfig":
        values = dict(values)
        # Preserve the exact graph used by older checkpoints.
        values.setdefault("class_token_scale", 1.0)
        values.setdefault("joint_route_router", False)
        values.setdefault("separate_route_branch", False)
        values.setdefault("route_dimension", int(values.get("dimension", 64)))
        values.setdefault("route_class_token_scale", 0.0)
        return cls(**values)


class FrozenPatchEncoder(nn.Module):
    """Return normalized raw patch and class tokens from one trunk pass."""

    def __init__(self, name: str, repository: str, frozen: bool = True) -> None:
        super().__init__()
        repo = Path(repository).expanduser()
        if not repo.is_absolute():
            repo = Path(__file__).resolve().parent / repo
        if not (repo / "hubconf.py").is_file():
            raise FileNotFoundError(f"Local pretrained encoder repository is missing: {repo}")
        self.backbone = torch.hub.load(str(repo), name, source="local", pretrained=True)
        self.embed_dim = int(getattr(self.backbone, "embed_dim"))
        self.frozen = bool(frozen)
        if self.frozen:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> "FrozenPatchEncoder":
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        if self.frozen:
            self.backbone.eval()
            with torch.no_grad():
                encoded = self.backbone.forward_features(image)
        else:
            encoded = self.backbone.forward_features(image)
        return encoded["x_norm_patchtokens"], encoded["x_norm_clstoken"]


class RootRouter(nn.Module):
    """Small region-aware MLP that predicts one GarmentCode topology root."""

    def __init__(self, dimension: int, classes: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dimension * 2),
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, classes),
        )

    def forward(self, global_feature: Tensor, region_feature: Tensor) -> Tensor:
        return self.network(torch.cat((global_feature, region_feature), dim=-1))
class GarmentExpertHead(nn.Module):
    """Independent region router, part queries, and GarmentCode parameter heads."""

    def __init__(
        self,
        schema: dict[str, Any],
        config: GarmentEnsembleConfig,
        embed_dim: int,
        valid_root_tuples: Sequence[Sequence[int]],
        observed_root_tuples: Sequence[Sequence[int]],
    ) -> None:
        super().__init__()
        self.schema = schema
        self.config = config
        self.numeric_paths = [
            *schema.get("cont_slots", {}).keys(),
            *schema.get("const_slots", {}).keys(),
        ]
        self.categorical_paths = list(schema["cat_vocab"])
        self.group_index = {name: index for index, name in enumerate(SEMANTIC_GROUPS)}
        self.root_indices = tuple(
            self.categorical_paths.index(path) for path in ROOT_PATHS
        )
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, config.dimension),
            nn.GELU(),
            nn.LayerNorm(config.dimension),
        )
        self.class_projection = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, config.dimension)
        )
        if config.class_token_scale == 0.0:
            self.class_projection.requires_grad_(False)
        self.part_decoder = PartQueryDecoder(
            config.dimension,
            config.attention_heads,
            config.query_layers,
            config.dropout,
        )
        route_dimension = (
            config.route_dimension
            if config.separate_route_branch
            else config.dimension
        )
        self.route_patch_projection = (
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, route_dimension),
                nn.GELU(),
                nn.LayerNorm(route_dimension),
            )
            if config.separate_route_branch
            else None
        )
        self.route_class_projection = (
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, route_dimension),
                nn.GELU(),
                nn.LayerNorm(route_dimension),
            )
            if config.separate_route_branch
            and config.route_class_token_scale > 0.0
            else None
        )
        self.root_routers = nn.ModuleList(
            RootRouter(
                route_dimension,
                len(schema["cat_vocab"][path]),
                config.dropout,
            )
            for path in ROOT_PATHS
        )
        self.register_buffer(
            "valid_routes",
            torch.tensor(valid_root_tuples, dtype=torch.long),
            persistent=False,
        )
        observed_routes = {
            tuple(int(value) for value in route)
            for route in observed_root_tuples
        }
        self.register_buffer(
            "observed_route_mask",
            torch.tensor(
                [tuple(map(int, route)) in observed_routes for route in valid_root_tuples],
                dtype=torch.bool,
            ),
            persistent=False,
        )
        self.route_router = (
            nn.Sequential(
                nn.LayerNorm(route_dimension * 4),
                nn.Linear(route_dimension * 4, route_dimension * 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(route_dimension * 2, len(valid_root_tuples)),
            )
            if config.joint_route_router
            else None
        )
        if self.route_router is not None:
            nn.init.zeros_(self.route_router[-1].weight)
            nn.init.zeros_(self.route_router[-1].bias)
        self.categorical_heads = nn.ModuleDict(
            {
                str(index): CategoricalDecoder(
                    config.dimension,
                    len(schema["cat_vocab"][path]),
                    config.dropout,
                )
                for index, path in enumerate(self.categorical_paths)
                if index not in self.root_indices
            }
        )
        root_classes = sum(
            len(schema["cat_vocab"][path]) for path in ROOT_PATHS
        )
        self.root_context = nn.Sequential(
            nn.LayerNorm(root_classes),
            nn.Linear(root_classes, config.dimension),
            nn.GELU(),
            nn.Linear(config.dimension, config.dimension),
        )
        self.total_classes = sum(
            len(schema["cat_vocab"][path]) for path in self.categorical_paths
        )
        self.topology_projection = nn.Sequential(
            nn.LayerNorm(self.total_classes),
            nn.Linear(self.total_classes, config.dimension),
            nn.GELU(),
            nn.Linear(config.dimension, config.dimension),
        )
        grouped: dict[str, list[int]] = {group: [] for group in SEMANTIC_GROUPS}
        for index, path in enumerate(self.numeric_paths):
            grouped[parameter_group(path)].append(index)
        self.numeric_layout = {
            group: indices for group, indices in grouped.items() if indices
        }
        self.numeric_heads = nn.ModuleDict(
            {
                group: NumericGroupDecoder(
                    config.dimension, len(indices), config.dropout
                )
                for group, indices in self.numeric_layout.items()
            }
        )

    def encode_parts(self, patches: Tensor, class_token: Tensor) -> Tensor:
        projected = self.patch_projection(patches)
        if self.config.class_token_scale != 0.0:
            projected = projected + self.config.class_token_scale * self.class_projection(
                class_token
            ).unsqueeze(1)
        side = int(round(projected.shape[1] ** 0.5))
        if side * side != projected.shape[1]:
            raise RuntimeError("The frozen encoder returned a non-square patch grid")
        feature = projected.transpose(1, 2).reshape(
            projected.shape[0], projected.shape[-1], side, side
        )
        return self.part_decoder(feature)

    @staticmethod
    def _probability(
        logits: Tensor,
        target: Tensor | None,
        ratio: float,
        use_truth: Tensor | None = None,
    ) -> Tensor:
        probability = logits.softmax(dim=-1)
        if target is None or ratio <= 0.0:
            return probability
        active = target.ge(0)
        truth = F.one_hot(target.clamp_min(0), logits.shape[-1]).to(probability.dtype)
        if use_truth is None:
            use_truth = torch.rand(
                (logits.shape[0], 1), device=logits.device
            ).lt(ratio)
        return torch.where(use_truth & active[:, None], truth, probability)

    def _router_features(self, parts: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        global_feature = parts[:, self.group_index["global"]]
        upper = torch.stack(
            [parts[:, self.group_index[group]] for group in sorted(UPPER_GROUPS)],
            dim=1,
        ).mean(dim=1)
        waist = parts[:, self.group_index["waistband"]]
        lower_groups = [*sorted(SKIRT_GROUPS), "pants"]
        lower = torch.stack(
            [parts[:, self.group_index[group]] for group in lower_groups], dim=1
        ).mean(dim=1)
        return global_feature, upper, waist, lower
    def _spatial_route_features(
        self, patches: Tensor, class_token: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.route_patch_projection is None:
            raise RuntimeError("The isolated route branch is disabled")
        projected = self.route_patch_projection(patches)
        side = int(round(projected.shape[1] ** 0.5))
        if side * side != projected.shape[1]:
            raise RuntimeError("The route branch requires a square patch grid")
        grid = projected.reshape(
            projected.shape[0],
            side,
            side,
            projected.shape[-1],
        )

        def vertical_pool(start: float, stop: float) -> Tensor:
            first = min(max(int(round(side * start)), 0), side - 1)
            last = min(max(int(round(side * stop)), first + 1), side)
            return grid[:, first:last].mean(dim=(1, 2))

        global_feature = grid.mean(dim=(1, 2))
        if self.route_class_projection is not None:
            global_feature = global_feature + (
                self.config.route_class_token_scale
                * self.route_class_projection(class_token)
            )
        return (
            global_feature,
            vertical_pool(0.10, 0.62),
            vertical_pool(0.35, 0.72),
            vertical_pool(0.45, 1.00),
        )

    def route_parameters(self) -> list[nn.Parameter]:
        modules = [self.root_routers, self.route_router]
        if self.route_patch_projection is not None:
            modules.append(self.route_patch_projection)
        if self.route_class_projection is not None:
            modules.append(self.route_class_projection)
        return [
            parameter
            for module in modules
            if module is not None
            for parameter in module.parameters()
        ]

    def forward_tokens(
        self,
        patches: Tensor,
        class_token: Tensor,
        target_categories: Tensor | None = None,
        teacher_force: float = 0.0,
    ) -> dict[str, Any]:
        if not 0.0 <= teacher_force <= 1.0:
            raise ValueError("teacher_force must be in [0, 1]")
        parts = self.encode_parts(patches, class_token)
        global_feature, upper_feature, waist_feature, lower_feature = (
            self._router_features(parts)
        )
        if self.route_patch_projection is None:
            route_global, route_upper, route_waist, route_lower = (
                global_feature,
                upper_feature,
                waist_feature,
                lower_feature,
            )
        else:
            route_global, route_upper, route_waist, route_lower = (
                self._spatial_route_features(patches, class_token)
            )
        region_features = (route_upper, route_waist, route_lower)
        logits: list[Tensor | None] = [None] * len(self.categorical_paths)
        activities: list[Tensor | None] = [None] * len(self.categorical_paths)
        root_logits = []
        for index, router, region in zip(
            self.root_indices, self.root_routers, region_features
        ):
            field_logits = router(route_global, region)
            logits[index] = field_logits
            activities[index] = field_logits.new_full((field_logits.shape[0],), 10.0)
            root_logits.append(field_logits)

        factorized_route_logits = root_logits[0].new_zeros(
            (root_logits[0].shape[0], self.valid_routes.shape[0])
        )
        for root_position, field_logits in enumerate(root_logits):
            selected = self.valid_routes[:, root_position]
            factorized_route_logits = (
                factorized_route_logits
                + field_logits.log_softmax(dim=-1)[:, selected]
            )
        if self.route_router is not None:
            raw_route_residual = self.route_router(
                torch.cat(
                    (
                        route_global,
                        route_upper,
                        route_waist,
                        route_lower,
                    ),
                    dim=-1,
                )
            )
            observed = self.observed_route_mask.unsqueeze(0).to(
                raw_route_residual.dtype
            )
            bounded = raw_route_residual.tanh()
            observed_mean = (bounded * observed).sum(dim=-1, keepdim=True)
            observed_mean = observed_mean / observed.sum().clamp_min(1.0)
            route_residual = (bounded - observed_mean) * observed
            route_logits = factorized_route_logits + route_residual
        else:
            route_logits = factorized_route_logits

        root_teacher_mask = None
        if target_categories is not None and teacher_force > 0.0:
            root_teacher_mask = torch.rand(
                (route_logits.shape[0], 1),
                device=route_logits.device,
            ).lt(teacher_force)

        root_probabilities = []
        if self.route_router is None:
            for index, field_logits in zip(self.root_indices, root_logits):
                target = (
                    None
                    if target_categories is None
                    else target_categories[:, index]
                )
                root_probabilities.append(
                    self._probability(
                        field_logits,
                        target,
                        teacher_force,
                        root_teacher_mask,
                    ).detach()
                )
        else:
            route_probability = route_logits.softmax(dim=-1)
            for root_position, index in enumerate(self.root_indices):
                classes = len(
                    self.schema["cat_vocab"][ROOT_PATHS[root_position]]
                )
                marginal = route_probability.new_zeros(
                    (route_probability.shape[0], classes)
                )
                selected = self.valid_routes[:, root_position].unsqueeze(0).expand(
                    route_probability.shape[0], -1
                )
                marginal.scatter_add_(1, selected, route_probability)
                target = (
                    None
                    if target_categories is None
                    else target_categories[:, index]
                )
                root_probabilities.append(
                    self._probability(
                        marginal.clamp_min(1e-8).log(),
                        target,
                        teacher_force,
                        root_teacher_mask,
                    ).detach()
                )
        conditioned_global = global_feature + self.root_context(
            torch.cat(root_probabilities, dim=-1)
        )
        root_set = set(self.root_indices)
        for index, path in enumerate(self.categorical_paths):
            if index in root_set:
                continue
            local = parts[:, self.group_index[parameter_group(path)]]
            field_logits, field_activity = self.categorical_heads[str(index)](
                torch.cat((local, conditioned_global), dim=-1)
            )
            logits[index] = field_logits
            activities[index] = field_activity
        if any(value is None for value in logits + activities):
            raise RuntimeError("An expert failed to decode every categorical field")
        categorical_logits = [value for value in logits if value is not None]
        categorical_activity = torch.stack(
            [value for value in activities if value is not None], dim=-1
        )

        topology_probabilities = []
        root_probability_by_index = {
            index: root_probabilities[position]
            for position, index in enumerate(self.root_indices)
        }
        for index, field_logits in enumerate(categorical_logits):
            if index in root_probability_by_index:
                probability = root_probability_by_index[index]
                activity = probability.new_ones((probability.shape[0], 1))
            else:
                probability = field_logits.softmax(dim=-1)
                activity = categorical_activity[:, index : index + 1].sigmoid()
                if target_categories is not None and teacher_force > 0.0:
                    target = target_categories[:, index]
                    active = target.ge(0)
                    truth = F.one_hot(
                        target.clamp_min(0), field_logits.shape[-1]
                    ).to(probability.dtype)
                    use_truth = torch.rand_like(activity).lt(teacher_force)
                    probability = torch.where(
                        use_truth & active[:, None], truth, probability
                    )
                    activity = torch.where(
                        use_truth, active[:, None].to(activity.dtype), activity
                    )
            topology_probabilities.append(probability.detach() * activity.detach())
        topology = self.topology_projection(torch.cat(topology_probabilities, dim=-1))

        batch = patches.shape[0]
        numeric_count = len(self.numeric_paths)
        numeric_mean = topology.new_zeros((batch, numeric_count))
        numeric_log_scale = topology.new_zeros((batch, numeric_count))
        numeric_activity = topology.new_zeros((batch, numeric_count))
        for group, indices in self.numeric_layout.items():
            local = parts[:, self.group_index[group]]
            feature = torch.cat((local, global_feature, topology), dim=-1)
            mean, log_scale, activity = self.numeric_heads[group](feature)
            numeric_mean[:, indices] = mean
            numeric_log_scale[:, indices] = log_scale
            numeric_activity[:, indices] = activity
        return {
            "numeric_mean": numeric_mean,
            "numeric_log_scale": numeric_log_scale,
            "numeric_activity": numeric_activity,
            "categorical_logits": categorical_logits,
            "categorical_activity": categorical_activity,
            "route_logits": route_logits,
            "part_features": parts,
        }
class GarmentTreeEnsemble(nn.Module):
    """One frozen encoder plus independently trainable route-aware experts."""

    def __init__(
        self,
        schema: dict[str, Any],
        constraints: RouteConstraints,
        config: GarmentEnsembleConfig | None = None,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.constraints = constraints
        self.config = config or GarmentEnsembleConfig()
        self.config.validate()
        self.numeric_paths = [
            *schema.get("cont_slots", {}).keys(),
            *schema.get("const_slots", {}).keys(),
        ]
        self.categorical_paths = list(schema["cat_vocab"])
        self.root_indices = tuple(
            self.categorical_paths.index(path) for path in ROOT_PATHS
        )
        constraints.validate(
            schema, self.numeric_paths, self.categorical_paths
        )
        self.encoder = FrozenPatchEncoder(
            self.config.backbone_name,
            self.config.backbone_repo,
            self.config.freeze_backbone,
        )
        self.members = nn.ModuleList(
            GarmentExpertHead(
                schema,
                self.config,
                self.encoder.embed_dim,
                constraints.valid_root_tuples,
                constraints.observed_root_tuples,
            )
            for _ in range(self.config.member_count)
        )
        count = self.config.member_count
        self.stacker_numeric_logits = nn.Parameter(
            torch.zeros(count, len(self.numeric_paths))
        )
        self.stacker_numeric_activity_logits = nn.Parameter(
            torch.zeros(count, len(self.numeric_paths))
        )
        self.stacker_categorical_logits = nn.Parameter(
            torch.zeros(count, len(self.categorical_paths))
        )
        self.stacker_categorical_activity_logits = nn.Parameter(
            torch.zeros(count, len(self.categorical_paths))
        )

    def encode(self, image: Tensor) -> tuple[Tensor, Tensor]:
        return self.encoder(image)

    def forward_member_tokens(
        self,
        member_index: int,
        patches: Tensor,
        class_token: Tensor,
        target_categories: Tensor | None = None,
        teacher_force: float = 0.0,
    ) -> dict[str, Any]:
        return self.members[member_index].forward_tokens(
            patches, class_token, target_categories, teacher_force
        )

    def aggregate(self, outputs: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if len(outputs) != self.config.member_count:
            raise ValueError("One prediction is required from every ensemble member")
        numeric = torch.stack([item["numeric_mean"] for item in outputs], dim=0)
        scales = torch.stack(
            [item["numeric_log_scale"].exp() for item in outputs], dim=0
        )
        numeric_weights = self.stacker_numeric_logits.softmax(dim=0)[:, None, :]
        numeric_mean = (numeric_weights * numeric).sum(dim=0)
        second_moment = (
            numeric_weights * (scales.square() + numeric.square())
        ).sum(dim=0)
        variance = (second_moment - numeric_mean.square()).clamp_min(1e-8)
        numeric_log_scale = variance.sqrt().log().clamp(-4.0, 2.0)

        numeric_activities = torch.stack(
            [item["numeric_activity"] for item in outputs], dim=0
        )
        numeric_activity_weights = (
            self.stacker_numeric_activity_logits.softmax(dim=0)[:, None, :]
        )
        raw_numeric_activity = (
            numeric_activity_weights * numeric_activities
        ).sum(dim=0)

        categorical_logits = []
        for index in range(len(self.categorical_paths)):
            field = torch.stack(
                [item["categorical_logits"][index] for item in outputs], dim=0
            )
            weight = self.stacker_categorical_logits[:, index].softmax(dim=0)
            categorical_logits.append(
                (weight[:, None, None] * field).sum(dim=0)
            )
        categorical_activities = torch.stack(
            [item["categorical_activity"] for item in outputs], dim=0
        )
        categorical_activity_weights = (
            self.stacker_categorical_activity_logits.softmax(dim=0)[:, None, :]
        )
        raw_categorical_activity = (
            categorical_activity_weights * categorical_activities
        ).sum(dim=0)

        route_logits = torch.stack(
            [item["route_logits"] for item in outputs], dim=0
        ).mean(dim=0)
        routes, numeric_masks, categorical_masks = self.constraints.tensors(
            numeric_mean.device
        )
        root_logits = route_marginal_logits(
            route_logits, routes,
            [len(self.schema["cat_vocab"][path]) for path in ROOT_PATHS],
        )
        for root_position, field_index in enumerate(self.root_indices):
            categorical_logits[field_index] = root_logits[root_position]
        route_id = route_logits.argmax(dim=-1)
        root_selection = routes[route_id]
        route_numeric_mask = numeric_masks[route_id]
        route_categorical_mask = categorical_masks[route_id]
        numeric_activity = raw_numeric_activity.masked_fill(~route_numeric_mask, -30.0)
        categorical_activity = raw_categorical_activity.masked_fill(
            ~route_categorical_mask, -30.0
        )
        return {
            "numeric_mean": numeric_mean,
            "numeric_log_scale": numeric_log_scale,
            "numeric_activity": numeric_activity,
            "categorical_logits": categorical_logits,
            "categorical_activity": categorical_activity,
            "route_logits": route_logits,
            "raw_numeric_activity": raw_numeric_activity,
            "raw_categorical_activity": raw_categorical_activity,
            "route_id": route_id,
            "root_selection": root_selection,
            "route_numeric_mask": route_numeric_mask,
            "route_categorical_mask": route_categorical_mask,
            "member_outputs": list(outputs),
            "part_features": (
                torch.stack([item["part_features"] for item in outputs], dim=0).mean(dim=0)
                if all("part_features" in item for item in outputs)
                else numeric_mean.new_zeros((numeric_mean.shape[0], 1, 1))
            ),
        }
    def forward_tokens(
        self,
        patches: Tensor,
        class_token: Tensor,
        target_categories: Tensor | None = None,
        teacher_force: float = 0.0,
    ) -> dict[str, Any]:
        outputs = [
            member.forward_tokens(
                patches, class_token, target_categories, teacher_force
            )
            for member in self.members
        ]
        return self.aggregate(outputs)

    def forward(
        self,
        image: Tensor,
        target_categories: Tensor | None = None,
        teacher_force: float = 0.0,
    ) -> dict[str, Any]:
        patches, class_token = self.encode(image)
        return self.forward_tokens(
            patches, class_token, target_categories, teacher_force
        )

    def reset_member(self, index: int) -> None:
        """Replace one expert for an independent bootstrap/cross-fit run."""
        device = next(self.parameters()).device
        self.members[index] = GarmentExpertHead(
            self.schema,
            self.config,
            self.encoder.embed_dim,
            self.constraints.valid_root_tuples,
            self.constraints.observed_root_tuples,
        ).to(device)

    def stacker_parameters(self) -> list[nn.Parameter]:
        return [
            self.stacker_numeric_logits,
            self.stacker_numeric_activity_logits,
            self.stacker_categorical_logits,
            self.stacker_categorical_activity_logits,
        ]


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
