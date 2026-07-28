from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from garment_ensemble_model import (
    GarmentEnsembleConfig,
    GarmentExpertHead,
    RouteConstraints,
    path_region,
    route_marginal_logits,
)
from train_garment_tree import (
    ObjectiveWeights,
    macro_binary_f1,
    objective_weights,
    supervised_objective,
)
from train_garment_ensemble import (
    auxiliary_sample_indices,
    cache_view_groups,
    clone_member_state_on_device,
    concatenate_evaluation_objective,
    coverage_bagging_subsets,
    coverage_sample_indices,
    decoded_categorical_prediction,
    evaluation_objective_output,
    loss_options,
    paired_view_indices,
    root_balanced_extra_indices,
    router_objective,
    scheduled_teacher_force,
    should_validate_epoch,
    update_ema_state,
)


def schema():
    return {
        "cont_slots": {},
        "const_slots": {
            "shirt.length": 0,
            "waistband.width": 1,
            "skirt.length": 2,
            "flare-skirt.suns": 3,
            "pants.length": 4,
        },
        "cat_vocab": {
            "meta.upper": ["Shirt", None],
            "meta.wb": ["StraightWB", None],
            "meta.bottom": ["Skirt2", "SkirtCircle", "Pants", None],
            "shirt.strapless": [True, False],
            "pants.cuff.type": ["CuffBand", None],
        },
    }


def targets():
    # roots: upper, waistband, bottom; -1 means inactive detail field
    y_cat = torch.tensor(
        [
            [0, 1, 2, 1, 0],  # top + pants
            [0, 0, 0, 0, -1],  # top + waistband + skirt
            [0, 1, 3, 1, -1],  # top only
            [0, 1, 1, 1, -1],  # top + circle skirt
        ]
    )
    numeric = torch.tensor(
        [
            [1, 0, 0, 0, 1],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0],
        ],
        dtype=torch.bool,
    )
    return y_cat, numeric


def weights_for_targets(y_cat: torch.Tensor, numeric: torch.Tensor):
    return objective_weights(
        SimpleNamespace(
            schema=schema(),
            garment_ids=[f"g{index}" for index in range(y_cat.shape[0])],
            row_of={f"g{index}": index for index in range(y_cat.shape[0])},
            y_cat=y_cat.numpy(),
            mask=torch.zeros((y_cat.shape[0], 0), dtype=torch.bool).numpy(),
            const_mask=numeric.numpy(),
        )
    )


def test_lower_body_routes_are_mutually_exclusive():
    y_cat, numeric = targets()
    constraints = RouteConstraints.from_targets(schema(), y_cat, numeric)
    bottom_vocab = schema()["cat_vocab"]["meta.bottom"]
    numeric_paths = list(schema()["const_slots"])
    for route, mask in zip(constraints.valid_root_tuples, constraints.numeric_masks):
        bottom = bottom_vocab[route[2]]
        regions = {
            path_region(path) for path, active in zip(numeric_paths, mask) if active
        }
        if bottom == "Pants":
            assert "pants" in regions
            assert "skirt" not in regions
        elif bottom is None:
            assert "pants" not in regions
            assert "skirt" not in regions
        else:
            assert "skirt" in regions
            assert "pants" not in regions


def test_pants_contamination_is_rejected():
    y_cat, numeric = targets()
    numeric[0, 2] = True
    with pytest.raises(ValueError, match="Pants route"):
        RouteConstraints.from_targets(schema(), y_cat, numeric)


def test_skirt_subtype_masks_do_not_enable_unrelated_fields():
    y_cat, numeric = targets()
    constraints = RouteConstraints.from_targets(schema(), y_cat, numeric)
    numeric_paths = list(schema()["const_slots"])
    flare_index = numeric_paths.index("flare-skirt.suns")
    plain_skirt = constraints.valid_root_tuples.index((0, 0, 0))
    circle_skirt = constraints.valid_root_tuples.index((0, 1, 1))
    assert not constraints.numeric_masks[plain_skirt][flare_index]
    assert constraints.numeric_masks[circle_skirt][flare_index]


def test_observed_routes_receive_compositional_not_empirical_masks():
    y_cat, numeric = targets()
    y_cat = torch.cat(
        (y_cat, torch.tensor([[1, 1, 0, -1, -1]])),
        dim=0,
    )
    numeric = torch.cat(
        (numeric, torch.tensor([[0, 0, 0, 1, 0]], dtype=torch.bool)),
        dim=0,
    )
    constraints = RouteConstraints.from_targets(schema(), y_cat, numeric)
    route_index = constraints.valid_root_tuples.index((0, 0, 0))
    numeric_paths = list(schema()["const_slots"])
    flare_index = numeric_paths.index("flare-skirt.suns")
    assert constraints.numeric_masks[route_index][flare_index]


def test_auxiliary_support_can_extend_component_masks_without_observing_routes():
    y_cat, numeric = targets()
    primary_numeric = numeric.clone()
    primary_numeric[3, 3] = False
    constraints = RouteConstraints.from_targets(
        schema(),
        y_cat,
        primary_numeric,
        support_y_cat=y_cat,
        support_numeric_mask=numeric,
    )
    circle_route = constraints.valid_root_tuples.index((0, 1, 1))
    flare_index = list(schema()["const_slots"]).index("flare-skirt.suns")
    assert constraints.numeric_masks[circle_route][flare_index]
    assert constraints.observed_root_tuples == tuple(
        sorted(tuple(map(int, row[:3])) for row in y_cat.tolist())
    )


def test_observed_route_that_violates_grammar_is_rejected():
    y_cat, numeric = targets()
    y_cat[0, :3] = torch.tensor([1, 1, 3])
    numeric[0].zero_()
    with pytest.raises(
        ValueError,
        match="No-upper route|No-bottom route|Empty-garment route|semantic garment grammar",
    ):
        RouteConstraints.from_targets(schema(), y_cat, numeric)


def test_constraint_validation_rejects_truncated_masks():
    y_cat, numeric = targets()
    constraints = RouteConstraints.from_targets(schema(), y_cat, numeric)
    values = constraints.to_dict()
    numeric_masks = [
        list(route_mask) for route_mask in values["numeric_masks"]
    ]
    numeric_masks[0] = numeric_masks[0][:-1]
    values["numeric_masks"] = numeric_masks
    malformed = RouteConstraints.from_dict(values)
    with pytest.raises(ValueError, match="numeric mask has the wrong width"):
        malformed.validate(schema())


def test_router_allows_valid_unseen_semantic_tuple():
    y_cat, numeric = targets()
    constraints = RouteConstraints.from_targets(schema(), y_cat, numeric)
    unseen_but_valid = (1, 1, 2)  # lower-only pants without waistband
    assert unseen_but_valid in constraints.valid_root_tuples
    assert (1, 1, 3) not in constraints.valid_root_tuples  # empty garment
    assert (0, 0, 3) not in constraints.valid_root_tuples  # waistband, no bottom

    logits = [
        torch.tensor([[0.0, 4.0]]),
        torch.tensor([[0.0, 3.0]]),
        torch.tensor([[0.0, 0.0, 5.0, 0.0]]),
    ]
    _, selected = constraints.select(logits)
    assert tuple(selected[0].tolist()) == unseen_but_valid

def test_new_config_disables_class_token_but_legacy_checkpoint_keeps_it():
    assert GarmentEnsembleConfig().class_token_scale == 0.0
    legacy = GarmentEnsembleConfig.from_dict(
        {
            "dimension": 128,
            "query_layers": 2,
            "attention_heads": 4,
            "dropout": 0.1,
            "member_count": 5,
            "backbone_name": "dinov2_vits14",
            "backbone_repo": "DINOv2",
            "freeze_backbone": True,
        }
    )
    assert legacy.class_token_scale == 1.0
    assert GarmentEnsembleConfig().joint_route_router is True
    assert GarmentEnsembleConfig().separate_route_branch is True
    assert GarmentEnsembleConfig().route_dimension == 32
    assert legacy.joint_route_router is False
    assert GarmentEnsembleConfig().route_class_token_scale == 1.0
    assert legacy.route_class_token_scale == 0.0
    assert legacy.separate_route_branch is False
    assert legacy.route_dimension == legacy.dimension


def test_config_validation_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="divisible"):
        GarmentEnsembleConfig(dimension=10, attention_heads=4).validate()
    with pytest.raises(ValueError, match="member_count"):
        GarmentEnsembleConfig(member_count=0).validate()


def test_unseen_route_residual_is_fixed_and_observed_common_shift_is_removed():
    y_cat, numeric = targets()
    constraints = RouteConstraints.from_targets(schema(), y_cat, numeric)
    config = GarmentEnsembleConfig(
        dimension=8,
        route_dimension=4,
        attention_heads=2,
        dropout=0.0,
        member_count=1,
    )
    head = GarmentExpertHead(
        schema(),
        config,
        embed_dim=6,
        valid_root_tuples=constraints.valid_root_tuples,
        observed_root_tuples=constraints.observed_root_tuples,
    ).eval()
    patches = torch.randn(2, 4, 6)
    class_token = torch.randn(2, 6)
    baseline = head.forward_tokens(patches, class_token)["route_logits"]
    unseen = next(
        index
        for index, route in enumerate(constraints.valid_root_tuples)
        if route not in set(constraints.observed_root_tuples)
    )
    with torch.no_grad():
        head.route_router[-1].bias[unseen] = 100.0
    unchanged_unseen = head.forward_tokens(patches, class_token)["route_logits"]
    assert torch.allclose(baseline, unchanged_unseen)
    with torch.no_grad():
        head.route_router[-1].bias.zero_()
        head.route_router[-1].bias[head.observed_route_mask] = 100.0
    unchanged_common_shift = head.forward_tokens(patches, class_token)["route_logits"]
    assert torch.allclose(baseline, unchanged_common_shift)


def test_router_objective_does_not_push_unseen_route_logits_down():
    y_cat, numeric = targets()
    constraints = RouteConstraints.from_targets(schema(), y_cat, numeric)
    route_logits = torch.zeros(
        (1, len(constraints.valid_root_tuples)), requires_grad=True
    )
    categorical_logits = [
        torch.zeros((1, len(vocabulary)), requires_grad=True)
        for vocabulary in schema()["cat_vocab"].values()
    ]
    output = {"route_logits": route_logits, "categorical_logits": categorical_logits}
    batch = {"y_cat": y_cat[:1]}
    loss = router_objective(
        output, batch, constraints.root_indices, constraints, label_smoothing=0.0
    )
    loss.backward()
    observed = set(constraints.observed_root_tuples)
    for index, route in enumerate(constraints.valid_root_tuples):
        if route not in observed:
            assert route_logits.grad[0, index] == 0


def test_router_objective_uses_target_class_weights_for_roots():
    y_cat, numeric = targets()
    constraints = RouteConstraints.from_targets(schema(), y_cat, numeric)
    weighted = weights_for_targets(y_cat, numeric)
    route_logits = torch.zeros(
        (1, len(constraints.valid_root_tuples)), requires_grad=True
    )
    categorical_logits = [
        torch.zeros((1, len(vocabulary)), requires_grad=True)
        for vocabulary in schema()["cat_vocab"].values()
    ]
    loss = router_objective(
        {"route_logits": route_logits, "categorical_logits": categorical_logits},
        {"y_cat": y_cat[:1]},
        constraints.root_indices,
        constraints,
        label_smoothing=0.0,
        weights=weighted,
    )
    loss.backward()
    for index in constraints.root_indices:
        assert categorical_logits[index].grad is not None
        assert categorical_logits[index].grad.abs().sum() > 0


def test_unseen_auxiliary_route_trains_factorized_roots_only():
    y_cat, numeric = targets()
    constraints = RouteConstraints.from_targets(schema(), y_cat, numeric)
    unseen = torch.tensor([[1, 1, 2, -1, -1]])
    assert tuple(unseen[0, :3].tolist()) not in set(
        constraints.observed_root_tuples
    )
    route_logits = torch.zeros(
        (1, len(constraints.valid_root_tuples)), requires_grad=True
    )
    categorical_logits = [
        torch.zeros((1, len(vocabulary)), requires_grad=True)
        for vocabulary in schema()["cat_vocab"].values()
    ]
    loss = router_objective(
        {"route_logits": route_logits, "categorical_logits": categorical_logits},
        {"y_cat": unseen},
        constraints.root_indices,
        constraints,
        label_smoothing=0.0,
    )
    loss.backward()
    assert route_logits.grad is None
    for index in constraints.root_indices:
        assert categorical_logits[index].grad is not None
        assert categorical_logits[index].grad.abs().sum() > 0

def test_sampler_uses_equal_distinct_views_per_garment():
    cache = {
        "gid": [
            "garment-a",
            "garment-a",
            "garment-a",
            "garment-b",
            "garment-b",
            "garment-b",
        ]
    }
    selected = coverage_sample_indices(
        cache,
        ["garment-a", "garment-b"],
        samples_per_garment=2,
        seed=42,
        epoch=0,
    )
    assert len(selected) == 4
    assert len(set(selected)) == 4
    assert [cache["gid"][index] for index in selected].count("garment-a") == 2
    assert [cache["gid"][index] for index in selected].count("garment-b") == 2

def test_auxiliary_sampler_spreads_one_view_across_garments_first():
    cache = {
        "gid": [
            "garment-a",
            "garment-a",
            "garment-b",
            "garment-b",
            "garment-c",
            "garment-c",
        ]
    }
    selected = auxiliary_sample_indices(
        cache,
        ["garment-a", "garment-b", "garment-c"],
        count=3,
        seed=42,
        epoch=0,
    )
    assert len(selected) == 3
    assert len({cache["gid"][index] for index in selected}) == 3

def test_root_balanced_sampler_adds_extra_rare_bottom_views():
    cache = {
        "metadata": {"root_indices": [0, 1, 2]},
        "gid": ["rare", "common-a", "common-b", "common-c", "common-d"],
        "y_cat": torch.tensor(
            [
                [0, 0, 1],
                [0, 0, 3],
                [0, 0, 3],
                [0, 0, 3],
                [0, 0, 3],
            ]
        ),
    }
    primary = [0, 1, 2, 3, 4]
    extra = root_balanced_extra_indices(
        cache,
        cache["gid"],
        primary,
        root_position=2,
        strength=1.0,
        seed=42,
        epoch=0,
    )
    assert extra
    assert all(cache["y_cat"][index, 2].item() == 1 for index in extra)


def test_auxiliary_sampler_rejects_empty_allowed_set_instead_of_looping():
    with pytest.raises(ValueError, match="at least one garment"):
        auxiliary_sample_indices(
            {"gid": []}, [], count=1, seed=42, epoch=0
        )


def test_bagging_rejects_zero_members():
    with pytest.raises(ValueError, match="at least one ensemble member"):
        coverage_bagging_subsets(["garment-a"], [], 0, 1.0, 42)


def test_bagging_keeps_core_and_ensemble_union():
    gids = [f"garment-{index}" for index in range(20)]
    core = ["garment-0", "garment-1", "garment-2"]
    subsets = coverage_bagging_subsets(
        gids,
        core,
        members=5,
        fraction=0.75,
        seed=42,
    )
    assert all(set(core) <= set(subset) for subset in subsets)
    assert set().union(*(set(subset) for subset in subsets)) == set(gids)
    assert len({tuple(subset) for subset in subsets}) > 1

def test_pose_pairing_uses_a_different_underlying_view():
    cache = {
        "gid": ["garment-a"] * 6,
        "metadata": {"repeats": 2},
    }
    groups = cache_view_groups(cache)
    selected = [0, 1, 2, 3, 4, 5]
    paired = paired_view_indices(cache, selected, groups, seed=42)
    assert len(paired) == len(selected)
    assert all(left != right for left, right in zip(selected, paired))
    assert all(left % 3 != right % 3 for left, right in zip(selected, paired))


def test_pose_pairing_allows_singleton_as_feature_consistency_fallback():
    cache = {
        "gid": ["garment-a"],
        "metadata": {"repeats": 1},
    }
    groups = cache_view_groups(cache)
    assert paired_view_indices(cache, [0], groups, seed=42) == [0]


def test_ema_state_moves_toward_updated_parameters():
    member = torch.nn.Linear(2, 1, bias=False)
    ema = clone_member_state_on_device(member)
    initial = ema["weight"].clone()
    with torch.no_grad():
        member.weight.add_(1.0)
    update_ema_state(member, ema, decay=0.5)
    assert torch.allclose(ema["weight"], initial + 0.5)


def test_evaluation_objective_uses_raw_activity_before_route_masking():
    output = {
        "numeric_mean": torch.zeros((1, 2)),
        "numeric_log_scale": torch.zeros((1, 2)),
        "numeric_activity": torch.full((1, 2), -30.0),
        "raw_numeric_activity": torch.tensor([[2.0, -2.0]]),
        "categorical_logits": [torch.zeros((1, 3))],
        "categorical_activity": torch.full((1, 1), -30.0),
        "raw_categorical_activity": torch.tensor([[1.5]]),
    }

    objective = evaluation_objective_output(output)

    assert torch.equal(objective["numeric_activity"], output["raw_numeric_activity"])
    assert torch.equal(
        objective["categorical_activity"], output["raw_categorical_activity"]
    )

def test_categorical_accuracy_uses_the_root_tuple_that_decode_emits():
    output = {
        "root_selection": torch.tensor([[0]]),
        # The marginal prefers class 1 even though the most likely legal route
        # selected by the joint router contains class 0.
        "categorical_logits": [torch.tensor([[0.0, 1.0]])],
    }

    predicted = decoded_categorical_prediction(output, 0, {0: 0})

    assert output["categorical_logits"][0].argmax(dim=-1).item() == 1
    assert predicted.item() == 0


def test_macro_activity_f1_counts_false_positive_only_fields():
    score = macro_binary_f1(
        tp=torch.tensor([0.0]),
        fp=torch.tensor([3.0]),
        fn=torch.tensor([0.0]),
    )

    assert score == 0.0




def test_supervised_objective_accepts_boolean_activity_masks():
    output = {
        "numeric_mean": torch.zeros((1, 2)),
        "numeric_log_scale": torch.zeros((1, 2)),
        "numeric_activity": torch.zeros((1, 2)),
        "categorical_logits": [torch.tensor([[2.0, 0.0]])],
        "categorical_activity": torch.zeros((1, 1)),
    }
    batch = {
        "y_cont": torch.zeros((1, 1)),
        "mask": torch.ones((1, 1), dtype=torch.bool),
        "y_const": torch.zeros((1, 1)),
        "const_mask": torch.ones((1, 1), dtype=torch.bool),
        "y_cat": torch.zeros((1, 1), dtype=torch.long),
    }
    weights = ObjectiveWeights(
        numeric_field=torch.ones(2),
        numeric_activity_positive=torch.ones(2),
        categorical_class=[torch.ones(2)],
        categorical_activity_positive=torch.ones(1),
        categorical_ordinal_values=[None],
    )
    options = SimpleNamespace(
        label_smoothing=0.0,
        lambda_category=0.45,
        lambda_ordinal=0.20,
        lambda_activity=0.10,
        lambda_uncertainty=0.02,
    )

    loss, _ = supervised_objective(output, batch, weights, options)

    assert torch.isfinite(loss)

def test_evaluation_objective_is_concatenated_before_loss_normalization():
    outputs = []
    batches = []
    for batch_size in (1, 3):
        outputs.append(
            {
                "numeric_mean": torch.zeros((batch_size, 2)),
                "numeric_log_scale": torch.zeros((batch_size, 2)),
                "numeric_activity": torch.zeros((batch_size, 2)),
                "categorical_logits": [torch.zeros((batch_size, 3))],
                "categorical_activity": torch.zeros((batch_size, 1)),
            }
        )
        batches.append(
            {
                "y_cont": torch.zeros((batch_size, 1)),
                "mask": torch.ones((batch_size, 1), dtype=torch.bool),
                "y_const": torch.zeros((batch_size, 1)),
                "const_mask": torch.ones((batch_size, 1), dtype=torch.bool),
                "y_cat": torch.zeros((batch_size, 1), dtype=torch.long),
            }
        )

    output, batch = concatenate_evaluation_objective(outputs, batches)

    assert output["numeric_mean"].shape == (4, 2)
    assert output["categorical_logits"][0].shape == (4, 3)
    assert batch["y_cat"].shape == (4, 1)

def test_evaluation_loss_does_not_inherit_training_label_smoothing():
    args = type("Args", (), {"label_smoothing": 0.1})()
    assert loss_options(args).label_smoothing == 0.1
    assert loss_options(args, evaluation=True).label_smoothing == 0.0


def test_teacher_forcing_is_capped_and_reaches_zero():
    assert scheduled_teacher_force(0, 10, 0.5) == pytest.approx(0.5)
    assert scheduled_teacher_force(5, 10, 0.5) == pytest.approx(0.25)
    assert scheduled_teacher_force(10, 10, 0.5) == 0.0
    assert scheduled_teacher_force(20, 10, 0.5) == 0.0


def test_validation_is_dense_early_and_periodic_later():
    assert should_validate_epoch(0, 100, 5, 20)
    assert should_validate_epoch(19, 100, 5, 20)
    assert not should_validate_epoch(20, 100, 5, 20)
    assert should_validate_epoch(24, 100, 5, 20)
    assert should_validate_epoch(99, 100, 5, 20)


def test_joint_route_logits_are_marginalized_into_root_probabilities():
    routes = torch.tensor([[0, 0], [0, 1], [1, 1]])
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    marginals = route_marginal_logits(logits, routes, [2, 2])
    route_probability = logits.softmax(dim=-1)
    expected_first = torch.stack(
        (route_probability[0, :2].sum(), route_probability[0, 2])
    ).unsqueeze(0)
    expected_second = torch.stack(
        (route_probability[0, 0], route_probability[0, 1:].sum())
    ).unsqueeze(0)
    assert torch.allclose(
        marginals[0].exp(), expected_first
    )
    assert torch.allclose(
        marginals[1].exp(), expected_second
    )
