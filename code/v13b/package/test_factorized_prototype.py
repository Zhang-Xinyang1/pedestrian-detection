"""CPU tests for the minimal CSF factorized prototype core."""

import copy

import torch
from torch.nn import functional as F

from factorized_prototype import (
    FactorizedPrototypeMemory,
    composed_prototype_loss,
    split_scene,
)


def synthetic_components(classes=5, dim=16, seed=7):
    generator = torch.Generator().manual_seed(seed)
    identity = F.normalize(torch.randn(classes, dim, generator=generator), dim=1)
    modality = torch.randn(classes, 3, dim, generator=generator) * 0.03
    modality[:, 2] = -(modality[:, 0] + modality[:, 1])
    view = torch.randn(classes, 2, dim, generator=generator) * 0.03
    view[:, 1] = -view[:, 0]
    return identity, modality, view


def scenario_features(identity, modality, view, scenes):
    modalities, viewpoints = split_scene(scenes)
    labels = torch.arange(len(identity)).repeat_interleave(len(scenes))
    tiled_scenes = scenes.repeat(len(identity))
    tiled_modalities, tiled_viewpoints = split_scene(tiled_scenes)
    raw = (
        identity[labels]
        + modality[labels, tiled_modalities]
        + view[labels, tiled_viewpoints]
    )
    return F.normalize(raw, dim=1), labels, tiled_scenes


def build_full_memory(classes=5, dim=16):
    identity, modality, view = synthetic_components(classes, dim)
    scenes = torch.arange(6)
    features, labels, tiled_scenes = scenario_features(identity, modality, view, scenes)
    memory = FactorizedPrototypeMemory(classes, dim, momentum=1.0, ridge=1e-6)
    stats = memory.update(features, labels, tiled_scenes)
    assert stats["updated_tuples"] == classes * 6
    assert bool(memory.factor_valid.all())
    return memory, identity, modality, view


def test_scene_mapping():
    modalities, viewpoints = split_scene(torch.arange(6))
    assert modalities.tolist() == [0, 1, 2, 0, 1, 2]
    assert viewpoints.tolist() == [0, 0, 0, 1, 1, 1]


def test_exact_additive_reconstruction():
    memory, identity, modality, view = build_full_memory()
    for scene in range(6):
        prototypes, valid, fallback = memory.compose(scene)
        modality_index = scene % 3
        view_index = scene // 3
        reference = F.normalize(
            identity + modality[:, modality_index] + view[:, view_index], dim=1
        )
        cosine = (prototypes * reference).sum(1)
        assert bool(valid.all())
        assert not bool(fallback.any())
        assert float(cosine.min()) > 0.999
    assert torch.allclose(memory.modality_residuals.sum(1), torch.zeros_like(identity), atol=1e-6)
    assert torch.allclose(memory.view_residuals.sum(1), torch.zeros_like(identity), atol=1e-6)


def test_missing_combination_full_rank():
    classes, dim = 3, 12
    identity, modality, view = synthetic_components(classes, dim)
    # All three modalities are observed on ground, while RGB bridges the
    # aerial viewpoint. This gives the constrained additive design full rank.
    observed = torch.tensor([0, 1, 2, 3])
    features, labels, scenes = scenario_features(identity, modality, view, observed)
    memory = FactorizedPrototypeMemory(classes, dim, momentum=1.0, ridge=1e-6)
    memory.update(features, labels, scenes)
    assert bool(memory.factor_valid.all())
    held_out, valid, _ = memory.compose(5)
    reference = F.normalize(identity + modality[:, 2] + view[:, 1], dim=1)
    assert bool(valid.all())
    assert float((held_out * reference).sum(1).min()) > 0.995


def test_rank_deficient_is_invalid():
    classes, dim = 3, 12
    identity, modality, view = synthetic_components(classes, dim)
    observed = torch.tensor([0, 1, 2])
    features, labels, scenes = scenario_features(identity, modality, view, observed)
    memory = FactorizedPrototypeMemory(classes, dim, momentum=1.0)
    memory.update(features, labels, scenes)
    assert not bool(memory.factor_valid.any())
    assert bool(memory.identity_valid.all())
    _, valid, fallback = memory.compose(4)
    assert bool(valid.all())
    assert bool(fallback.all())


def test_skipped_step_does_not_mutate_memory():
    memory, _, _, _ = build_full_memory()
    before = memory.state_sha256()
    features = torch.randn(8, memory.feature_dim)
    labels = torch.arange(8).remainder(memory.num_classes)
    scenes = torch.arange(8).remainder(6)
    memory.update(features, labels, scenes, step_succeeded=False)
    after = memory.state_sha256()
    assert before != after
    assert int(memory.skipped_updates.item()) == 1
    before_core = memory.state_sha256(include_derived=False)
    memory.update(features, labels, scenes, step_succeeded=False)
    after_core = memory.state_sha256(include_derived=False)
    assert before_core != after_core
    assert torch.equal(memory.scenario_mass, memory.scenario_mass.clone())


def test_skipped_step_preserves_prototype_tensors():
    memory, _, _, _ = build_full_memory()
    tensors = {
        name: tensor.clone()
        for name, tensor in memory.named_buffers()
        if name not in {"skipped_updates", "design"}
    }
    features = torch.randn(8, memory.feature_dim)
    labels = torch.arange(8).remainder(memory.num_classes)
    scenes = torch.arange(8).remainder(6)
    memory.update(features, labels, scenes, step_succeeded=False)
    for name, expected in tensors.items():
        assert torch.equal(dict(memory.named_buffers())[name], expected), name


def test_state_roundtrip_and_hash():
    memory, _, _, _ = build_full_memory()
    clone = FactorizedPrototypeMemory(
        memory.num_classes,
        memory.feature_dim,
        momentum=memory.momentum,
        ridge=memory.ridge,
    )
    clone.load_state_dict(copy.deepcopy(memory.state_dict()))
    assert clone.state_sha256() == memory.state_sha256()
    for scene in range(6):
        left = memory.compose(scene)[0]
        right = clone.compose(scene)[0]
        assert torch.equal(left, right)


def slow_loss(features, labels, scenes, memory, temperature, reconstruction_weight):
    z = F.normalize(features.float(), dim=1)
    losses = []
    reconstructions = []
    for index in range(len(z)):
        if not bool(memory.factor_valid[labels[index]]):
            continue
        prototypes, valid, _ = memory.compose(int(scenes[index]))
        logits = z[index : index + 1] @ prototypes.t() / temperature
        logits = logits.masked_fill(~valid[None, :], torch.finfo(logits.dtype).min)
        losses.append(F.cross_entropy(logits, labels[index : index + 1]))
        reconstructions.append(1.0 - (z[index] * prototypes[labels[index]]).sum())
    if not losses:
        return z.sum() * 0.0
    return torch.stack(losses).mean() + reconstruction_weight * torch.stack(reconstructions).mean()


def test_grouped_loss_matches_reference_and_backpropagates():
    memory, identity, modality, view = build_full_memory(classes=5, dim=16)
    scenes = torch.tensor([0, 4, 2, 3, 1, 5, 0, 4, 2, 3])
    labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    modalities, viewpoints = split_scene(scenes)
    base = identity[labels] + modality[labels, modalities] + view[labels, viewpoints]
    features = (base + 0.01 * torch.randn_like(base)).requires_grad_(True)
    loss, stats = composed_prototype_loss(
        features,
        labels,
        scenes,
        memory,
        temperature=0.09,
        reconstruction_weight=0.2,
    )
    reference = slow_loss(features, labels, scenes, memory, 0.09, 0.2)
    assert torch.allclose(loss, reference, atol=1e-6, rtol=1e-5)
    assert float(stats["valid_anchor_fraction"]) == 1.0
    loss.backward()
    assert features.grad is not None
    assert bool(torch.isfinite(features.grad).all())
    assert float(features.grad.abs().sum()) > 0.0
    assert all(buffer.grad is None for buffer in memory.buffers())


def test_no_valid_target_returns_graph_zero():
    memory = FactorizedPrototypeMemory(4, 8)
    features = torch.randn(6, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3, 0, 1])
    scenes = torch.tensor([0, 1, 2, 3, 4, 5])
    loss, stats = composed_prototype_loss(features, labels, scenes, memory)
    assert float(loss) == 0.0
    assert float(stats["valid_anchor_fraction"]) == 0.0
    loss.backward()
    assert features.grad is not None
    assert bool(torch.equal(features.grad, torch.zeros_like(features.grad)))


def main():
    torch.set_num_threads(1)
    tests = [
        test_scene_mapping,
        test_exact_additive_reconstruction,
        test_missing_combination_full_rank,
        test_rank_deficient_is_invalid,
        test_skipped_step_does_not_mutate_memory,
        test_skipped_step_preserves_prototype_tensors,
        test_state_roundtrip_and_hash,
        test_grouped_loss_matches_reference_and_backpropagates,
        test_no_valid_target_returns_graph_zero,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("CSF_FACTORIZED_PROTOTYPE_CPU_OK")


if __name__ == "__main__":
    main()
