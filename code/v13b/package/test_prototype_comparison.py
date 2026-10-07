"""Test matched candidate masks, reference logits, state and control equivalence."""
import copy
import torch
from torch.nn import functional as F
from prototype_comparison import METHODS, PrototypeComparisonMemory, prototype_comparison_loss

def main():
    torch.set_num_threads(1)
    torch.manual_seed(37)
    memory = PrototypeComparisonMemory(4, 12, momentum=1.0)
    labels = torch.arange(4).repeat_interleave(6)
    scenes = torch.arange(6).repeat(4)
    features = torch.randn(24, 12)
    memory.update(features, labels, scenes)
    assert memory.factor_valid.all()
    for scene in range(6):
        masks = [memory.matched_prototypes(method, scene)[1] for method in METHODS]
        assert all(torch.equal(masks[0], mask) for mask in masks)
    expected_identity = torch.stack([
        F.normalize(F.normalize(features[labels == identity], dim=1).mean(0), dim=0)
        for identity in range(4)
    ])
    assert torch.allclose(memory.instance_identity_centers, expected_identity, atol=1e-6)
    for scene in range(6):
        bank, _ = memory.matched_prototypes("six_scene", scene)
        assert torch.allclose(
            bank, memory.scenario_centers[:, scene % 3, scene // 3], atol=1e-6
        )
    masks_before = memory.scenario_mass.clone()
    centers_before = memory.instance_identity_centers.clone()
    memory.update(features, labels, scenes, step_succeeded=False)
    assert torch.equal(masks_before, memory.scenario_mass)
    assert torch.equal(centers_before, memory.instance_identity_centers)
    clone = PrototypeComparisonMemory(4, 12, momentum=1.0)
    clone.load_state_dict(copy.deepcopy(memory.state_dict()))
    assert clone.state_sha256() == memory.state_sha256()
    for method in METHODS:
        x = features.clone().requires_grad_(True)
        loss, stats = prototype_comparison_loss(x, labels, scenes, memory, method)
        assert stats["valid_anchor_fraction"] == 1
        if method == "control":
            assert loss.item() == 0
            reference_features = x.detach().clone().requires_grad_(True)
            reference_features.square().mean().backward()
            (x.square().mean() + 0.0 * loss).backward()
            assert torch.equal(x.grad, reference_features.grad)
        else:
            expected = []
            for i in range(len(x)):
                bank, valid = memory.matched_prototypes(method, int(scenes[i]))
                row = F.normalize(x[i:i+1], dim=1)
                logits = row @ bank.t() / 0.07
                logits = logits.masked_fill(~valid[None, :], torch.finfo(logits.dtype).min)
                expected.append(F.cross_entropy(logits, labels[i:i+1])
                    + 0.1 * (1.0 - (row[0] * bank[labels[i]]).sum()))
            assert torch.allclose(loss, torch.stack(expected).mean(), atol=1e-5)
            loss.backward()
            assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    # Matched methods invalidate a missing current scene; the full identity arm
    # intentionally keeps the observed global identity fallback valid.
    memory.scenario_mass[0, 2, 1] = 0
    masks = [memory.matched_prototypes(method, 5)[1] for method in METHODS]
    for method, mask in zip(METHODS, masks):
        if method in ('identity_full', 'csf_full_identity'):
            assert bool(mask[0])
        else:
            assert not bool(mask[0])
    assert torch.equal(masks[0], masks[1])
    assert torch.equal(masks[1], masks[3])
    assert torch.equal(masks[3], masks[4])
    assert torch.equal(masks[4], masks[5])
    assert sum(parameter.numel() for parameter in memory.parameters()) == 0
    print("PROTOTYPE_COMPARISON_CPU_OK", list(METHODS))

if __name__ == "__main__":
    main()
