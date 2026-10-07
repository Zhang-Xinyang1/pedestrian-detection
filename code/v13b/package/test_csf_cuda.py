"""CUDA tests for CSF gradients, memory state, and resource budget."""

import torch
from torch.nn import functional as F

from factorized_prototype import (
    FactorizedPrototypeMemory,
    composed_prototype_loss,
    split_scene,
)


def additive_batch(classes, dim, device, seed=17):
    generator = torch.Generator(device=device).manual_seed(seed)
    identity = F.normalize(
        torch.randn(classes, dim, generator=generator, device=device), dim=1
    )
    modality = torch.randn(
        classes, 3, dim, generator=generator, device=device
    ) * 0.02
    modality[:, 2] = -(modality[:, 0] + modality[:, 1])
    view = torch.randn(classes, 2, dim, generator=generator, device=device) * 0.02
    view[:, 1] = -view[:, 0]
    labels = torch.arange(classes, device=device).repeat_interleave(6)
    scenes = torch.arange(6, device=device).repeat(classes)
    modalities, viewpoints = split_scene(scenes)
    features = F.normalize(
        identity[labels] + modality[labels, modalities] + view[labels, viewpoints],
        dim=1,
    )
    return features, labels, scenes


def test_cuda_gradient_and_detached_memory(device):
    classes, dim = 8, 768
    memory = FactorizedPrototypeMemory(
        classes, dim, momentum=1.0, ridge=1e-6
    ).to(device)
    bootstrap, bootstrap_labels, bootstrap_scenes = additive_batch(
        classes, dim, device
    )
    memory.update(bootstrap, bootstrap_labels, bootstrap_scenes)
    assert bool(memory.factor_valid.all())

    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7] * 2, device=device)
    scenes = torch.tensor([0, 1, 2, 3, 4, 5, 0, 3] * 2, device=device)
    modalities, viewpoints = split_scene(scenes)
    prototypes = []
    for label, modality, viewpoint in zip(labels, modalities, viewpoints):
        raw = (
            memory.identity_components[label]
            + memory.modality_residuals[label, modality]
            + memory.view_residuals[label, viewpoint]
        )
        prototypes.append(raw)
    features = (torch.stack(prototypes) + 0.01 * torch.randn(16, dim, device=device))
    features.requires_grad_(True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss, stats = composed_prototype_loss(
            features,
            labels,
            scenes,
            memory,
            temperature=0.07,
            reconstruction_weight=0.1,
        )
    loss.backward()
    assert features.grad is not None
    assert bool(torch.isfinite(features.grad).all())
    assert float(features.grad.abs().sum()) > 0.0
    assert float(stats["valid_anchor_fraction"]) == 1.0
    assert all(parameter.requires_grad is False for parameter in memory.parameters())
    assert all(buffer.grad is None for buffer in memory.buffers())
    print(
        "CUDA_GRADIENT_OK",
        {
            "loss": float(loss.detach()),
            "grad_norm": float(features.grad.float().norm()),
            "valid_anchor_fraction": float(stats["valid_anchor_fraction"]),
        },
    )


def test_cuda_skipped_update_contract(device):
    memory = FactorizedPrototypeMemory(8, 64, momentum=1.0).to(device)
    bootstrap, labels, scenes = additive_batch(8, 64, device, seed=23)
    memory.update(bootstrap, labels, scenes)
    protected = {
        name: tensor.clone()
        for name, tensor in memory.named_buffers()
        if name not in {"skipped_updates", "design"}
    }
    state_before = memory.state_sha256()
    memory.update(bootstrap[:16], labels[:16], scenes[:16], step_succeeded=False)
    state_after = memory.state_sha256()
    assert state_before != state_after
    assert int(memory.skipped_updates.item()) == 1
    current = dict(memory.named_buffers())
    for name, expected in protected.items():
        assert torch.equal(current[name], expected), name
    print("CUDA_SKIPPED_UPDATE_OK", {"skipped_updates": 1})


def test_cuda_memory_budget(device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.memory_allocated(device)
    memory = FactorizedPrototypeMemory(500, 768).to(device)
    allocated = torch.cuda.memory_allocated(device) - start
    persistent_bytes = sum(
        tensor.numel() * tensor.element_size()
        for name, tensor in memory.named_buffers()
        if name != "design"
    )
    assert allocated < 64 * 1024 * 1024
    assert persistent_bytes < 20 * 1024 * 1024
    assert sum(parameter.numel() for parameter in memory.parameters()) == 0
    print(
        "CUDA_MEMORY_BUDGET_OK",
        {
            "allocated_mib": allocated / 1024**2,
            "persistent_mib": persistent_bytes / 1024**2,
            "trainable_parameters": 0,
        },
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print("CSF_CUDA_DEVICE", torch.cuda.get_device_name(device))
    test_cuda_gradient_and_detached_memory(device)
    test_cuda_skipped_update_contract(device)
    test_cuda_memory_budget(device)
    print("CSF_CUDA_TESTS_OK")


if __name__ == "__main__":
    main()
