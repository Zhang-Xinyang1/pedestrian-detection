"""Matched prototype geometry ablation using one shared observation memory."""
import torch
from torch.nn import functional as F
from factorized_prototype import FactorizedPrototypeMemory, split_scene

METHODS = ("control", "identity", "identity_full", "balanced_modality", "six_scene", "csf", "csf_full_identity")

class PrototypeComparisonMemory(FactorizedPrototypeMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "instance_identity_centers",
            torch.zeros(self.num_classes, self.feature_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "instance_identity_mass", torch.zeros(self.num_classes, dtype=torch.long)
        )

    @torch.no_grad()
    def update(self, features, labels, scenes, step_succeeded=True):
        features, labels, scenes = self._validate_batch(features, labels, scenes)
        result = super().update(features, labels, scenes, step_succeeded)
        if not step_succeeded:
            return result
        z = F.normalize(features.detach().float(), dim=1)
        for identity in labels.unique(sorted=True).tolist():
            selected = labels == identity
            center = F.normalize(z[selected].mean(0), dim=0)
            if self.instance_identity_mass[identity] > 0:
                center = F.normalize(
                    (1.0 - self.momentum) * self.instance_identity_centers[identity]
                    + self.momentum * center,
                    dim=0,
                )
            self.instance_identity_centers[identity] = center
            self.instance_identity_mass[identity] += selected.sum()
        return result

    def matched_prototypes(self, method, scene):
        if method not in METHODS:
            raise ValueError("Unknown prototype method")
        modality, viewpoint = split_scene(
            torch.tensor([scene], device=self.scenario_centers.device)
        )
        modality, viewpoint = int(modality.item()), int(viewpoint.item())
        modality_mass = self.scenario_mass.sum(dim=2)
        # Matched methods retain the original factor-valid mask. The new
        # identity_full/csf_full_identity arm uses every identity observed in
        # memory; identities without a full-rank factor solve fall back to
        # their observed global center and never fabricate an unseen view.
        factor_valid = (
            self.factor_valid
            & (self.scenario_mass[:, modality, viewpoint] >= self.min_mass)
            & (modality_mass >= self.min_mass).all(dim=1)
            & (self.instance_identity_mass > 0)
        )
        full_valid = self.instance_identity_mass > 0
        if method in ("control", "identity"):
            valid = factor_valid
            raw = self.instance_identity_centers
        elif method == "identity_full":
            valid = full_valid
            raw = self.instance_identity_centers
        elif method == "balanced_modality":
            valid = factor_valid
            per_modality = (
                self.scenario_centers * self.scenario_mass[..., None]
            ).sum(dim=2) / modality_mass.clamp_min(1e-12)[..., None]
            raw = F.normalize(per_modality, dim=2).mean(dim=1)
        elif method == "six_scene":
            valid = factor_valid
            raw = self.scenario_centers[:, modality, viewpoint]
        elif method == "csf":
            valid = factor_valid
            raw = (
                self.identity_components
                + self.modality_residuals[:, modality]
                + self.view_residuals[:, viewpoint]
            )
        else:
            valid = full_valid
            composed = (
                self.identity_components
                + self.modality_residuals[:, modality]
                + self.view_residuals[:, viewpoint]
            )
            raw = composed.clone()
            fallback = full_valid & ~self.factor_valid
            raw[fallback] = self.identity_components[fallback]
        return F.normalize(raw.float(), dim=1), valid

def prototype_comparison_loss(
    features, labels, scenes, memory, method,
    temperature=0.07, reconstruction_weight=0.1,
):
    if method not in METHODS or temperature <= 0 or reconstruction_weight < 0:
        raise ValueError("Invalid prototype comparison configuration")
    features, labels, scenes = memory._validate_batch(features, labels, scenes)
    z = F.normalize(features.float(), dim=1)
    ce = z.sum() * 0.0
    reconstruction = z.sum() * 0.0
    valid_count = candidate_sum = groups = 0
    for scene in scenes.unique(sorted=True).tolist():
        prototypes, candidates = memory.matched_prototypes(method, scene)
        indices = torch.nonzero(scenes == scene, as_tuple=False).reshape(-1)
        indices = indices[candidates[labels[indices]]]
        if len(indices) == 0:
            continue
        valid_count += len(indices)
        candidate_sum += int(candidates.sum().item())
        groups += 1
        if method == "control":
            continue
        logits = z[indices] @ prototypes.detach().t() / temperature
        logits = logits.masked_fill(~candidates[None, :], torch.finfo(logits.dtype).min)
        ce = ce + F.cross_entropy(logits, labels[indices], reduction="sum")
        reconstruction = reconstruction + (
            1.0 - (z[indices] * prototypes.detach()[labels[indices]]).sum(1)
        ).sum()
    ce = ce / max(valid_count, 1)
    reconstruction = reconstruction / max(valid_count, 1)
    loss = ce + reconstruction_weight * reconstruction
    stats = {
        "ce": ce.detach(), "reconstruction": reconstruction.detach(),
        "valid_anchor_fraction": z.new_tensor(valid_count / max(len(z), 1)),
        "factor_target_fraction": memory.factor_valid[labels].float().mean(),
        "identity_target_fraction": memory.identity_valid[labels].float().mean(),
        "mean_candidate_identities": z.new_tensor(candidate_sum / max(groups, 1)),
        "mean_fallback_candidates": z.new_zeros(()),
        "groups": z.new_tensor(float(groups)),
    }
    return loss, stats
