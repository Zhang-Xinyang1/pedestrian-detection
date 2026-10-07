"""Factorized cross-batch prototype memory for compositional scenarios.

The module has no trainable parameters. It stores detached training-identity
scenario centers and factorizes them into identity, modality, and viewpoint
components under sum-to-zero constraints.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from torch.nn import functional as F


NUM_MODALITIES = 3
NUM_VIEWPOINTS = 2
NUM_SCENARIOS = NUM_MODALITIES * NUM_VIEWPOINTS


def split_scene(scenes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    scenes = torch.as_tensor(scenes)
    if scenes.dtype == torch.bool or scenes.is_complex():
        raise ValueError("scene indices must be integers")
    if scenes.is_floating_point():
        if not bool((torch.isfinite(scenes) & (scenes == scenes.round())).all()):
            raise ValueError("scene indices must be finite integers")
    scenes = scenes.long()
    if bool(((scenes < 0) | (scenes >= NUM_SCENARIOS)).any()):
        raise ValueError("scene indices must be in [0, 5]")
    return scenes.remainder(NUM_MODALITIES), torch.div(
        scenes, NUM_MODALITIES, rounding_mode="floor"
    )


def _factor_design(device=None) -> torch.Tensor:
    rows = []
    for viewpoint in range(NUM_VIEWPOINTS):
        for modality in range(NUM_MODALITIES):
            if modality == 0:
                modality_code = (1.0, 0.0)
            elif modality == 1:
                modality_code = (0.0, 1.0)
            else:
                modality_code = (-1.0, -1.0)
            view_code = 1.0 if viewpoint == 0 else -1.0
            rows.append((1.0, modality_code[0], modality_code[1], view_code))
    return torch.tensor(rows, dtype=torch.float32, device=device)


class FactorizedPrototypeMemory(nn.Module):
    """Detached scenario memory with constrained additive factorization."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        momentum: float = 0.2,
        ridge: float = 1e-4,
        min_mass: float = 1.0,
        mass_cap: float = 64.0,
        max_condition: float = 1e6,
    ) -> None:
        super().__init__()
        if num_classes <= 1 or feature_dim <= 1:
            raise ValueError("num_classes and feature_dim must exceed one")
        if not 0.0 < momentum <= 1.0:
            raise ValueError("momentum must be in (0, 1]")
        if ridge <= 0.0 or min_mass <= 0.0 or mass_cap < min_mass:
            raise ValueError("invalid factorization hyperparameters")
        if max_condition <= 1.0:
            raise ValueError("max_condition must exceed one")

        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.ridge = float(ridge)
        self.min_mass = float(min_mass)
        self.mass_cap = float(mass_cap)
        self.max_condition = float(max_condition)

        shape = (self.num_classes, NUM_MODALITIES, NUM_VIEWPOINTS)
        self.register_buffer(
            "scenario_centers",
            torch.zeros(*shape, self.feature_dim, dtype=torch.float32),
        )
        self.register_buffer("scenario_mass", torch.zeros(*shape, dtype=torch.float32))
        self.register_buffer("scenario_updates", torch.zeros(*shape, dtype=torch.long))
        self.register_buffer(
            "identity_components",
            torch.zeros(self.num_classes, self.feature_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "modality_residuals",
            torch.zeros(
                self.num_classes, NUM_MODALITIES, self.feature_dim, dtype=torch.float32
            ),
        )
        self.register_buffer(
            "view_residuals",
            torch.zeros(
                self.num_classes, NUM_VIEWPOINTS, self.feature_dim, dtype=torch.float32
            ),
        )
        self.register_buffer("identity_valid", torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer("factor_valid", torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer("factor_rank", torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer(
            "factor_condition",
            torch.full((self.num_classes,), float("inf"), dtype=torch.float32),
        )
        self.register_buffer("successful_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("skipped_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("design", _factor_design(), persistent=False)

    def extra_repr(self) -> str:
        return (
            f"classes={self.num_classes}, dim={self.feature_dim}, "
            f"momentum={self.momentum}, ridge={self.ridge}"
        )

    def _validate_batch(
        self, features: torch.Tensor, labels: torch.Tensor, scenes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must be [batch, {self.feature_dim}], got {tuple(features.shape)}"
            )
        labels = torch.as_tensor(labels, device=features.device).reshape(-1).long()
        scenes = torch.as_tensor(scenes, device=features.device).reshape(-1)
        if len(features) != len(labels) or len(features) != len(scenes):
            raise ValueError("batch metadata length mismatch")
        if bool(((labels < 0) | (labels >= self.num_classes)).any()):
            raise ValueError("identity label outside memory range")
        split_scene(scenes)
        if self.scenario_centers.device != features.device:
            raise ValueError("memory and features must be on the same device")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features contain nonfinite values")
        return features, labels, scenes.long()

    @torch.no_grad()
    def _refit_identity(self, identity: int) -> None:
        centers = self.scenario_centers[identity].permute(1, 0, 2).reshape(
            NUM_SCENARIOS, self.feature_dim
        )
        mass = self.scenario_mass[identity].permute(1, 0).reshape(NUM_SCENARIOS)
        observed = mass >= self.min_mass
        self.identity_valid[identity] = bool(observed.any())

        if not bool(observed.any()):
            self.identity_components[identity].zero_()
            self.modality_residuals[identity].zero_()
            self.view_residuals[identity].zero_()
            self.factor_valid[identity] = False
            self.factor_rank[identity] = 0
            self.factor_condition[identity] = float("inf")
            return

        weights = mass[observed].clamp(max=self.mass_cap)
        observed_centers = centers[observed]
        fallback = (observed_centers * weights[:, None]).sum(0) / weights.sum().clamp_min(1e-12)
        self.identity_components[identity] = fallback
        self.modality_residuals[identity].zero_()
        self.view_residuals[identity].zero_()

        design = self.design.to(device=centers.device)[observed]
        weighted_design = design * weights.sqrt()[:, None]
        rank = int(torch.linalg.matrix_rank(weighted_design).item())
        self.factor_rank[identity] = rank
        if rank < 4:
            self.factor_valid[identity] = False
            self.factor_condition[identity] = float("inf")
            return

        normal = design.transpose(0, 1) @ (weights[:, None] * design)
        regularizer = torch.diag(
            torch.tensor(
                [self.ridge * 1e-3, self.ridge, self.ridge, self.ridge],
                dtype=normal.dtype,
                device=normal.device,
            )
        )
        system = normal + regularizer
        condition = float(torch.linalg.cond(system).item())
        self.factor_condition[identity] = condition
        if not torch.isfinite(torch.tensor(condition)) or condition > self.max_condition:
            self.factor_valid[identity] = False
            return

        right = design.transpose(0, 1) @ (weights[:, None] * observed_centers)
        try:
            theta = torch.linalg.solve(system, right)
        except RuntimeError:
            self.factor_valid[identity] = False
            return

        identity_component = theta[0]
        modality_rgb = theta[1]
        modality_nir = theta[2]
        modality_tir = -(modality_rgb + modality_nir)
        view_ground = theta[3]
        view_aerial = -view_ground

        factors = torch.cat(
            [
                identity_component.reshape(1, -1),
                torch.stack((modality_rgb, modality_nir, modality_tir)),
                torch.stack((view_ground, view_aerial)),
            ],
            dim=0,
        )
        if not bool(torch.isfinite(factors).all()):
            self.factor_valid[identity] = False
            return

        self.identity_components[identity] = identity_component
        self.modality_residuals[identity] = torch.stack(
            (modality_rgb, modality_nir, modality_tir)
        )
        self.view_residuals[identity] = torch.stack((view_ground, view_aerial))
        self.factor_valid[identity] = True

    @torch.no_grad()
    def update(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        scenes: torch.Tensor,
        step_succeeded: bool = True,
    ) -> Dict[str, float]:
        features, labels, scenes = self._validate_batch(features, labels, scenes)
        if not step_succeeded:
            self.skipped_updates.add_(1)
            return self.batch_diagnostics(labels)

        normalized = F.normalize(features.detach().float(), dim=1)
        modalities, viewpoints = split_scene(scenes)
        packed = labels * NUM_SCENARIOS + scenes
        touched_identities = []
        updated_tuples = 0

        for packed_value in torch.unique(packed, sorted=True).tolist():
            identity = int(packed_value // NUM_SCENARIOS)
            scene = int(packed_value % NUM_SCENARIOS)
            modality = scene % NUM_MODALITIES
            viewpoint = scene // NUM_MODALITIES
            mask = packed == packed_value
            center = F.normalize(normalized[mask].mean(0), dim=0)
            old_mass = self.scenario_mass[identity, modality, viewpoint]
            if float(old_mass.item()) < self.min_mass:
                updated = center
            else:
                previous = self.scenario_centers[identity, modality, viewpoint]
                updated = F.normalize(
                    (1.0 - self.momentum) * previous + self.momentum * center, dim=0
                )
            self.scenario_centers[identity, modality, viewpoint] = updated
            self.scenario_mass[identity, modality, viewpoint] += int(mask.sum().item())
            self.scenario_updates[identity, modality, viewpoint] += 1
            touched_identities.append(identity)
            updated_tuples += 1

        for identity in sorted(set(touched_identities)):
            self._refit_identity(identity)
        self.successful_updates.add_(1)
        result = self.batch_diagnostics(labels)
        result["updated_tuples"] = float(updated_tuples)
        result["updated_identities"] = float(len(set(touched_identities)))
        return result

    def compose(
        self, scene: int, allow_identity_fallback: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scene_tensor = torch.tensor([scene], device=self.scenario_centers.device)
        modalities, viewpoints = split_scene(scene_tensor)
        modality = int(modalities.item())
        viewpoint = int(viewpoints.item())
        raw = (
            self.identity_components
            + self.modality_residuals[:, modality]
            + self.view_residuals[:, viewpoint]
        )
        fallback = self.identity_valid & ~self.factor_valid
        if allow_identity_fallback and bool(fallback.any()):
            raw = raw.clone()
            raw[fallback] = self.identity_components[fallback]
        valid = self.factor_valid | (self.identity_valid if allow_identity_fallback else False)
        prototypes = F.normalize(raw.float(), dim=1)
        prototypes = torch.where(valid[:, None], prototypes, torch.zeros_like(prototypes))
        return prototypes, valid, fallback & valid

    def batch_diagnostics(self, labels: torch.Tensor) -> Dict[str, float]:
        labels = torch.as_tensor(labels, device=self.factor_valid.device).reshape(-1).long()
        factor_targets = self.factor_valid[labels] if len(labels) else self.factor_valid[:0]
        identity_targets = self.identity_valid[labels] if len(labels) else self.identity_valid[:0]
        finite_conditions = self.factor_condition[
            self.factor_valid & torch.isfinite(self.factor_condition)
        ]
        return {
            "factor_valid_identity_fraction": float(self.factor_valid.float().mean().item()),
            "identity_initialized_fraction": float(self.identity_valid.float().mean().item()),
            "factor_valid_sample_fraction": float(
                factor_targets.float().mean().item() if len(labels) else 0.0
            ),
            "identity_initialized_sample_fraction": float(
                identity_targets.float().mean().item() if len(labels) else 0.0
            ),
            "mean_condition": float(
                finite_conditions.mean().item() if len(finite_conditions) else float("inf")
            ),
            "successful_memory_updates": float(self.successful_updates.item()),
            "skipped_memory_updates": float(self.skipped_updates.item()),
        }

    def state_sha256(self, include_derived: bool = True) -> str:
        digest = hashlib.sha256()
        derived = {
            "identity_components",
            "modality_residuals",
            "view_residuals",
            "identity_valid",
            "factor_valid",
            "factor_rank",
            "factor_condition",
        }
        for name, tensor in sorted(self.named_buffers()):
            if name == "design" or (not include_derived and name in derived):
                continue
            value = tensor.detach().contiguous().cpu()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def factor_metrics(self) -> Dict[str, float]:
        valid = self.factor_valid
        if not bool(valid.any()):
            return {
                "identity_modality_orthogonality": 0.0,
                "identity_view_orthogonality": 0.0,
                "modality_view_orthogonality": 0.0,
                "modality_residual_norm": 0.0,
                "view_residual_norm": 0.0,
            }
        identity = F.normalize(self.identity_components[valid].float(), dim=1)
        modality = self.modality_residuals[valid].float()
        view = self.view_residuals[valid].float()
        identity_modality = torch.einsum("bd,bmd->bm", identity, modality).abs().mean()
        identity_view = torch.einsum("bd,bvd->bv", identity, view).abs().mean()
        modality_view = torch.einsum("bmd,bvd->bmv", modality, view).abs().mean()
        return {
            "identity_modality_orthogonality": float(identity_modality.item()),
            "identity_view_orthogonality": float(identity_view.item()),
            "modality_view_orthogonality": float(modality_view.item()),
            "modality_residual_norm": float(modality.norm(dim=2).mean().item()),
            "view_residual_norm": float(view.norm(dim=2).mean().item()),
        }


def composed_prototype_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    scenes: torch.Tensor,
    memory: FactorizedPrototypeMemory,
    temperature: float = 0.07,
    reconstruction_weight: float = 0.1,
    require_factor_target: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Grouped composed-prototype discrimination without B by C by D expansion."""

    if temperature <= 0.0 or reconstruction_weight < 0.0:
        raise ValueError("invalid CSF loss hyperparameters")
    features, labels, scenes = memory._validate_batch(features, labels, scenes)
    z = F.normalize(features.float(), dim=1)
    total_ce = z.sum() * 0.0
    total_reconstruction = z.sum() * 0.0
    valid_count = 0
    factor_target_count = int(memory.factor_valid[labels].sum().item())
    identity_target_count = int(memory.identity_valid[labels].sum().item())
    candidate_count_sum = 0
    fallback_candidate_sum = 0
    groups = 0

    for scene_value in torch.unique(scenes, sorted=True).tolist():
        group = torch.nonzero(scenes == scene_value, as_tuple=False).reshape(-1)
        prototypes, candidate_valid, fallback_candidates = memory.compose(
            int(scene_value), allow_identity_fallback=True
        )
        target_valid = memory.identity_valid[labels[group]]
        if require_factor_target:
            target_valid = target_valid & memory.factor_valid[labels[group]]
        selected = group[target_valid]
        if not len(selected) or not bool(candidate_valid.any()):
            continue
        logits = z[selected] @ prototypes.detach().transpose(0, 1)
        logits = logits / float(temperature)
        logits = logits.masked_fill(
            ~candidate_valid[None, :], torch.finfo(logits.dtype).min
        )
        total_ce = total_ce + F.cross_entropy(
            logits, labels[selected], reduction="sum"
        )
        positives = prototypes.detach()[labels[selected]]
        total_reconstruction = total_reconstruction + (
            1.0 - (z[selected] * positives).sum(dim=1)
        ).sum()
        valid_count += len(selected)
        candidate_count_sum += int(candidate_valid.sum().item())
        fallback_candidate_sum += int(fallback_candidates.sum().item())
        groups += 1

    denominator = max(valid_count, 1)
    ce = total_ce / denominator
    reconstruction = total_reconstruction / denominator
    loss = ce + float(reconstruction_weight) * reconstruction
    batch_size = max(len(features), 1)
    stats = {
        "ce": ce.detach(),
        "reconstruction": reconstruction.detach(),
        "valid_anchor_fraction": z.new_tensor(valid_count / batch_size),
        "factor_target_fraction": z.new_tensor(factor_target_count / batch_size),
        "identity_target_fraction": z.new_tensor(identity_target_count / batch_size),
        "mean_candidate_identities": z.new_tensor(
            candidate_count_sum / max(groups, 1)
        ),
        "mean_fallback_candidates": z.new_tensor(
            fallback_candidate_sum / max(groups, 1)
        ),
        "groups": z.new_tensor(float(groups)),
    }
    return loss, stats

