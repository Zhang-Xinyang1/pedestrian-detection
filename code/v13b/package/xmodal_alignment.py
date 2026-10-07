"""Cross-modality alignment loss without changing data or sampling."""
import torch
from torch.nn import functional as F


def cross_modal_supcon_loss(features, labels, scenes, temperature=0.07):
    if features.ndim != 2:
        raise ValueError("features must be [batch, dim]")
    labels = labels.reshape(-1)
    scenes = scenes.reshape(-1)
    if len(features) != len(labels) or len(features) != len(scenes):
        raise ValueError("batch metadata length mismatch")
    if not 0.0 < float(temperature):
        raise ValueError("temperature must be positive")
    modalities = scenes.long().remainder(3)
    same_identity = labels[:, None].eq(labels[None, :])
    same_modality = modalities[:, None].eq(modalities[None, :])
    eye = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positives = same_identity & ~same_modality & ~eye
    denominator = ~eye & ~(same_identity & same_modality)
    valid = positives.any(dim=1)
    z = F.normalize(features.float(), dim=1)
    if not bool(valid.any()):
        return z.sum() * 0.0, valid.float().mean()
    logits = z @ z.t()
    logits = logits / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    negative_infinity = torch.finfo(logits.dtype).min
    log_denominator = torch.logsumexp(
        logits.masked_fill(~denominator, negative_infinity), dim=1
    )
    log_probability = logits - log_denominator[:, None]
    positive_count = positives.sum(dim=1).clamp_min(1)
    per_anchor = -(
        log_probability.masked_fill(~positives, 0.0).sum(dim=1) / positive_count
    )
    return per_anchor[valid].mean(), valid.float().mean()
