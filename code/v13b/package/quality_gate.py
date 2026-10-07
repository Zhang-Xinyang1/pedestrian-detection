"""Small, sample-wise reliability gate for the existing 512-D I2T branch.

The gate is deliberately initialized to an exact identity mapping.  Its inputs
are image-only statistics computed from the already normalized tensor; no
identity, split, or evaluation metadata is used.
"""
import torch
from torch import nn
from torch.nn import functional as F


def image_quality_features(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("expected BCHW RGB tensor")
    if not torch.isfinite(x).all():
        raise ValueError("image tensor contains non-finite values")
    # Statistics are scale-stable for the normalized CLIP input and remain
    # valid for RGB, NIR, and pseudo-colour thermal images.
    mean = x.mean(dim=(1, 2, 3))
    std = x.std(dim=(1, 2, 3), unbiased=False)
    gray = x.mean(dim=1)
    gray_std = gray.std(dim=(1, 2), unbiased=False)
    dx = gray[:, :, 1:] - gray[:, :, :-1]
    dy = gray[:, 1:, :] - gray[:, :-1, :]
    edge = 0.5 * (dx.abs().mean(dim=(1, 2)) + dy.abs().mean(dim=(1, 2)))
    # A bounded saturation/contrast proxy and a clipping proxy.  The latter
    # is measured in normalized coordinates, so it does not inspect labels.
    channel_spread = x.amax(dim=1).sub(x.amin(dim=1)).mean(dim=(1, 2))
    clipped = (x.abs() > 2.5).float().mean(dim=(1, 2, 3))
    out = torch.stack((mean, std, gray_std, edge, channel_spread, clipped), dim=1)
    return torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)


class ReliabilityGate(nn.Module):
    """Return alpha in [0.75, 1.25], multiplying only the I2T feature."""
    def __init__(self, feature_dim: int = 6):
        super().__init__()
        self.proj = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = image_quality_features(x).to(dtype=self.proj.weight.dtype)
        return 1.0 + 0.25 * torch.tanh(self.proj(z)).squeeze(1)

