"""Unit tests for cross-modality alignment loss."""
import argparse
import torch
from xmodal_alignment import cross_modal_supcon_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.cuda:
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.manual_seed(1)
    features = torch.randn(8, 768, device=device, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device)
    scenes = torch.tensor([0, 1, 2, 0, 3, 4, 5, 3], device=device)
    loss, valid = cross_modal_supcon_loss(features, labels, scenes, 0.07)
    assert torch.isfinite(loss) and float(valid) == 1.0
    loss.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert float(features.grad.abs().sum()) > 0.0
    unique_labels = torch.arange(8, device=device)
    no_positive = torch.randn(8, 32, device=device, requires_grad=True)
    zero, zero_valid = cross_modal_supcon_loss(no_positive, unique_labels, scenes, 0.07)
    assert float(zero_valid) == 0.0 and float(zero) == 0.0
    zero.backward()
    assert float(no_positive.grad.abs().sum()) == 0.0
    permutation = torch.tensor([7, 2, 5, 0, 3, 6, 1, 4], device=device)
    permuted_loss, permuted_valid = cross_modal_supcon_loss(
        features.detach()[permutation], labels[permutation], scenes[permutation], 0.07
    )
    assert torch.allclose(loss.detach(), permuted_loss, atol=1e-6, rtol=1e-5)
    assert torch.equal(valid, permuted_valid)
    print("XMODAL_ALIGNMENT_UNIT_OK", {
        "device": str(device), "loss": float(loss.detach()),
        "valid_anchor_fraction": float(valid), "feature_dim": features.shape[1]
    })


if __name__ == "__main__":
    main()
