import torch
from prototype_comparison import METHODS, PrototypeComparisonMemory, prototype_comparison_loss

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(47)
    memory = PrototypeComparisonMemory(8, 768).cuda()
    labels = torch.arange(8, device="cuda").repeat_interleave(6)
    scenes = torch.arange(6, device="cuda").repeat(8)
    samples = torch.randn(48, 768, device="cuda")
    memory.update(samples, labels, scenes)
    for method in METHODS:
        x = samples.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss, stats = prototype_comparison_loss(x, labels, scenes, memory, method)
        if method == "control":
            assert loss.item() == 0
        loss.backward()
        assert torch.isfinite(x.grad).all()
        if method != "control":
            assert x.grad.abs().sum() > 0
        assert stats["valid_anchor_fraction"] == 1
    saved = {name:tensor.clone() for name,tensor in memory.state_dict().items()}
    memory.update(samples, labels, scenes, step_succeeded=False)
    for name,tensor in memory.state_dict().items():
        if name != "skipped_updates":
            assert torch.equal(tensor,saved[name]), name
    print("PROTOTYPE_COMPARISON_CUDA_OK",torch.cuda.get_device_name(0))

if __name__=="__main__":
    main()
