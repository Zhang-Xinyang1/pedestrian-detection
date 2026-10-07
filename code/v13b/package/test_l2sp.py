"""Analytical gradient, strict reference, metadata and AMP tests."""
import argparse
import torch
from torch import nn
from starting_point_regularization import StartingPointRegularizer, COEFFICIENT, compare_visual_states


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder=nn.Module()
        self.image_encoder.proj=nn.Parameter(torch.randn(8,4))
        self.image_encoder.linear=nn.Linear(8,8)
        self.classifier=nn.Linear(8,5)


def main(device):
    torch.manual_seed(1);torch.set_num_threads(2)
    model=ToyModel().to(device)
    reference={n:t.detach().clone() for n,t in model.state_dict().items()}
    reg=StartingPointRegularizer(model)
    assert sum(p.numel() for p in reg.parameters())==0
    assert reg(model).item()==0
    assert reg.anchor_hashes()==reg.initial_hashes
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n.startswith('image_encoder.'):p.add_(.02)
            else:p.add_(.4)
    reg(model).backward()
    expected=.5*COEFFICIENT*reg.element_count*(.02**2)
    assert abs(reg(model).item()-expected)<1e-8
    for name,p in model.named_parameters():
        if name.startswith('image_encoder.'):
            assert torch.allclose(p.grad,COEFFICIENT*(p-reference[name]),atol=1e-10)
        else:assert p.grad is None
    drift=reg.drift(model)
    compared=compare_visual_states(reference,model.state_dict())
    assert abs(drift['weighted_penalty']-compared['weighted_penalty'])<1e-12
    assert reg.anchor_hashes()==reg.initial_hashes
    restored=StartingPointRegularizer(ToyModel().to(device))
    restored.load_state_dict(reg.state_dict(),strict=True)
    assert torch.equal(restored(model),reg(model))
    try:StartingPointRegularizer(model,coefficient=float('nan'))
    except ValueError:pass
    else:raise AssertionError('Nonfinite coefficient accepted')
    if device=='cuda':
        assert torch.cuda.device_count()==1
        model.zero_grad()
        with torch.autocast('cuda',dtype=torch.float16):penalty=reg(model)
        assert penalty.dtype==torch.float32 and torch.isfinite(penalty)
        scaler=torch.cuda.amp.GradScaler()
        opt=torch.optim.Adam(model.parameters(),lr=1e-4)
        old={n:p.detach().clone() for n,p in model.named_parameters()}
        scaler.scale(penalty*float('inf')).backward()
        scaler.step(opt);scaler.update()
        assert all(torch.equal(p,old[n]) for n,p in model.named_parameters())
        assert reg.anchor_hashes()==reg.initial_hashes
    print('L2SP_TESTS_OK',device,'coefficient',COEFFICIENT,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    main(p.parse_args().device)
