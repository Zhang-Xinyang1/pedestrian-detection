"""Real memory behavior for single-view identities; no invented opposite-view factors."""
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).parent/'package'))
from prototype_comparison import PrototypeComparisonMemory,prototype_comparison_loss

torch.manual_seed(19)
memory=PrototypeComparisonMemory(3,12,momentum=1.)
labels=torch.arange(3).repeat_interleave(3)
scenes=torch.arange(3).repeat(3)
features=torch.randn(9,12)
memory.update(features,labels,scenes)
assert not memory.factor_valid.any() and memory.identity_valid.all()
for scene in range(6):
    bank,mask=memory.matched_prototypes('csf_full_identity',scene)
    assert mask.all()
    assert torch.allclose(bank,torch.nn.functional.normalize(memory.identity_components,dim=1))
    assert not memory.matched_prototypes('csf',scene)[1].any()
x=features.clone().requires_grad_()
loss,stats=prototype_comparison_loss(x,labels,scenes,memory,'csf_full_identity')
assert stats['valid_anchor_fraction']==1. and stats['mean_candidate_identities']==3
loss.backward();assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
assert not memory.modality_residuals.any() and not memory.view_residuals.any()
before=memory.state_sha256()
memory.update(features,labels,scenes,step_succeeded=False)
assert memory.successful_updates.item()==1 and memory.skipped_updates.item()==1
print('SCNET_SINGLE_VIEW_IDENTITY_FALLBACK_PASS gradient=True fabricated_view=False')
