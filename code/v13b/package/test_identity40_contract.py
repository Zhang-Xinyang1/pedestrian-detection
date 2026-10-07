"""Check global identity targets, single-view coverage, gradients and fixed40 LR."""
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'upstream'))
from prototype_comparison import PrototypeComparisonMemory,prototype_comparison_loss
from solver.lr_scheduler import WarmupMultiStepLR
from verify_results_csf import validate_epoch_budget

def main():
    torch.set_num_threads(1);torch.manual_seed(51)
    m=PrototypeComparisonMemory(4,12,momentum=.2)
    labels=torch.tensor([0]*6+[1]*3)
    scenes=torch.tensor(list(range(6))+[0,1,2])
    features=torch.randn(9,12)
    m.update(features,labels,scenes)
    assert bool(m.factor_valid[0]) and not bool(m.factor_valid[1])
    reference=F.normalize(m.instance_identity_centers,dim=1)
    first=None
    for scene in range(6):
        centers,valid=m.matched_prototypes('identity_full',scene)
        assert valid.tolist()==[True,True,False,False]
        assert torch.equal(centers,reference)
        if first is None:first=centers
        else:assert torch.equal(first,centers)
    assert not torch.count_nonzero(m.scenario_mass[1,:,1])
    assert not torch.count_nonzero(m.scenario_centers[1,:,1])
    x=torch.randn(9,12,requires_grad=True)
    loss,stats=prototype_comparison_loss(x,labels,scenes,m,'identity_full')
    assert stats['valid_anchor_fraction']==1 and stats['mean_candidate_identities']==2
    loss.backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    p=torch.nn.Parameter(torch.zeros(1));opt=torch.optim.Adam([p],lr=5e-6)
    schedule=WarmupMultiStepLR(opt,[21],.1,.1,10,'linear');values=[]
    for epoch in range(1,41):
        schedule.step();values.append(opt.param_groups[0]['lr'])
        p.sum().backward();opt.step();opt.zero_grad()
    assert all(math.isclose(v,5e-7,rel_tol=1e-12) for v in values[20:])
    record={'final_epoch':40,'stopped_early':False,'early_stop_reason':None,
            'csf':{'final_epoch':40,'stopped_early':False,'early_stop_reason':None}}
    assert validate_epoch_budget(record,screening_40=True)==40
    assert "'identity_full'" in (ROOT/'run_stage2_only.py').read_text()
    print('IDENTITY_FULL_FIXED40_CONTRACT_OK')

if __name__=='__main__':main()
