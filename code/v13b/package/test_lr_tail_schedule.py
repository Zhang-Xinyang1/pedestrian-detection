"""Verify exact first40 control, actual optimizer LR and compressed GPU tail."""
import argparse
import copy
import json
import math
import torch
from lr_tail_schedule import MatchedTailScheduler,tail_lr,tail_specification,validate_optimizer_lrs
from solver.lr_scheduler import WarmupMultiStepLR

def make_optimizer(device):
    weight=torch.nn.Parameter(torch.tensor([1.,2.],device=device));bias=torch.nn.Parameter(torch.tensor([.5],device=device))
    return [weight,bias],torch.optim.Adam([{'params':[weight],'lr':5e-6},{'params':[bias],'lr':1e-5}],weight_decay=1e-4)

def main(device):
    pa,oa=make_optimizer(device);pb,ob=make_optimizer(device);pr,original=make_optimizer(device)
    sa=MatchedTailScheduler(oa,[21],.1,.1,10,'linear',arm='A')
    sb=MatchedTailScheduler(ob,[21],.1,.1,10,'linear',arm='B')
    reference=WarmupMultiStepLR(original,[21],.1,.1,10,'linear')
    history=[]
    for epoch in range(1,101):
        sa.step();sb.step();reference.step()
        rowa=validate_optimizer_lrs(oa,sa,epoch);rowb=validate_optimizer_lrs(ob,sb,epoch)
        assert oa.param_groups[0]['lr']==original.param_groups[0]['lr']
        if epoch<=40:assert rowa['base_lr']==rowb['base_lr']
        else:assert rowb['base_lr']<rowa['base_lr']
        for params,optimizer in [(pa,oa),(pb,ob),(pr,original)]:
            optimizer.zero_grad();sum((p*p).sum() for p in params).backward();optimizer.step()
        assert all(torch.equal(a,r) for a,r in zip(pa,pr))
        if epoch<=40:assert all(torch.equal(a,b) for a,b in zip(pa,pb))
        history.append({'epoch':epoch,'A':rowa['base_lr'],'B':rowb['base_lr']})
    assert math.isclose(history[-1]['B'],1e-8,abs_tol=1e-16)
    assert history[39]['A']==history[39]['B'] and math.isclose(history[39]['A'],5e-7,abs_tol=1e-16)
    assert all(torch.isfinite(p).all() for p in pa+pb)
    params,opt=make_optimizer(device);scheduler=MatchedTailScheduler(opt,[21],.1,.1,10,'linear',arm='B',tail_start=2,tail_end=4)
    for epoch in range(1,5):
        scheduler.step();validate_optimizer_lrs(opt,scheduler,epoch)
        opt.zero_grad();sum(p.square().sum() for p in params).backward();opt.step()
    assert opt.param_groups[0]['lr']==1e-8
    clone=MatchedTailScheduler(opt,[21],.1,.1,10,'linear',arm='B',tail_start=2,tail_end=4)
    clone.load_state_dict(scheduler.state_dict());assert clone.last_epoch==4 and clone.arm=='B'
    print('V13_MATCHED_LR_CONTROL_PASS '+json.dumps({'device':device,'first40_optimizer_updates_identical':True,'A_matches_original100':True,'B_monotone_tail':True,'B_final':history[-1]['B'],'bias_ratio':2,'compressed_real_optimizer_tail':True}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');main(parser.parse_args().device)
