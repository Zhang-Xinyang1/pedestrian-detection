"""Behavioral checks for permutation, rejection, identity margin and gradient."""
import argparse
import json
import torch
from torch.nn import functional as F
from local_observed_auxiliary import ObservedLocalMemory
from partial_local_matching import soft_partial_plan, observed_correspondence, matched_local_alignment


def main(device):
    torch.set_num_threads(2)
    torch.manual_seed(113)
    if device=='cuda':assert torch.cuda.device_count()==1
    m=ObservedLocalMemory(4,16).to(device)
    basis=torch.eye(16,device=device)
    with torch.no_grad():
        m.centers[0,1]=basis[:4];m.centers[1,1]=basis[4:8]
        m.mass[:2,1]=8
    labels=torch.tensor([0,1],device=device);scenes=torch.zeros(2,dtype=torch.long,device=device)
    permutation=[2,0,3,1]
    features=torch.stack([basis[permutation],basis[4:8]]).clone().requires_grad_()
    _,weights,_,_,foreign=m.foreign_targets(labels,scenes)
    plan,stats=observed_correspondence(features,labels,scenes,m,weights,foreign)
    assert plan[0,1].argmax(1).tolist()==permutation
    assert stats['matched_slot_fraction']>.9 and not plan.requires_grad
    assert (plan.sum(-1)<=1.00001).all() and (plan.sum(-2)<=1.00001).all()
    assert not plan[:,0].any() and not plan[:,2:].any()
    # A source/anchor without any observed other identity has no margin evidence.
    ids=labels[:1];ss=scenes[:1]
    _,w,_,_,f=m.foreign_targets(ids,ss)
    alone,_=observed_correspondence(features[:1],ids,ss,m,w,f)
    assert not alone.any()
    # Unseen/uninformative slot can take the unmatched option.
    partly=features.detach().clone();partly[0,3]=basis[12]
    partial,ps=observed_correspondence(partly,labels,scenes,m,weights,foreign)
    assert not partial[0,1,3].any() and partial[0,1,:3].sum()>2.
    # Identical-looking other identities make raw cosine alone insufficient.
    original=m.centers[1,1].clone()
    m.centers[1,1]=m.centers[0,1]
    ambiguous,ambstats=observed_correspondence(features,labels,scenes,m,weights,foreign)
    assert not ambiguous[0].any()
    m.centers[1,1]=original
    live=(features.detach()+.05*torch.randn_like(features)).requires_grad_()
    loss,ls=matched_local_alignment(live,labels,scenes,m,weights,foreign)
    assert torch.isfinite(loss) and loss>0 and ls['matched_slot_mass']>0
    loss.backward();assert torch.isfinite(live.grad).all() and live.grad.abs().sum()>0
    assert all(p.grad is None for p in m.parameters())
    zero=ObservedLocalMemory(4,16).to(device)
    _,w,_,_,f=zero.foreign_targets(labels,scenes)
    xx=torch.randn_like(features,requires_grad=True)
    empty,es=matched_local_alignment(xx,labels,scenes,zero,w,f)
    assert empty==0 and es['matched_slot_mass']==0
    empty.backward();assert not xx.grad.any()
    # Simultaneous cross-view and cross-modality observed sources are supported.
    m.centers[:2,4]=m.centers[:2,1];m.mass[:2,4]=8
    _,w,_,_,f=m.foreign_targets(labels,scenes)
    joint,jstats=observed_correspondence(features,labels,scenes,m,w,f)
    assert jstats['matched_joint_pairs']==2 and jstats['matched_cross_view_pairs']==2
    # CUDA production-sized finite operations without a second backbone forward.
    full=ObservedLocalMemory(500,768).to(device)
    idx=torch.arange(16,device=device).repeat_interleave(4)
    sc=torch.arange(64,device=device)%6
    z=F.normalize(torch.randn(16,4,768,device=device),dim=-1)
    with torch.no_grad():
        full.centers[:16]=z[:,None].expand(-1,6,-1,-1)
        full.mass[:16]=16
    x=(z[idx]+.01*torch.randn(64,4,768,device=device)).requires_grad_()
    _,w,_,_,f=full.foreign_targets(idx,sc)
    value,details=matched_local_alignment(x,idx,sc,full,w,f)
    assert torch.isfinite(value) and details['matched_slot_mass']>0
    value.backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    print('PARTIAL_LOCAL_MATCHING_OK '+json.dumps(dict(device=device,query_permutation=True,
        unmatched_slot=True,identity_margin_rejection=True,missing_source_rejection=True,
        detached_correspondence=True,encoder_gradient=True,capacity=True,production64x500_shape=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    main(p.parse_args().device)
