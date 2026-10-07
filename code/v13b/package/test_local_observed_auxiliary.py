"""Exercise real local supervision, foreign-target masks, AMP and stop-gradient."""
import argparse
import copy
import json
import torch
from torch import nn
from torch.nn import functional as F
from local_observed_auxiliary import (LocalObservedAuxiliary, ObservedLocalMemory,
    initialize_auxiliary, transfer_coefficient, knowledge_transfer_loss, state_hashes)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(16,768)
        self.proj=nn.Parameter(torch.randn(768,512)*.01)
    def forward(self,x):
        tokens=self.linear(x)
        return tokens,tokens,tokens@self.proj


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes=8
        self.image_encoder=Encoder()
        self.classifier=nn.Linear(768,8,bias=False)
    def forward(self,x):
        _,tokens,projection=self.image_encoder(x)
        return torch.cat((tokens[:,0],projection[:,0]),dim=1)


def main(device):
    torch.set_num_threads(2);torch.manual_seed(23)
    if device=='cuda':assert torch.cuda.device_count()==1
    model=Model().to(device)
    cpu=torch.get_rng_state().clone()
    gpu=[s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    aux=initialize_auxiliary(model,1)
    assert torch.equal(cpu,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all()))
    if device=='cuda':
        mixed=Model()
        mixed.image_encoder.cuda()
        assert mixed.classifier.weight.device.type=='cpu'
        mixed_aux=initialize_auxiliary(mixed,1)
        assert next(mixed_aux.parameters()).device.type=='cuda'
        del mixed_aux,mixed
    assert not {id(p) for p in model.parameters()}.intersection(id(p) for p in aux.parameters())
    assert aux.specification()['trainable_parameters']<2000000
    assert not list(aux.memory.parameters())
    x=torch.randn(6,9,16,device=device)
    model.eval()
    baseline=model(x).detach().clone()
    aux.attach(model.image_encoder)
    assert torch.equal(baseline,model(x)) and aux._capture is None
    model.train()
    with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
        output=model(x)
    assert aux._capture is not None
    tokens=aux.pop_tokens()
    labels=torch.tensor([0,0,1,1,2,2],device=device)
    scenes=torch.tensor([0,1,0,1,0,1],device=device)
    student=model.classifier(output[:,:768].float())
    loss,features,stats=aux.compute(tokens,labels,scenes,student,1)
    assert stats['foreign_anchor_fraction']==0 and stats['transfer_coefficient']==0
    assert torch.isfinite(loss) and loss>0
    loss.backward()
    assert aux.shared_queries.grad is not None and aux.shared_queries.grad.abs().sum()>0
    assert model.image_encoder.linear.weight.grad.abs().sum()>0
    aux.begin_epoch();aux.record_batch(features.detach(),labels,scenes,True,stats)
    before=state_hashes(aux.memory)
    aux.record_batch(features.detach(),labels,scenes,False,stats)
    after=state_hashes(aux.memory)
    assert before['centers']==after['centers'] and before['mass']==after['mass']
    assert int(aux.memory.successful_updates)==1 and int(aux.memory.skipped_updates)==1
    assert aux.memory.mass.sum()==6
    assert not torch.count_nonzero(aux.memory.centers[:,3:])
    row=aux.end_epoch(1);assert row['batches']==2 and row['successful_updates']==1
    aux.detach()
    model.eval();assert torch.equal(baseline,model(x))

    # Known class directions make correct foreign-scene teachers deterministic.
    known=LocalObservedAuxiliary(8,768).to(device)
    with torch.no_grad():
        for head in known.classifiers:
            head.weight.zero_();head.weight[:8,:8].copy_(torch.eye(8,device=device))
    z=torch.zeros(12,4,768,device=device)
    ids=torch.arange(3,device=device).repeat_interleave(4)
    ss=torch.tensor([0,0,1,1]*3,device=device)
    z.scatter_(2,ids[:,None,None].expand(-1,4,1),1.)
    known.memory.update(z,ids,ss,True)
    anchor_ids=torch.arange(3,device=device)
    anchor_scenes=torch.zeros(3,dtype=torch.long,device=device)
    centers,weights,target,valid,foreign=known.memory.foreign_targets(anchor_ids,anchor_scenes)
    assert not foreign[:,0].any() and foreign[:,1].all() and not foreign[:,2:].any()
    assert valid.all() and not weights[:,0].any() and not weights[:,2:].any()
    test_tokens=torch.randn(3,8,768,device=device,requires_grad=True)
    test_student=torch.randn(3,8,device=device,requires_grad=True)
    active_loss,_,active=known.compute(test_tokens,anchor_ids,anchor_scenes,test_student,12)
    assert active['teacher_anchor_fraction']==1 and active['transfer_loss']>0
    assert active['cross_view_pairs']==0 and active['cross_modality_pairs']==3
    active_loss.backward()
    assert test_student.grad is not None and test_student.grad.abs().sum()>0
    assert torch.isfinite(test_tokens.grad).all()
    # Adding an observed aerial scene activates cross-view pairs, not fake views.
    known.memory.update(z[:2],torch.zeros(2,dtype=torch.long,device=device),torch.full((2,),3,device=device),True)
    _,_,joint=known.compute(torch.randn(1,8,768,device=device),torch.tensor([0],device=device),
                          torch.tensor([1],device=device),torch.randn(1,8,device=device,requires_grad=True),12)
    assert joint['cross_view_pairs']==1 and joint['joint_pairs']==1

    teacher=torch.randn(3,8,device=device,requires_grad=True)
    student=torch.randn(3,8,device=device,requires_grad=True)
    probability=teacher.softmax(1)
    kd=knowledge_transfer_loss(student,probability,torch.ones(3,dtype=torch.bool,device=device),anchor_scenes)
    grad_student,grad_teacher=torch.autograd.grad(kd,(student,teacher),allow_unused=True)
    assert grad_student.abs().sum()>0 and grad_teacher is None
    assert transfer_coefficient(5)==0 and abs(transfer_coefficient(6)-.01)<1e-12
    assert transfer_coefficient(10)==transfer_coefficient(40)==.05
    assert transfer_coefficient(40,True)==0
    empty=ObservedLocalMemory(8,768).to(device)
    empty.update(z[:2],torch.zeros(2,dtype=torch.long,device=device),torch.zeros(2,dtype=torch.long,device=device),True)
    _,w,_,v,_=empty.foreign_targets(torch.tensor([0],device=device),torch.tensor([0],device=device))
    assert not v.any() and not w.any()
    # Deserialization preserves hashes and observational eligibility.
    clone=LocalObservedAuxiliary(8,768).to(device);clone.load_state_dict(known.state_dict(),strict=True)
    assert state_hashes(clone)==state_hashes(known)
    print('LOCAL_OBSERVED_AUXILIARY_OK '+json.dumps(dict(device=device,raw1280_unchanged=True,
        cold_start=True,foreign_only=True,no_missing_view=True,amp_skip_memory=True,
        local_encoder_gradient=True,active_KD_student_gradient=True,teacher_stop_gradient=True,
        RNG_preserved=True,parameter_roundtrip=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    main(parser.parse_args().device)
