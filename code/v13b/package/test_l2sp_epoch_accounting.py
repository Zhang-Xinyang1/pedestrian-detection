"""Exercise the real training loop with different train/evaluation batch counts."""
import argparse
import contextlib
import importlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes=2
        self.image_encoder=nn.Module()
        self.image_encoder.linear=nn.Linear(4,768)
        self.image_encoder.proj=nn.Parameter(torch.randn(768,512)*.01)
        self.classifier=nn.Linear(768,2)

    def to(self, *args, **kwargs):return self

    def forward(self,x,**kwargs):
        z=self.image_encoder.linear(x)
        projected=z@self.image_encoder.proj
        if not self.training:return torch.cat([z,projected],dim=1)
        return [self.classifier(z)],[z,z,projected],projected


class Loader(list):batch_size=4


class Memory:
    def __init__(self):
        self.factor_valid=torch.ones(2,dtype=torch.bool)
        self.identity_valid=torch.ones(2,dtype=torch.bool)
        self.updates=0

    def update(self,*args,step_succeeded,**kwargs):
        assert step_succeeded
        self.updates+=1

    def batch_diagnostics(self,*args):return {'mean_condition':1.}
    def factor_metrics(self):return {}
    def state_sha256(self):return 'test-memory'


class Scaler:
    def scale(self,loss):return loss
    def step(self,optimizer):optimizer.step()
    def update(self):pass
    def get_scale(self):return 1.


class Evaluator:
    def __init__(self,*args,**kwargs):self.seen=0
    def reset(self):self.seen=0
    def update(self,*args):self.seen+=1
    def compute(self):
        assert self.seen==12
        return [1.]*10,1.,None,None,None,None,None


def main(package):
    sys.path[:0]=[str(package),str(package/'upstream')]
    torch.set_num_threads(2);torch.manual_seed(7)
    stage=importlib.import_module('scene_stage2_csf')
    tail=importlib.import_module('lr_tail_schedule')
    scene_prompt=importlib.import_module('scene_prompt')
    from starting_point_regularization import StartingPointRegularizer
    model=Model();anchor=StartingPointRegularizer(model)
    penalties=[]

    class ObservedAnchor:
        def __call__(self,network):
            value=anchor(network);penalties.append(float(value.detach()))
            return value
        def drift(self,network):return anchor.drift(network)

    data=torch.randn(4,4);labels=torch.tensor([0,0,1,1]);scene=torch.zeros(4,dtype=torch.long)
    train=Loader([(data,labels,scene,scene) for _ in range(6)])
    val=Loader([(data,labels,scene,scene,scene,['x']*4) for _ in range(12)])
    memory=Memory();optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    scheduler=NS(step=lambda:None,get_lr=lambda:[.01])
    original_to=torch.Tensor.to
    original_empty=torch.empty

    def cpu_to(tensor,*args,**kwargs):
        args=list(args)
        if args and (isinstance(args[0],int) or str(args[0]).startswith('cuda')):args[0]='cpu'
        if str(kwargs.get('device','')).startswith('cuda'):kwargs['device']='cpu'
        return original_to(tensor,*args,**kwargs)

    def cpu_empty(*args,**kwargs):
        if str(kwargs.get('device','')).startswith('cuda'):kwargs['device']='cpu'
        return original_empty(*args,**kwargs)

    def prototype(feature,*args,**kwargs):
        one=feature.new_tensor(1.)
        return feature.square().mean(),{k:one for k in ['valid_anchor_fraction','factor_target_fraction',
                    'identity_target_fraction','mean_candidate_identities']}

    with tempfile.TemporaryDirectory() as directory:
        cfg=NS(SOLVER=NS(STAGE2=NS(LOG_PERIOD=999,CHECKPOINT_PERIOD=99,EVAL_PERIOD=2,
                                 IMS_PER_BATCH=4,MAX_EPOCHS=2)),
               DATALOADER=NS(NUM_INSTANCE=2),TEST=NS(FEAT_NORM='yes'),OUTPUT_DIR=directory,
               MODEL=NS(SIE_CAMERA=False,SIE_VIEW=False,DIST_TRAIN=False,METRIC_LOSS_TYPE='triplet',NAME='toy'))
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(tail,'validate_optimizer_lrs',lambda opt,sched,epoch:
                {'epoch':epoch,'arm':'A','base_lr':.01,'optimizer_rates_match':True}))
            stack.enter_context(patch.object(torch.Tensor,'to',cpu_to))
            stack.enter_context(patch.object(torch,'empty',cpu_empty))
            stack.enter_context(patch.object(torch.cuda,'device_count',return_value=1))
            stack.enter_context(patch.object(torch.cuda,'synchronize'))
            stack.enter_context(patch.object(torch.cuda,'empty_cache'))
            stack.enter_context(patch.object(stage.amp,'GradScaler',Scaler))
            stack.enter_context(patch.object(stage.amp,'autocast',lambda **kw:contextlib.nullcontext()))
            stack.enter_context(patch.object(stage,'SupConLoss',lambda *args:None))
            stack.enter_context(patch.object(stage,'R1_mAP_eval',Evaluator))
            stack.enter_context(patch.object(stage,'prototype_comparison_loss',prototype))
            stack.enter_context(patch.object(stage,'sample_training_diagnostics',lambda *args,**kwargs:{}))
            stack.enter_context(patch.object(scene_prompt,'build_text_banks',return_value=None))
            stack.enter_context(patch.object(scene_prompt,'matched_scene_logits',lambda f,bank,s:model.classifier(f[:,:768])
                                            if f.shape[1]==768 else torch.stack([f.mean(1),-f.mean(1)],1)))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            summary,_=stage.do_train_stage2(cfg,model,None,train,val,optimizer,optimizer,scheduler,
                lambda scores,features,*args:scores[0].square().mean(),4,0,csf_memory=memory,csf_weight=.1,
                starting_point_regularizer=ObservedAnchor())
        assert memory.updates==12
        assert [r['batches'] for r in summary['epoch_history']]==[6,6]
        history=summary['l2sp_epoch_history']
        assert [r['batches'] for r in history]==[6,6], 'Evaluation loop overwrote training batch count'
        assert len(penalties)==12
        for index,row in enumerate(history):
            expected=sum(penalties[index*6:(index+1)*6])/6
            assert abs(row['mean_weighted_penalty']-expected)<1e-12,'Incorrect penalty denominator'
    print('L2SP_EPOCH_ACCOUNTING_OK actual_loop=True train_batches=6 eval_batches=12 penalty_mean=PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path,default=Path(__file__).resolve().parent)
    main(parser.parse_args().package)
