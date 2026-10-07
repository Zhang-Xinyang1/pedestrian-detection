"""Verify that observing the actual loss/gradients does not alter an update."""
import copy
from pathlib import Path
import sys
import tempfile

import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).parent/'upstream'))
from loss.make_loss import make_loss
from config import cfg as defaults
from l2sp_diagnostics import sample_training_diagnostics, PROBE_NAME, save_branch_diagnostics
from starting_point_regularization import StartingPointRegularizer


class ProbeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes=8
        self.image_encoder=nn.Module()
        self.image_encoder.transformer=nn.Module()
        blocks=[nn.Identity() for _ in range(12)]
        blocks[11]=nn.Module()
        blocks[11].mlp=nn.Module()
        blocks[11].mlp.c_proj=nn.Linear(16,768)
        self.image_encoder.transformer.resblocks=nn.ModuleList(blocks)
        self.image_encoder.proj=nn.Parameter(torch.randn(768,512)*.03)
        self.classifier=nn.Linear(768,8)
        self.classifier_proj=nn.Linear(512,8)

    def forward(self, x):
        z=self.image_encoder.transformer.resblocks[11].mlp.c_proj(x)
        p=z@self.image_encoder.proj
        return [self.classifier(z),self.classifier_proj(p)],[z,z,p],p


def main():
    assert torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.manual_seed(3)
    cfg=defaults.clone();cfg.defrost()
    cfg.MODEL.IF_LABELSMOOTH='on';cfg.MODEL.NO_MARGIN=True
    cfg.MODEL.METRIC_LOSS_TYPE='triplet';cfg.DATALOADER.SAMPLER='softmax_triplet'
    cfg.MODEL.ID_LOSS_WEIGHT=.25;cfg.MODEL.TRIPLET_LOSS_WEIGHT=1.;cfg.MODEL.I2T_LOSS_WEIGHT=1.
    cfg.freeze()
    loss_fn,_=make_loss(cfg,8)
    model=ProbeModel().cuda()
    reference=StartingPointRegularizer(model)
    with torch.no_grad():model.image_encoder.transformer.resblocks[11].mlp.c_proj.weight.add_(.003)
    twin=copy.deepcopy(model)
    twin_ref=StartingPointRegularizer(twin);twin_ref.load_state_dict(reference.state_dict(),strict=True)
    x=torch.randn(32,16,device='cuda');labels=torch.arange(8,device='cuda').repeat_interleave(4)
    bank=torch.randn(8,512,device='cuda')*.03
    output=[]
    for network,reg,observe in [(model,reference,True),(twin,twin_ref,False)]:
        with torch.autocast('cuda',dtype=torch.float16):
            scores,feat,projection=network(x)
            logits=projection@bank.t()
            base=loss_fn(scores,feat,labels,None,logits)
            proto=feat[1].float().square().mean()
        penalty=reg(network)
        if observe:
            with torch.autocast('cuda',dtype=torch.float16):
                row=sample_training_diagnostics(cfg,network,scores,feat,labels,logits,base,proto,.1,penalty,1,1)
            assert row['gradient_norms']['l2sp']>0
            assert all(p.grad is None for p in network.parameters())
        (base+.1*proto+penalty).backward()
        output.append({n:p.grad.detach().clone() for n,p in network.named_parameters()})
    assert output[0].keys()==output[1].keys()
    for name in output[0]:
        assert torch.equal(output[0][name],output[1][name]),name
    # Branch observations must use a common sample and exclude same-camera negatives.
    q=torch.randn(3,1280);g=q.clone()
    query=[('q0',1,0,0),('q1',2,0,1),('q2',3,5,2)]
    gallery=[('g0',1,1,0),('g1',2,1,1),('g2',3,6,2)]
    with tempfile.TemporaryDirectory() as directory:
        record=save_branch_diagnostics(directory,10,q,g,query,gallery)
        assert record['diagnostic_only'] and record['query_indices']==[0,1,2]
        for row in record['branch_metrics'].values():assert row['r1']==1 and row['ap']==1
    print('L2SP_DIAGNOSTICS_CUDA_OK unchanged_gradient=True branch_protocol=True',flush=True)


if __name__=='__main__':main()
