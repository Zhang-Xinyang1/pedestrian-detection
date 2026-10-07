"""Observations only: diagnostics never change loss, sampler, features or optimizer."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F


PROBE_NAME='image_encoder.transformer.resblocks.11.mlp.c_proj.weight'
INTERVAL=300


def sample_training_diagnostics(cfg, model, scores, features, labels, logits,
                                base_loss, proto_loss, proto_weight, reg_loss, epoch, batch):
    from loss.softmax_loss import CrossEntropyLabelSmooth
    from loss.triplet_loss import TripletLoss
    xent=CrossEntropyLabelSmooth(num_classes=model.num_classes) if cfg.MODEL.IF_LABELSMOOTH=='on' else F.cross_entropy
    triplet=TripletLoss() if cfg.MODEL.NO_MARGIN else TripletLoss(cfg.SOLVER.MARGIN)
    id_loss=sum(xent(score,labels) for score in scores)*cfg.MODEL.ID_LOSS_WEIGHT
    tri_loss=sum(triplet(feature,labels)[0] for feature in features)*cfg.MODEL.TRIPLET_LOSS_WEIGHT
    i2t_loss=xent(logits,labels)*cfg.MODEL.I2T_LOSS_WEIGHT
    reconstructed=id_loss+tri_loss+i2t_loss
    if not torch.allclose(base_loss.detach().float(),reconstructed.detach().float(),rtol=2e-3,atol=2e-3):
        raise ValueError('Diagnostic component sum differs from actual base objective')
    probe=dict(model.named_parameters())[PROBE_NAME]
    pieces={'identity':id_loss,'triplet':tri_loss,'i2t':i2t_loss,
            'prototype_weighted':proto_weight*proto_loss,'l2sp':reg_loss}
    gradients={}
    norms={}
    for name,loss in pieces.items():
        g=torch.autograd.grad(loss,probe,retain_graph=True,allow_unused=True)[0]
        if g is None:g=torch.zeros_like(probe)
        g=g.detach().float()
        if not bool(torch.isfinite(g).all()):raise ValueError('Nonfinite diagnostic gradient: '+name)
        gradients[name]=g
        norms[name]=float(g.norm().cpu())
    task=gradients['identity']+gradients['triplet']+gradients['i2t']+gradients['prototype_weighted']
    reg=gradients['l2sp']
    denominator=float(task.norm()*reg.norm())
    cosine=None if denominator==0 else float((task*reg).sum().cpu())/denominator
    norm768=features[1].detach().float().norm(dim=1)
    norm512=features[2].detach().float().norm(dim=1)
    fraction512=norm512.square()/(norm512.square()+norm768.square()).clamp_min(1e-12)
    return {'epoch':epoch,'batch':batch,'probe_parameter':PROBE_NAME,
        'weighted_losses':{name:float(loss.detach().float().cpu()) for name,loss in pieces.items()},
        'base_loss':float(base_loss.detach().float().cpu()),'gradient_norms':norms,
        'task_gradient_norm':float(task.norm().cpu()),'task_l2sp_gradient_cosine':cosine,
        'norm768_mean':float(norm768.mean().cpu()),'norm512_mean':float(norm512.mean().cpu()),
        'raw512_squared_norm_fraction_mean':float(fraction512.mean().cpu())}


def save_branch_diagnostics(output, epoch, qf, gf, query, gallery):
    """All-gallery diagnostic, fixed <=32 evenly spaced queries per scenario.

    Official overall evaluation remains full 6405 queries / 93609 gallery.
    Single-branch metrics are labelled subset diagnostics, never official scores.
    """
    from whu_metrics import retrieval_metrics
    indices=[]
    for scene in range(6):
        population=[i for i,r in enumerate(query) if 3*int(r[2]>=5)+r[3]==scene]
        count=min(len(population),32)
        if count:indices.extend(population[(k*len(population))//count] for k in range(count))
    indices=sorted(indices)
    sample_query=[query[i] for i in indices]
    records={}
    for name,columns in [('identity768',slice(0,768)),('projection512',slice(768,1280)),('concat1280',slice(0,1280))]:
        result=retrieval_metrics(qf[indices,columns],gf[:,columns],sample_query,gallery)
        records[name]={key:result[key] for key in ['ap','r1','r5','r10','scenes','valid_queries','skipped_queries','queries','gallery']}
    full=torch.cat([qf,gf])
    a=full[:,:768].norm(dim=1);b=full[:,768:].norm(dim=1)
    record={'epoch':epoch,'diagnostic_only':True,'selection':'up_to32_evenly_spaced_queries_per_scene',
        'query_indices':indices,'gallery_scope':'unchanged_full_gallery','protocol':'WHU_all_same_camera_excluded',
        'official_descriptor_dim':1280,'branch_metrics':records,
        'all_eval_feature_norms':{'mean768':float(a.mean()),'std768':float(a.std(unbiased=False)),
            'mean512':float(b.mean()),'std512':float(b.std(unbiased=False)),
            'raw512_squared_norm_fraction_mean':float((b.square()/(a.square()+b.square()).clamp_min(1e-12)).mean())}}
    path=Path(output)/f'branch_diagnostics_stage2_{epoch:03d}.json'
    if path.exists():raise ValueError('Branch diagnostics already exist')
    path.write_text(json.dumps(record,indent=2,sort_keys=True),encoding='utf-8')
    return record


def diagnostic_evaluator_class(base_class, dataset, output, eval_period):
    class DiagnosticEvaluator(base_class):
        def compute(self):
            values=super().compute()
            save_branch_diagnostics(output,self.counter*eval_period,values[-2],values[-1],
                                    dataset.query_meta,dataset.gallery_meta)
            return values
    return DiagnosticEvaluator
