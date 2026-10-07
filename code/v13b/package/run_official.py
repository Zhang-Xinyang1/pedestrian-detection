"""Run pinned official CLIP-ReID with a WHU dataset/evaluation adapter."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml

# Use named shared-memory files for batches passed from workers to the trainer.
torch.multiprocessing.set_sharing_strategy('file_system')

PACKAGE=Path(__file__).resolve().parent
sys.path.insert(0,str(PACKAGE/'upstream'))
from config import cfg as defaults
from yacs.config import CfgNode as CN
from datasets import make_dataloader_clipreid as loaders
from datasets.sampler import RandomIdentitySampler
from model import make_model_clipreid as models
from solver.make_optimizer_prompt import make_optimizer_1stage,make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
import scene_stage1 as stage1
import scene_stage2 as stage2
from utils.logger import setup_logger
from whu_adapter import WHUMARS,evaluator_class
from scene_prompt import ScenePromptModel, attach_scene_prompts


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def fingerprints(model):
    return {n:hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for n,p in model.named_parameters()}


def config(args):
    raw=yaml.safe_load((PACKAGE/'upstream/configs/person/vit_clipreid.yml').read_text())
    raw.pop('DATASETS',None)  # Official example has only comments under this key.
    cfg=defaults.clone(); cfg.merge_from_other_cfg(CN(raw))
    cfg.DATASETS.NAMES='whumars'; cfg.DATASETS.ROOT_DIR=str(args.data_root)
    cfg.OUTPUT_DIR=str(args.output); cfg.SOLVER.SEED=1
    cfg.MODEL.DIST_TRAIN=False; cfg.MODEL.SIE_CAMERA=False; cfg.MODEL.SIE_VIEW=False
    cfg.DATALOADER.NUM_WORKERS=2
    cfg.TEST.IMS_PER_BATCH=32
    cfg.SOLVER.STAGE2.MAX_EPOCHS=120
    cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD=120
    cfg.SOLVER.STAGE2.STEPS=[60,100]
    cfg.SOLVER.STAGE2.EVAL_PERIOD=10
    # Expose only the image-to-text term for a controlled modality ablation.
    i2t_weight=float(getattr(args,'i2t_weight',1.0))
    if i2t_weight not in (0.25,0.5,1.0): raise ValueError('Unsupported I2T weight')
    cfg.MODEL.I2T_LOSS_WEIGHT=i2t_weight
    if args.local_smoke:
        cfg.DATALOADER.NUM_WORKERS=0; cfg.DATALOADER.NUM_INSTANCE=2
        cfg.SOLVER.STAGE1.IMS_PER_BATCH=4; cfg.SOLVER.STAGE2.IMS_PER_BATCH=4; cfg.TEST.IMS_PER_BATCH=4
        cfg.SOLVER.STAGE1.MAX_EPOCHS=1; cfg.SOLVER.STAGE2.MAX_EPOCHS=4
        cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD=1; cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD=4
        cfg.SOLVER.STAGE1.LOG_PERIOD=1; cfg.SOLVER.STAGE2.LOG_PERIOD=1; cfg.SOLVER.STAGE2.EVAL_PERIOD=4
    cfg.freeze(); return cfg


def small_dataset(dataset,six_scenes=False):
    # 17 image cache entries avoids the upstream divisible-N empty final batch.
    chosen=[]
    for pid in range(4):
        for mod in range(3):
            chosen.append(next(r for r in dataset.train_meta if r[1]==pid and r[3]==mod))
        chosen.append(next(r for r in dataset.train_meta if r[1]==pid and r not in chosen))
    chosen.extend([r for r in dataset.train_meta if r[1]==0][4:5])
    if six_scenes:
        groups={}
        for row in dataset.train_meta:
            groups.setdefault(row[1],{}).setdefault(3*int(row[2]>=5)+row[3],row)
        ids=[pid for pid,scenes in groups.items() if len(scenes)==6][:4]
        assert len(ids)==4
        chosen=[groups[pid][scene] for pid in ids for scene in range(6)]
        chosen.append(next(r for r in dataset.train_meta if r[1]==ids[0] and r not in chosen))
    dataset.train_meta=chosen
    dataset.train=[(p,y,c,int(c>=5)) for p,y,c,m in chosen]
    test_ids=sorted({r[1] for r in dataset.query_meta})[:4]
    def select(rows):
        groups={}
        for r in rows:
            if r[1] in test_ids: groups.setdefault((r[1],r[3],int(r[2]>=5)),r)
        return list(groups.values())
    dataset.query_meta=select(dataset.query_meta); dataset.gallery_meta=select(dataset.gallery_meta)
    dataset.query=[(p,y,c,int(c>=5)) for p,y,c,m in dataset.query_meta]
    dataset.gallery=[(p,y,c,int(c>=5)) for p,y,c,m in dataset.gallery_meta]
    return dataset


class CountingLoader:
    def __init__(self,loader): self.loader=loader; self.counts=[]
    def __getattr__(self,name): return getattr(self.loader,name)
    def __len__(self): return len(self.loader)
    def __iter__(self):
        count=0
        for row in self.loader:
            count+=1; yield row
        self.counts.append(count)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--weights',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--prompt-mode',choices=['neutral','modality','view','both'],required=True)
    p.add_argument('--variant',choices=['control','quality_gate'],default='quality_gate',
                   help='control keeps the modality baseline; quality_gate learns a sample-wise scale for the 512-D branch')
    p.add_argument('--i2t-weight',type=float,choices=[0.25,0.5,1.0],default=1.0)
    p.add_argument('--xmodal-weight',type=float,choices=[0.0,0.05,0.1,0.2],default=0.0,
                   help='Cross-modality contrastive loss on the existing 768-D identity branch')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--local-smoke',action='store_true')
    p.add_argument('--smoke-six-scenes',action='store_true',help='Local only: 25 images covering all six scenes')
    a=p.parse_args()
    if a.prompt_mode!='modality': raise ValueError('This I2T experiment is modality-only')
    if a.smoke_six_scenes and not a.local_smoke:
        raise ValueError('--smoke-six-scenes requires --local-smoke')
    if a.output.exists(): raise ValueError('Output exists; never overwrite')
    provenance=json.loads((PACKAGE/'PROVENANCE.json').read_text())
    for name,digest in provenance['runtime_upstream_hashes'].items():
        if sha(PACKAGE/'upstream'/name)!=digest: raise ValueError('Official runtime changed: '+name)
    if sha(a.weights)!=provenance['clip_weights_sha256']: raise ValueError('Wrong CLIP weights')
    dataset=WHUMARS(a.data_root)
    assert (dataset.num_train_pids,len(dataset.train),len(dataset.query),len(dataset.gallery))==(500,92133,6405,93609)
    assert dataset.train_signature==provenance['train_metadata_sha256']
    cfg=config(a)
    # Validate a real official PK batch plan without changing the training RNG.
    random.seed(1); np.random.seed(1)
    sampler=RandomIdentitySampler(dataset.train,64,4)
    indices=list(iter(sampler)); assert len(indices)%64==0
    for start in range(0,len(indices),64):
        labels=[dataset.train[i][1] for i in indices[start:start+64]]
        assert len(set(labels))==16 and all(labels.count(y)==4 for y in set(labels))
    info=dict(classes=500,train=92133,query=6405,gallery=93609,
              stage1_epochs=cfg.SOLVER.STAGE1.MAX_EPOCHS,stage2_epochs=cfg.SOLVER.STAGE2.MAX_EPOCHS,
              stage1_cache_batches=math.ceil(92133/64),stage2_seed1_plan_batches=len(indices)//64,
              feature_dim=1280,seed=1,local_smoke=a.local_smoke,
              train_metadata_sha256=dataset.train_signature,upstream_commit=provenance['commit'])
    info['prompt_mode']=a.prompt_mode
    info['variant']=a.variant
    info['i2t_loss_weight']=float(a.i2t_weight)
    info['xmodal_loss_weight']=float(a.xmodal_weight)
    info['xmodal_temperature']=0.07
    info['data_protocol']='unchanged_official_P16K4'
    print('SCENE_PROMPT_INPUTS_OK '+json.dumps(info),flush=True)
    if a.check_only:
        print('SCENE_PROMPT_PREFLIGHT_OK (no output writes or training)',flush=True); return
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('This official-recipe baseline requires exactly one visible CUDA GPU')
    torch.set_num_threads(4)
    # Match official seeding settings and retain the upstream two-stage code.
    torch.manual_seed(1); torch.cuda.manual_seed_all(1); np.random.seed(1); random.seed(1)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=True
    a.output.mkdir(parents=True,exist_ok=False)
    if a.local_smoke: dataset=small_dataset(dataset,a.smoke_six_scenes)
    # Only training loader field4 carries scene IDs. SIE is disabled, and eval
    # metadata keeps the original physical view/camera protocol.
    dataset.train=[(p,y,c,3*int(c>=5)+m) for p,y,c,m in dataset.train_meta]
    manifest=dict(info=info,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                  config=yaml.safe_load(cfg.dump()),provenance=provenance,
                  adapter_sources={name:sha(PACKAGE/name) for name in ['run_official.py','whu_adapter.py','whu_metrics.py','scene_prompt.py','quality_gate.py','scene_stage1.py','scene_stage2.py','xmodal_alignment.py']},
                  actual_run_counts=dict(train=len(dataset.train),query=len(dataset.query),gallery=len(dataset.gallery)),
                  train_scene_counts={str(s):sum(r[3]==s for r in dataset.train) for s in range(6)},
                  eval_query=[str(Path(r[0]).relative_to(a.data_root)) for r in dataset.query_meta],
                  eval_gallery=[str(Path(r[0]).relative_to(a.data_root)) for r in dataset.gallery_meta],
                  torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    setup_logger('transreid',str(a.output),if_train=True)
    loaders.__factory['whumars']=lambda root:dataset
    train2,train1,val,nq,nclasses,ncams,nviews=loaders.make_dataloader(cfg)
    train2=CountingLoader(train2)
    def offline_download(url,*args,**kwargs):
        if url!=models.clip._MODELS['ViT-B-16']: raise ValueError('Unexpected weight request')
        return str(a.weights)
    models.clip._download=offline_download
    model=ScenePromptModel(nclasses,ncams,nviews,cfg,quality_gate=(a.variant=='quality_gate'))
    attach_scene_prompts(model,a.weights,a.prompt_mode)
    manifest['prompt_spec']=model.prompt_learner.specification()
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    assert model.num_classes==500 and model.in_planes==768 and model.in_planes_proj==512
    if a.variant=='quality_gate':
        with torch.no_grad():
            gate=torch.ones(2,3,32,32,device=next(model.parameters()).device)
            alpha=model.reliability_gate(gate)
            assert torch.equal(alpha,torch.ones_like(alpha)), 'gate must start as identity'
    print('SCENE_PROMPT_MODEL_OK classes=500 feature_dim=1280 SIE=False OLP=False',flush=True)
    loss_fn,center=make_loss(cfg,num_classes=nclasses)
    initial=fingerprints(model)
    (a.output/'initial_parameters.json').write_text(json.dumps(initial,indent=2),encoding='utf8')
    opt1=make_optimizer_1stage(cfg,model)
    ids={id(p):n for n,p in model.named_parameters()}
    names1=[ids[id(p)] for g in opt1.param_groups for p in g['params']]
    assert names1==['prompt_learner.cls_ctx']
    count1=[0]
    hook1=opt1.register_step_post_hook(lambda *_:count1.__setitem__(0,count1[0]+1))
    sch1=create_scheduler(opt1,num_epochs=cfg.SOLVER.STAGE1.MAX_EPOCHS,lr_min=cfg.SOLVER.STAGE1.LR_MIN,
                          warmup_lr_init=cfg.SOLVER.STAGE1.WARMUP_LR_INIT,warmup_t=cfg.SOLVER.STAGE1.WARMUP_EPOCHS)
    print('SCENE_PROMPT_STAGE1_START',flush=True)
    stage1.do_train_stage1(cfg,model,train1,opt1,sch1,0)
    hook1.remove()
    after1=fingerprints(model)
    changed1=[n for n in initial if initial[n]!=after1[n]]
    assert changed1==['prompt_learner.cls_ctx'] and count1[0]>0
    stage1_record=dict(changed=changed1,optimizer_parameters=names1,optimizer_steps=count1[0],
                       planned_steps=math.ceil(len(dataset.train)/cfg.SOLVER.STAGE1.IMS_PER_BATCH)*cfg.SOLVER.STAGE1.MAX_EPOCHS,
                       final_parameters=after1)
    (a.output/'stage1_audit.json').write_text(json.dumps(stage1_record,indent=2),encoding='utf8')
    print('SCENE_PROMPT_STAGE1_DONE '+json.dumps({k:v for k,v in stage1_record.items() if k!='final_parameters'}),flush=True)
    opt2,opt_center=make_optimizer_2stage(cfg,model,center)
    names2=[ids[id(p)] for g in opt2.param_groups for p in g['params']]
    assert not any(n.startswith(('prompt_learner.','text_encoder.')) for n in names2)
    count2=[0]
    hook2=opt2.register_step_post_hook(lambda *_:count2.__setitem__(0,count2[0]+1))
    sch2=WarmupMultiStepLR(opt2,cfg.SOLVER.STAGE2.STEPS,cfg.SOLVER.STAGE2.GAMMA,
                          cfg.SOLVER.STAGE2.WARMUP_FACTOR,cfg.SOLVER.STAGE2.WARMUP_ITERS,cfg.SOLVER.STAGE2.WARMUP_METHOD)
    stage2.R1_mAP_eval=evaluator_class(dataset,a.output,cfg.SOLVER.STAGE2.EVAL_PERIOD,cfg.SOLVER.STAGE2.MAX_EPOCHS,a.local_smoke)
    print('SCENE_PROMPT_STAGE2_START',flush=True)
    xmodal_summary=stage2.do_train_stage2(
        cfg,model,center,train2,val,opt2,opt_center,sch2,loss_fn,nq,0,
        xmodal_weight=a.xmodal_weight,xmodal_temperature=0.07)
    if a.xmodal_weight>0:
        assert xmodal_summary['batches']>0
        assert xmodal_summary['mean_valid_anchor_fraction']>0
        assert xmodal_summary['mean_loss']>0
    hook2.remove()
    after2=fingerprints(model)
    changed2=[n for n in after1 if after1[n]!=after2[n]]
    assert count2[0]>0 and any(n.startswith('image_encoder.') for n in changed2)
    assert all(not n.startswith(('text_encoder.','prompt_learner.')) for n in changed2)
    assert len(train2.counts)==cfg.SOLVER.STAGE2.MAX_EPOCHS
    stage2_record=dict(changed=changed2,optimizer_parameters=names2,optimizer_steps=count2[0],
                       actual_batches_per_epoch=train2.counts,planned_steps=sum(train2.counts),xmodal_alignment=xmodal_summary,final_parameters=after2)
    (a.output/'stage2_audit.json').write_text(json.dumps(stage2_record,indent=2),encoding='utf8')
    print('SCENE_PROMPT_STAGE2_DONE '+json.dumps({k:v for k,v in stage2_record.items() if k not in ['final_parameters','optimizer_parameters','changed']}),flush=True)
    print('SCENE_PROMPT_TRAIN_DONE',flush=True)


if __name__=='__main__': main()
