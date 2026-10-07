"""CPU-only validation of two-stage official-method WHU artifacts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch

PACKAGE=Path(__file__).resolve().parent


def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,msg):
    if not ok: raise ValueError(msg)


def finite(value):
    if isinstance(value,dict):
        for v in value.values(): finite(v)
    elif isinstance(value,list):
        for v in value: finite(v)
    elif isinstance(value,float): require(math.isfinite(value),'Nonfinite metric')


def verify(directory,smoke=False):
    m=read(directory/'manifest.json'); cfg=m['config']; provenance=read(PACKAGE/'PROVENANCE.json')
    from scene_prompt import templates,clip
    mode=m['args']['prompt_mode']
    i2t_weight=float(m['args']['i2t_weight'])
    variant=m['args'].get('variant','control')
    xmodal_weight=float(m['args'].get('xmodal_weight',0.0))
    require(variant in ('control','quality_gate'),'Unknown variant')
    require(i2t_weight in (0.25,0.5,1.0),'Unsupported I2T loss weight')
    require(xmodal_weight in (0.0,0.05,0.1,0.2),'Unsupported cross-modality weight')
    require(mode=='modality','This I2T experiment is modality-only')
    require(m['info']['i2t_loss_weight']==i2t_weight,'Weight metadata mismatch')
    require(m['info']['xmodal_loss_weight']==xmodal_weight,'Cross-modality metadata mismatch')
    require(m['info']['data_protocol']=='unchanged_official_P16K4','Data/sampler protocol changed')
    require(mode in ('neutral','modality','view','both'),'Unknown prompt mode')
    require(m['prompt_spec']['templates']==templates(mode),'Prompt wording differs')
    require(m['prompt_spec']['identity_tokens']==4 and m['prompt_spec']['shared_across_scenes'],'Wrong identity token sharing')
    require(m['args']['local_smoke']==smoke,'Smoke/production mismatch')
    require(m['provenance']==provenance,'Source provenance differs')
    required_sources={'run_official.py','whu_adapter.py','whu_metrics.py','scene_prompt.py','scene_stage1.py','scene_stage2.py','quality_gate.py','xmodal_alignment.py'}
    require(set(m['adapter_sources'])==required_sources,'Incomplete adapter provenance')
    for name,digest in m['adapter_sources'].items(): require(sha(PACKAGE/name)==digest,'Adapter source differs: '+name)
    for name,digest in provenance['runtime_upstream_hashes'].items(): require(sha(PACKAGE/'upstream'/name)==digest,'Upstream differs: '+name)
    require(sum(m['train_scene_counts'].values())==m['actual_run_counts']['train'],'Scene count mismatch')
    if m['args'].get('smoke_six_scenes') or not smoke:
        require(all(m['train_scene_counts'][str(s)]>0 for s in range(6)),'Training misses a scene')
    require(m['info']['classes']==500 and m['info']['train_metadata_sha256']==provenance['train_metadata_sha256'],'Data scope differs')
    require(cfg['SOLVER']['SEED']==1 and not cfg['MODEL']['SIE_CAMERA'] and not cfg['MODEL']['SIE_VIEW'],'Seed/SIE differs')
    require(cfg['MODEL']['NAME']=='ViT-B-16' and cfg['MODEL']['STRIDE_SIZE']==[16,16],'Backbone/OLP differs')
    require(cfg['TEST']['NECK_FEAT']=='before' and cfg['TEST']['FEAT_NORM']=='yes' and not cfg['TEST']['RE_RANKING'],'Retrieval protocol differs')
    require(cfg['MODEL']['ID_LOSS_WEIGHT']==.25 and cfg['MODEL']['TRIPLET_LOSS_WEIGHT']==1. and cfg['MODEL']['I2T_LOSS_WEIGHT']==i2t_weight,'Loss weights differ')
    require(m['info'].get('variant',variant)==variant,'Variant metadata mismatch')
    epochs=[1,4] if smoke else [120,120]
    for stage,epoch in zip(['STAGE1','STAGE2'],epochs): require(cfg['SOLVER'][stage]['MAX_EPOCHS']==epoch,'Epoch budget differs')
    require(cfg['SOLVER']['STAGE2']['STEPS']==[60,100],'Stage2 milestones differ')
    if not smoke:
        require(cfg['SOLVER']['STAGE1']['IMS_PER_BATCH']==cfg['SOLVER']['STAGE2']['IMS_PER_BATCH']==64,'Batch differs')
        require(cfg['DATALOADER']['NUM_INSTANCE']==4,'P/K differs')
        require(m['actual_run_counts']==dict(train=92133,query=6405,gallery=93609),'Incomplete production split')
    initial=read(directory/'initial_parameters.json')
    reports=[]
    previous=initial
    prompt_buffers=None
    for stage,epoch in enumerate(epochs,1):
        audit=read(directory/f'stage{stage}_audit.json')
        require(0<audit['optimizer_steps']<=audit['planned_steps'],'Missing/excess optimizer updates')
        state=torch.load(directory/(f'ViT-B-16_stage1_{epoch}.pth' if stage==1 else f'ViT-B-16_{epoch}.pth'),map_location='cpu',weights_only=False)
        require(state['classifier.weight'].shape==(500,768) and state['classifier_proj.weight'].shape==(500,512),'Classifier shape differs')
        require(state['prompt_learner.cls_ctx'].shape==(500,4,512),'Prompt shape differs')
        require(state['prompt_learner.scene_token_ids'].shape==(6,77),'Missing scene tokens')
        require(state['prompt_learner.scene_prefix'].shape==(6,5,512),'Wrong prompt prefix')
        require(state['prompt_learner.scene_suffix'].shape==(6,68,512),'Wrong prompt suffix')
        require(torch.equal(state['prompt_learner.scene_token_ids'],clip.tokenize(templates(mode))),'Checkpoint prompt text differs')
        buffers={k:v for k,v in state.items() if k.startswith('prompt_learner.') and k!='prompt_learner.cls_ctx'}
        if prompt_buffers is not None:
            require(all(torch.equal(v,prompt_buffers[k]) for k,v in buffers.items()),'Stage2 modified fixed prompt buffers')
        prompt_buffers=buffers
        for key,tensor in state.items(): require(bool(torch.isfinite(tensor).all()),'Nonfinite checkpoint tensor '+key)
        require(set(previous)==set(audit['final_parameters']),'Parameter schema changed')
        for key,digest in audit['final_parameters'].items():
            require(hashlib.sha256(state[key].contiguous().numpy().tobytes()).hexdigest()==digest,'Checkpoint/audit fingerprint differs: '+key)
        changed=[k for k in previous if previous[k]!=audit['final_parameters'][k]]
        require(changed==audit['changed'],'Changed-parameter audit differs')
        if stage==1:
            require(changed==audit['optimizer_parameters']==['prompt_learner.cls_ctx'],'Stage1 changed non-prompt parameter')
            batch=cfg['SOLVER']['STAGE1']['IMS_PER_BATCH']
            require(audit['planned_steps']==math.ceil(m['actual_run_counts']['train']/batch)*epoch,'Stage1 budget differs')
        else:
            require(any(k.startswith('image_encoder.') for k in changed),'Visual backbone did not update')
            require('classifier.weight' in changed and 'classifier_proj.weight' in changed,'Classifier did not update')
            require(not any(k.startswith(('text_encoder.','prompt_learner.')) for k in changed),'Stage2 changed fixed text/prompt')
            if variant=='quality_gate':
                require(any(k.startswith('reliability_gate.') for k in changed),'Quality gate did not update')
            else:
                require(not any(k.startswith('reliability_gate.') for k in changed),'Control updated quality gate')
            alignment=audit.get('xmodal_alignment')
            require(alignment is not None and alignment['weight']==xmodal_weight,'Missing cross-modality audit')
            require(alignment['temperature']==0.07 and alignment['feature_branch']=='identity_768','Wrong cross-modality contract')
            require(alignment['data_and_sampler_unchanged'],'Data/sampler audit differs')
            if xmodal_weight>0:
                require(alignment['batches']>0 and alignment['mean_loss']>0,'Cross-modality loss inactive')
                require(alignment['mean_valid_anchor_fraction']>0,'No cross-modality positives observed')
            else:
                require(alignment['batches']==0,'Zero-weight baseline computed cross-modality loss')
            require(len(audit['actual_batches_per_epoch'])==epoch and all(v>0 for v in audit['actual_batches_per_epoch']),'Incomplete stage2 epochs')
            require(sum(audit['actual_batches_per_epoch'])==audit['planned_steps'],'Stage2 budget differs')
        reports.append(dict(stage=stage,epochs=epoch,optimizer_steps=audit['optimizer_steps'],
                            batches=audit['planned_steps'],amp_skipped_steps=audit['planned_steps']-audit['optimizer_steps'],changed_parameters=len(changed)))
        previous=audit['final_parameters']; del state
    period=cfg['SOLVER']['STAGE2']['EVAL_PERIOD']
    for epoch in range(period,epochs[1]+1,period):
        r=read(directory/f'eval_stage2_{epoch:03d}.json'); finite(r)
        require(r['epoch']==epoch and r['local_smoke']==smoke and r['feature_dim']==1280,'Wrong evaluation checkpoint')
        require(r['protocol']=='WHU_all_same_camera_excluded','Wrong evaluation filter')
        require(r['queries']==len(m['eval_query'])==m['actual_run_counts']['query'],'Query count differs')
        require(r['gallery']==len(m['eval_gallery'])==m['actual_run_counts']['gallery'],'Gallery count differs')
        require(r['valid_queries']+r['skipped_queries']==r['queries'],'Query accounting differs')
        if not smoke: require(r['valid_queries']==6405 and r['skipped_queries']==0,'Incomplete valid query scope')
        require(set(r['scenes'])=={str(i) for i in range(6)},'Missing six-scene metrics')
        for key,size in [('scene',6),('modality',3)]:
            mat=r['matrices'][key]
            require(len(mat)==size and all(len(row)==size for row in mat),'Matrix shape differs')
    print('SCENE_PROMPT_ARTIFACTS_OK '+json.dumps(dict(local_smoke=smoke,stages=reports,
          final_map_percent=100*r['ap'],final_rank1_percent=100*r['r1'],feature_dim=1280,xmodal_weight=xmodal_weight)))
    return reports


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('directory',type=Path)
    p.add_argument('--local-smoke',action='store_true'); a=p.parse_args()
    torch.set_num_threads(4); verify(a.directory,a.local_smoke)
