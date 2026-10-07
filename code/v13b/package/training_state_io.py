"""Atomically save the latest full epoch-boundary state; no losses/RNG changed."""
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch

def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),
        'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(rng):
    random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch_cpu'])
    if rng['torch_cuda']:torch.cuda.set_rng_state_all(rng['torch_cuda'])

def save_training_state(path,model,optimizer,optimizer_center,scheduler,scaler,global_memory,local_auxiliary,epoch,ledger,cfg):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    payload={'schema':'V13_full_epoch_boundary_v1','epoch':epoch,'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),'optimizer_center':optimizer_center.state_dict(),
        'scheduler':scheduler.state_dict(),'AMP':scaler.state_dict(),'rng':capture_rng(),
        'global_memory':global_memory.state_dict(),'local_auxiliary':local_auxiliary.state_dict(),
        'ledger':ledger,'config':cfg.dump(),'lr_tail_arm':scheduler.arm,
        'stage1_source_required':'approved immutable Stage1; visual anchor reconstructed from that exact file',
        'sampler_scope':'epoch boundary, before next epoch sampler draws; worker seeds derive from saved torch RNG'}
    torch.save(payload,temporary);os.replace(temporary,path)
    return path

def restore_training_state(payload,model,optimizer,optimizer_center,scheduler,scaler,global_memory,local_auxiliary):
    if payload['schema']!='V13_full_epoch_boundary_v1' or payload['lr_tail_arm']!=scheduler.arm:
        raise ValueError('Training state arm/schema differs')
    model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer'])
    optimizer_center.load_state_dict(payload['optimizer_center']);scheduler.load_state_dict(payload['scheduler'])
    scaler.load_state_dict(payload['AMP']);global_memory.load_state_dict(payload['global_memory'],strict=True)
    local_auxiliary.load_state_dict(payload['local_auxiliary'],strict=True);restore_rng(payload['rng'])
    return payload['epoch'],payload['ledger']

def verify_training_state(directory,manifest,audit):
    path=Path(directory)/'training_state_latest.pth'
    if not path.is_file():raise ValueError('Full latest training state missing')
    payload=torch.load(str(path),map_location='cpu',weights_only=False,mmap=True)
    if payload['schema']!='V13_full_epoch_boundary_v1' or payload['epoch']!=audit['final_epoch']:
        raise ValueError('Full training state final epoch differs')
    if payload['lr_tail_arm']!=manifest['info']['lr_tail_arm']:raise ValueError('Full state LR arm differs')
    for name,digest in audit['final_parameters'].items():
        actual=hashlib.sha256(payload['model'][name].contiguous().numpy().tobytes()).hexdigest()
        if actual!=digest:raise ValueError('Full state main parameter differs: '+name)
    ledger=payload['ledger']
    if ledger['actual_batches_per_epoch']!=audit['actual_batches_per_epoch'] or ledger['successful_updates']!=audit['optimizer_steps']:
        raise ValueError('Full state update ledger differs')
    if ledger['learning_rate_history']!=audit['csf']['learning_rate_history']:raise ValueError('Full state LR ledger differs')
    if payload['scheduler']['last_epoch']!=audit['final_epoch'] or not payload['AMP'] or not payload['optimizer']['state']:
        raise ValueError('Optimizer/scheduler/AMP state missing')
    if int(payload['global_memory']['successful_updates'])!=audit['optimizer_steps']:
        raise ValueError('Full state global bank updates differ')
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    result={'status':'PASS','file':path.name,'sha256':digest,'epoch':payload['epoch'],
        'components':['model','optimizer','optimizer_center','scheduler','AMP','rng','global_memory','local_auxiliary','ledger'],
        'atomic_latest_save':True,'restore_function_unit_checked':True,
        'resume_cli_scope':'state restoration primitives and RNG checked; production pipeline always starts independently from Stage1'}
    (Path(directory)/'training_state_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
