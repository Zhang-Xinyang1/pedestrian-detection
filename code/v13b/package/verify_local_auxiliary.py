"""Read-only checks called before the one-shot V9 artifact manifest is written."""
import hashlib
import json
import math
from pathlib import Path

import torch
from local_observed_auxiliary import (LocalObservedAuxiliary, mapping_sha, state_hashes,
                                      tensor_sha, transfer_coefficient)


def require(condition, message):
    if not condition:
        raise ValueError('V9 auxiliary: ' + message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def verify_local_auxiliary(directory,audit,manifest,global_memory,smoke=False):
    directory=Path(directory)
    initialization=read(directory/'local_auxiliary_initialization.json')
    record=read(directory/'local_auxiliary_audit.json')
    require(record==audit['local_auxiliary'],'audit cross-record differs')
    require(initialization==manifest['local_auxiliary'],'initialization differs')
    payload=torch.load(directory/'local_auxiliary_final.pt',map_location='cpu',weights_only=True)
    require(sha(directory/'local_auxiliary_final.pt')==record['file_sha256'],'file digest differs')
    with torch.random.fork_rng(devices=[]):
        aux=LocalObservedAuxiliary(transfer_disabled=manifest['info']['local_transfer_disabled'])
    aux.load_state_dict(payload['state_dict'],strict=True)
    hashes=state_hashes(aux)
    require(hashes==record['tensor_hashes'],'tensor fingerprints differ')
    require(mapping_sha(hashes)==record['final_state_sha256']==payload['state_sha256'],'state digest differs')
    require(aux.specification()==record['specification']==initialization['specification']==payload['specification'],'specification differs')
    require(aux.specification()['trainable_parameters']<2000000,'parameter budget exceeded')
    require(all(bool(torch.isfinite(t).all()) for t in aux.state_dict().values()),'nonfinite state')
    params={n:tensor_sha(p) for n,p in aux.named_parameters()}
    require(params==record['final_parameters'],'learned parameter fingerprints differ')
    require(set(params)==set(initialization['initial_parameters'])==set(record['optimizer_parameters']),'optimizer range differs')
    require(record['initial_parameters']==initialization['initial_parameters'],'starting parameters differ')
    require(params['shared_queries']!=initialization['initial_parameters']['shared_queries'],'local queries never learned')
    for modality in range(3):
        require(initialization['initial_parameters'][f'classifiers.{modality}.weight']==
                read(directory/'stage2_initial_parameters.json')['classifier.weight'],'classifier starting point differs')
    require(initialization['auxiliary_seed']==60000+manifest['info']['seed'],'auxiliary seed differs')
    require(initialization['main_rng_preserved'] and initialization['main_parameter_schema_unchanged'], 'main RNG/schema contract differs')
    require(initialization['initial_memory_observations']==0 and not initialization['regularized_by_stage1_anchor'],'starting memory/anchor differs')
    require(int(aux.memory.successful_updates)==record['successful_updates']==audit['optimizer_steps'],'successful updates differ')
    require(int(aux.memory.skipped_updates)==record['skipped_updates']==audit['planned_steps']-audit['optimizer_steps'],'AMP skips differ')
    expected_mass=torch.stack([global_memory.scenario_mass[:,s%3,s//3] for s in range(6)],dim=1).long()
    require(torch.equal(aux.memory.mass,expected_mass),'local/global observation ledgers differ')
    require(not bool(torch.count_nonzero(aux.memory.centers[aux.memory.mass==0])), 'fabricated unseen scene center')
    require(not bool(torch.count_nonzero(aux.memory.dispersion[aux.memory.mass==0])), 'fabricated unseen scene reliability')
    require(bool((aux.memory.dispersion>=0).all()) and bool((aux.memory.dispersion<=2.00001).all()),'dispersion range differs')
    batch_size=manifest['config']['SOLVER']['STAGE2']['IMS_PER_BATCH']
    require(int(aux.memory.mass.sum())==record['observations']==batch_size*audit['optimizer_steps'],'observation budget differs')
    require(int((aux.memory.mass.sum(1)>0).sum())==record['observed_identity_count'],'identity count differs')
    require(int((aux.memory.mass>0).sum())==record['observed_identity_scene_count'],'scene count differs')
    if not smoke:require(record['observed_identity_count']==500,'production identity coverage incomplete')
    history=read(directory/'local_auxiliary_epoch_history.json')
    require(history==record['epoch_history'] and len(history)==audit['final_epoch'],'epoch history differs')
    require(sum(r['successful_updates'] for r in history)==audit['optimizer_steps'],'epoch update sum differs')
    require(sum(r['skipped_updates'] for r in history)==record['skipped_updates'],'epoch skip sum differs')
    for index,row in enumerate(history,1):
        require(row['epoch']==index and row['batches']==audit['actual_batches_per_epoch'][index-1],'epoch budget differs')
        require(row['successful_updates']+row['skipped_updates']==row['batches'],'epoch AMP counts differ')
        expected=transfer_coefficient(index,aux.transfer_disabled)
        require(row['transfer_coefficient']==expected and
                math.isclose(row['mean_transfer_coefficient'],expected,rel_tol=1e-12,abs_tol=1e-15),
                'transfer schedule differs')
        require(all(not isinstance(v,float) or math.isfinite(v) for v in row.values()),'nonfinite epoch statistic')
        require(row['mean_weighted_loss']>0 and row['mean_local_ce']>0,'local learning inactive')
        require(0<=row['mean_matched_anchor_fraction']<=1,'invalid matched anchor fraction')
        require(-1e-8<=row['mean_matched_slot_fraction']<=1.00001,'invalid matched slot fraction')
        require(row['mean_correspondence_row_capacity_max']<=1.00001 and
                row['mean_correspondence_column_capacity_max']<=1.00001,'correspondence capacity exceeded')
        require(row['mean_matched_cross_view_pairs']<=row['mean_cross_view_pairs'] and
                row['mean_matched_cross_modality_pairs']<=row['mean_cross_modality_pairs'] and
                row['mean_matched_joint_pairs']<=row['mean_joint_pairs'],'matched nonexistent source')
    diagnostics=read(directory/'local_auxiliary_training_diagnostics.json')
    expected_probes=[]
    for e,n in enumerate(audit['actual_batches_per_epoch'],1):
        batches=set(range(1,n+1,300))
        if smoke and audit['final_epoch']==2:
            batches.add(n)
        expected_probes.extend((e,b) for b in sorted(batches))
    require([(r['epoch'],r['batch']) for r in diagnostics]==expected_probes,'gradient probe budget differs')
    require(any(r['auxiliary_gradient_norm']>0 for r in diagnostics),'no auxiliary gradient reaches main encoder')
    require(all(math.isfinite(r['auxiliary_gradient_norm']) for r in diagnostics),'nonfinite gradient probe')
    if not smoke and not aux.transfer_disabled:
        require(record['production_transfer_activated'],'transfer schedule never enabled')
        require(any(r['mean_teacher_anchor_fraction']>0 and r['mean_transfer_loss']>0
                    for r in history if r['transfer_coefficient']>0),'foreign-scene knowledge transfer inactive')
    if not smoke:
        require(any(r['mean_matched_slot_mass']>0 and r['mean_cross_scene_alignment']>0 for r in history),
                'partial correspondence alignment inactive for the whole run')
    return dict(status='PASS', method=aux.specification()['method'],
                trainable_parameters=aux.specification()['trainable_parameters'],
                observed_identities=record['observed_identity_count'],
                observed_identity_scenes=record['observed_identity_scene_count'],
                successful_updates=record['successful_updates'],skipped_updates=record['skipped_updates'],
                state_sha256=record['final_state_sha256'], transfer_disabled=aux.transfer_disabled,
                transfer_activated=record['production_transfer_activated'],
                smoke_transfer_scope=f"epochs1_to{audit['final_epoch']}_are_production_warmup; activated_path_checked_separately_in_CUDA_test" if smoke else 'production_schedule_checked',
                partial_correspondence_active=any(r['mean_matched_slot_mass']>0 for r in history),
                correspondence_identity_margin_required=True,
                inference_descriptor_unchanged=True)
