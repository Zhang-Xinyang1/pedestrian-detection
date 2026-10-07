"""Verify fixed-final scene-prompt results against the existing official run."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import torch
from verify_results import verify,read,require,sha,PACKAGE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--reference-package',type=Path,default=PACKAGE.parent/'official_clipreid_whumars_20260918')
    p.add_argument('--local-smoke',action='store_true')
    p.add_argument('outputs',type=Path,nargs=3)
    a=p.parse_args();torch.set_num_threads(4)
    reference=read(PACKAGE/'REFERENCE.json')
    source=a.reference_package/'verify_results.py'
    require(sha(source)==reference['verifier_sha256'],'Official verifier differs from pinned reference')
    spec=importlib.util.spec_from_file_location('official_reference_verifier',source)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    original.verify(a.reference,a.local_smoke)
    m0=read(a.reference/'manifest.json');init=read(a.reference/'initial_parameters.json')
    cfg0=copy.deepcopy(m0['config']);cfg0.pop('OUTPUT_DIR')
    epoch=4 if a.local_smoke else 120
    base=read(a.reference/f'eval_stage2_{epoch:03d}.json')
    if not a.local_smoke:
        require(abs(100*base['ap']-reference['map_percent'])<1e-8,'Reference is not the completed official165536 result')
        require(abs(100*base['r1']-reference['rank1_percent'])<1e-8,'Reference Rank1 differs')
    result={};weights=set();stage1_parameters=None;arm_cfg=None;arm_counts=None
    for output in a.outputs:
        verify(output,a.local_smoke)
        manifest=read(output/'manifest.json');mode=manifest['args']['prompt_mode']
        weight=float(manifest['args'].get('i2t_weight',1.0))
        require(mode=='modality' and weight in (0.25,0.5,1.0) and weight not in weights,'Expected three distinct modality I2T weights')
        weights.add(weight)
        cfg=copy.deepcopy(manifest['config']);cfg.pop('OUTPUT_DIR')
        cfg_base=copy.deepcopy(cfg); cfg_base['MODEL']['I2T_LOSS_WEIGHT']=cfg0['MODEL']['I2T_LOSS_WEIGHT']
        require(cfg_base==cfg0,'Training/evaluation configuration differs from official reference except I2T weight')
        after1=read(output/'stage1_audit.json')['final_parameters']
        if stage1_parameters is not None:
            require(after1==stage1_parameters,'Arms have different stage1 prompts or visual starting parameters')
            require(cfg_base==arm_cfg and manifest['actual_run_counts']==arm_counts,'Arms differ beyond I2T weight')
        stage1_parameters=after1;arm_cfg=cfg_base;arm_counts=manifest['actual_run_counts']
        require(manifest['info']['train_metadata_sha256']==m0['info']['train_metadata_sha256'],'Training scope differs')
        require(manifest['provenance']==m0['provenance'],'Pretrained weights/upstream provenance differs')
        require(read(output/'initial_parameters.json')==init,'Initial trainable parameters differ')
        require(manifest['eval_query']==m0['eval_query'] and manifest['eval_gallery']==m0['eval_gallery'],'Evaluation order/scope differs')
        if not a.local_smoke:
            require(manifest['actual_run_counts']==m0['actual_run_counts'],'Training/data counts differ')
        r=read(output/f'eval_stage2_{epoch:03d}.json')
        if a.local_smoke:
            result[str(weight)]={'artifacts_verified':True,'performance_comparison':False,
                'reason':'Engineering smoke; modified arms use 25 images, official regression uses 17'}
            continue
        keys=['ap','r1','macro_map','worst_map']
        result[str(weight)]={'i2t_weight':weight,'percent':{k:100*r[k] for k in keys},
            'delta_pp_vs_official':{k:100*(r[k]-base[k]) for k in keys},
            'scene_delta_pp':{k:100*(r['scenes'][k]-base['scenes'][k]) for k in r['scenes']}}
    require(weights=={0.25,0.5,1.0},'Missing I2T weight arm')
    if not a.local_smoke:
        for row in result.values():
            row['delta_pp_vs_modality_w1']={k:v-result['1.0']['percent'][k] for k,v in row['percent'].items()}
    print('I2T_WEIGHT_COMPARISON_OK '+json.dumps({'local_smoke':a.local_smoke,'fixed_final_epoch':epoch,'results':result}))
    print('Single seed; no statistical-significance or CVPR novelty claim. Local smoke metrics are not research results.')


if __name__=='__main__':main()
