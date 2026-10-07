"""One-shot SCNet preparation -> complete import -> smoke+verify -> partial-matched observed-local identity_full fixed40+verify."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

REPAIR=Path(__file__).resolve().parent
PACKAGE=REPAIR/'package'
STAGE1_HASH='2032ee52590233f52f720189231c720a32172f1639a381a22e3dd8fd17dae7c3'
TRAIN_HASH='fe41ed4268bd6ec853e16b134937ff9ec6ac36e4ddcaa76db2b99a66f664f234'
SAMPLER_HASH='52c8e262252e57ee8febe2868dad9c4e11e1e8f11bd61e69d21ec9b12008d8a0'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1048576),b''):h.update(data)
    return h.hexdigest()

def verify_package():
    count=0
    for line in (PACKAGE/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1);path=PACKAGE/name
        assert PACKAGE.resolve() in path.resolve().parents,'Unsafe package manifest path'
        assert path.is_file() and sha(path)==digest,'Package hash differs: '+name
        count+=1
    expected=json.loads((REPAIR/"BUILD_MANIFEST.json").read_text())["package_file_count"]
    assert count==expected,"V13 shared package count differs"
    print("SCNET_PACKAGE_HASH_PASS entries="+str(count),flush=True)

def data_root(project,override,runtime):
    if override:
        candidate=Path(override)
        assert (candidate/'WHU-MARS/train/RGB').is_dir(),'DATA_ROOT must contain WHU-MARS/train/RGB'
        return candidate
    for candidate in [project/'datasets',project]:
        if (candidate/'WHU-MARS/train/RGB').is_dir():return candidate
    # The original upload ZIP placed images under datasets/train directly.
    source=project/'datasets'
    assert all((source/s/m).is_dir() for s in ('train','query','test') for m in ('RGB','IR','Thermal')), 'Dataset missing required 9 split/modality directories'
    target=runtime/'data_layout'
    target.mkdir(exist_ok=True)
    link=target/'WHU-MARS'
    if not link.exists():link.symlink_to(source.resolve(),target_is_directory=True)
    assert link.resolve()==source.resolve(),'Dataset link points to another source'
    print('SCNET_DATA_LAYOUT_PASS source='+str(source)+' alias='+str(link),flush=True)
    return target

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--site',type=Path)
    parser.add_argument('--runtime-root',type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--local-validation',action='store_true')
    parser.add_argument("--arm",choices=["A","B"],required=True)
    args=parser.parse_args()
    project=args.project_root.resolve()
    runtime=(args.runtime_root or project/'scnet_v13_lr100_ab_runs').resolve();runtime.mkdir(parents=True,exist_ok=True)
    # A restart of the same SCNet pod must not silently create another training run.
    run_id=args.run_id or ('V13A_v10_constant_tail_fixed100_seed1' if args.arm=='A' else 'V13B_v10_cosine_tail_fixed100_seed1')
    assert run_id and Path(run_id).name==run_id,'RUN_ID must be one path component'
    if not args.local_validation:
        assert not (project/'scnet_v6_runs/ACTIVE.lock').exists(), 'V6 ACTIVE.lock exists; inspect the existing task, never delete it to bypass'
    lock=runtime/('ACTIVE_'+args.arm+'.lock')
    with lock.open('x',encoding='utf-8') as f:f.write(json.dumps({'pid':os.getpid(),'run_id':run_id}))
    record={};destination=None
    try:
        new_destination=runtime/run_id;new_destination.mkdir(exist_ok=False)
        destination=new_destination
        record={'run_id':run_id,'status':'PREPARING','declared_budget':100,'main_checkpoint_epoch':100,'arm':args.arm,'hostname':os.environ.get('HOSTNAME') or __import__('socket').gethostname(),'local_validation':args.local_validation,'steps':[]}
        def save():
            (destination/'pipeline_status.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        save()
        def run(label,command,allowed=(0,),env=None):
            record.update(status='RUNNING',current_phase=label,phase_started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat());save()
            print('SCNET_PHASE '+label,flush=True)
            with (destination/(label+'.log')).open('w',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=PACKAGE,env=env or environment,
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                try:
                    for line in child.stdout:
                        log.write(line);log.flush()
                        # Keep interactive console concise; full logs stay on persistent storage.
                        if len(line)<2000:print(line.rstrip(),flush=True)
                        else:print(line.split(' ',1)[0]+' [full record in persistent log]',flush=True)
                    code=child.wait()
                except BaseException:
                    child.terminate();child.wait();raise
            record['steps'].append({'phase':label,'exit_code':code});save()
            if code not in allowed:raise RuntimeError(label+' failed; see '+str(destination/(label+'.log')))
            return code
        verify_package()
        environment=os.environ.copy()
        environment.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                           OPENBLAS_NUM_THREADS='4',PYTHONNOUSERSITE='1')
        if args.site:
            site=args.site.resolve()
        else:
            assert sys.version_info[:2]==(3,12),'Use the confirmed Python3.12 SCNet image'
            site=runtime/'env_linux_cp312_v13_lr100'
            site.mkdir(exist_ok=True)
            run('install_dependencies',[sys.executable,'-m','pip','install','--no-index','--no-deps','--ignore-installed',
                '--upgrade','--find-links',str(REPAIR/'wheels'),'--target',str(site),'-r',str(REPAIR/'requirements-lock.txt')],env=environment)
        environment['PYTHONPATH']=os.pathsep.join([str(site),str(PACKAGE),str(PACKAGE/'upstream')])
        check=[sys.executable,'-u','-B',str(REPAIR/'environment_preflight.py'),'--package',str(PACKAGE),
               '--output',str(destination/'environment.json')]
        if args.local_validation:check.append('--local-validation')
        run('environment',check)
        data=data_root(project,args.data_root,runtime)
        weights=project/'pretrained/ViT-B-16.pt';stage1=project/'pretrained/ViT-B-16_stage1_120.pth'
        assert weights.is_file() and stage1.is_file(),'Missing approved weights/Stage1'
        assert sha(stage1)==STAGE1_HASH,'Approved Stage1 file hash differs'
        common=['--data-root',str(data),'--weights',str(weights),'--stage1-checkpoint',str(stage1),
                '--seed','1','--csf-weight','0.1','--prototype-method','identity_full','--l2sp','--local-observed','--lr-tail-arm',args.arm]
        cli=[sys.executable,'-u','-B',str(PACKAGE/'run_stage2_only.py')]
        run('full_data_contract',cli+common+['--output',str(destination/'check_only_no_write'),'--fixed100-ab','--check-only'])
        content=(destination/'full_data_contract.log').read_text()
        info=json.loads(next(line.split('CSF_STAGE2_INPUTS_OK ',1)[1] for line in content.splitlines() if line.startswith('CSF_STAGE2_INPUTS_OK ')))
        assert info['train_metadata_sha256']==TRAIN_HASH and info['sampler_plan_sha256']==SAMPLER_HASH,'Metadata/sampler plan differs'
        assert info['stage2_epochs']==100 and info['feature_dim']==1280,'Identity40 feature contract differs'
        record['full_data_contract']=info;save()
        for name in ['test_data_protocol.py','test_fixed_epoch_budget.py','test_v13_lr100_contract.py','test_verify_csf_contract.py','test_lr_tail_schedule.py','test_training_state_io.py','test_partial_local_matching.py','test_local_observed_auxiliary.py','test_l2sp_epoch_accounting.py']:
            run(name.removesuffix('.py'),[sys.executable,'-u','-B',str(PACKAGE/name)])
        run('test_partial_matching_cuda',[sys.executable,'-u','-B',str(PACKAGE/'test_partial_local_matching.py'),'--device','cuda'])
        run('test_lr_tail_cuda',[sys.executable,'-u','-B',str(PACKAGE/'test_lr_tail_schedule.py'),'--device','cuda'])
        run('test_training_state_cuda',[sys.executable,'-u','-B',str(PACKAGE/'test_training_state_io.py'),'--device','cuda'])
        run('test_local_observed_cuda',[sys.executable,'-u','-B',str(PACKAGE/'test_local_observed_auxiliary.py'),'--device','cuda'])
        run('test_l2sp_cuda',[sys.executable,'-u','-B',str(PACKAGE/'test_l2sp.py'),'--device','cuda'])
        smoke=destination/'smoke'
        run('gpu_smoke',cli+common+['--output',str(smoke),'--local-smoke','--smoke-six-scenes','--smoke-epochs','4'])
        run('smoke_verify',[sys.executable,'-u','-B',str(PACKAGE/'verify_results_csf.py'),str(smoke),'--expected-seed','1','--local-smoke','--smoke-epochs','4'])
        verified=json.loads((smoke/'VERIFICATION.json').read_text())
        assert verified['status']=='PASS' and verified['final_epoch']==4,'Smoke did not verify'
        record['smoke_verification']=verified;save()
        print('SCNET_GPU_SMOKE_VERIFIED',flush=True)
        if args.local_validation:
            record['status']='LOCAL_VALIDATION_PASS';save()
            print('SCNET_LOCAL_PIPELINE_PASS '+str(destination),flush=True)
            return
        production=destination/'production100'
        run('production100',cli+common+['--output',str(production),'--fixed100-ab'])
        gate=run('fixed100_target_report',[sys.executable,'-u','-B',str(PACKAGE/'report_fixed100_target.py'),str(production)],allowed=(0,2))
        run('production_verify',[sys.executable,'-u','-B',str(PACKAGE/'verify_results_csf.py'),str(production),'--expected-seed','1','--fixed100-ab'])
        marker=json.loads((production/'VERIFICATION.json').read_text())
        assert marker['status']=='PASS' and marker['final_epoch']==100,'Production verification incomplete'
        record.update(status='COMPLETED_VERIFIED',gate_exit_code=gate,final_verification=marker);save()
        print('SCNET_V13_LR100_COMPLETED_VERIFIED '+str(destination),flush=True)
    except BaseException:
        if destination and destination.is_dir():
            record.update(status='FAILED',error=traceback.format_exc())
            (destination/'pipeline_status.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        raise
    finally:
        # The marker prevents a second simultaneous run; failed logs/checkpoints remain intact.
        if lock.is_file():
            owner=json.loads(lock.read_text())
            if owner.get('pid')==os.getpid() and owner.get('run_id')==run_id:lock.unlink()


if __name__=='__main__':main()
