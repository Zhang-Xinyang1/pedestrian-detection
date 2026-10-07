"""Gate the first 30 checkpoints and report the fixed-40 full-UAD comparison."""
import json, sys
from pathlib import Path
PACKAGE=Path(__file__).resolve().parent
BASELINE=json.loads((PACKAGE/'UAD_FIRST30_BASELINE.json').read_text(encoding='utf-8'))

def main():
    if len(sys.argv)!=2: raise SystemExit('usage: screen_against_uad_80.py OUTPUT_DIR')
    directory=Path(sys.argv[1]); observations={}; failures=[]; passing=[]
    for epoch in (10,20,30):
        path=directory/f'eval_stage2_{epoch:03d}.json'
        if not path.is_file(): failures.append(f'missing evaluation epoch {epoch}'); continue
        result=json.loads(path.read_text(encoding='utf-8'))
        candidate={'map_percent':100.*float(result['ap']),'rank1_percent':100.*float(result['r1'])}
        threshold=BASELINE['epochs'][str(epoch)]
        win=float(result['ap'])>threshold['map_percent']/100. and float(result['r1'])>threshold['rank1_percent']/100.
        observations[str(epoch)]={'candidate':candidate,'uad':threshold,'strict_both_exceed':win}
        if win: passing.append(epoch)
        else: failures.append(f'epoch {epoch} did not exceed both UAD metrics')
    epoch80_path=directory/'eval_stage2_080.json'
    epoch80=None
    if epoch80_path.is_file():
        r=json.loads(epoch80_path.read_text(encoding='utf-8'))
        epoch80={'map_percent':100.*float(r['ap']),'rank1_percent':100.*float(r['r1']),
                 'strict_full_uad':100.*float(r['ap'])>10.83 and 100.*float(r['r1'])>28.40,
                 'formal_target':100.*float(r['ap'])>=12.0 and 100.*float(r['r1'])>=29.0}
    else: failures.append('missing evaluation epoch 80')
    if len(passing)<2: failures.append(f'only {len(passing)}/3 checkpoints passed; at least 2 required')
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['info']['prototype_method']=='identity_full' and manifest['info']['stage2_epochs']==80 and manifest['info']['fixed80_cooldown'] and manifest['info']['local_observed_enabled'], 'V9 local40 recipe differs'
    assert not (directory/'SCREENING_GATE.json').exists(), 'Gate already exists; never overwrite'
    payload={'status':'PASS' if len(passing)>=2 and epoch80 is not None and not any(x.startswith('missing evaluation') for x in failures) else 'REJECT',
        'method':manifest['info']['prototype_method'],'seed':1,'screening_epochs':80,
        'passing_checkpoints':passing,'passing_checkpoint_count':len(passing),
        'first30_gate_rule':BASELINE['comparison_rule'],'uad_full120_reference':{'map_percent':10.83,'rank1_percent':28.40},
        'epoch80':epoch80,'uad_baseline':BASELINE,'observations':observations,'failures':failures,
        'epoch40_internal':json.loads((directory/'eval_stage2_040.json').read_text(encoding='utf-8')),
        'note':'Epoch80 has no fabricated historical first-40 threshold; full-UAD and formal-target comparisons are reported separately.'}
    (directory/'SCREENING_GATE.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('CSF_SCREENING_GATE_'+payload['status']+' '+json.dumps(payload,sort_keys=True),flush=True)
    return 0 if payload['status']=='PASS' else 2
if __name__=='__main__': raise SystemExit(main())
