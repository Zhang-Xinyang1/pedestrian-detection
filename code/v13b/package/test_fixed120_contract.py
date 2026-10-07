"""Regression checks for the authorized fixed120 budget, LR, and final gate."""
import json
import math
import sys
import tempfile
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'upstream'))
from solver.lr_scheduler import WarmupMultiStepLR
from verify_results_csf import validate_epoch_budget
import screen_against_uad_120 as gate

def record(epoch):
    return {'final_epoch':epoch,'stopped_early':False,'early_stop_reason':None,
            'csf':{'final_epoch':epoch,'stopped_early':False,'early_stop_reason':None}}

def rejected(value,**kwargs):
    try:validate_epoch_budget(value,**kwargs)
    except ValueError:return
    raise AssertionError('Wrong budget accepted')

def sequence(milestones):
    p=torch.nn.Parameter(torch.zeros(1))
    opt=torch.optim.Adam([p],lr=5e-6)
    schedule=WarmupMultiStepLR(opt,milestones,.1,.1,10,'linear')
    result=[]
    for epoch in range(1,121):
        schedule.step();result.append(opt.param_groups[0]['lr'])
        p.sum().backward();opt.step();opt.zero_grad()
    return result

def check_gate(equal=False):
    with tempfile.TemporaryDirectory(prefix='prvc_fixed120_gate_') as temp:
        directory=Path(temp)
        (directory/'manifest.json').write_text(json.dumps({'info':{'fixed120_cooldown':True,'stage2_epochs':120,'prototype_method':'csf_full_identity'}}))
        for epoch in (10,20,30):
            threshold=gate.BASELINE['epochs'][str(epoch)]
            value={'ap':(threshold['map_percent']+(0.0 if equal else 1.0))/100.,
                   'r1':(threshold['rank1_percent']+1.0)/100.}
            (directory/f'eval_stage2_{epoch:03d}.json').write_text(json.dumps(value))
        # A high diagnostic epoch40 does not replace the declared final epoch120.
        (directory/'eval_stage2_040.json').write_text(json.dumps({'ap':.99,'r1':.99}))
        (directory/'eval_stage2_120.json').write_text(json.dumps({'ap':.117,'r1':.28}))
        old_argv=sys.argv
        try:
            sys.argv=['screen_against_uad_120.py',str(directory)]
            code=gate.main()
        finally:sys.argv=old_argv
        result=json.loads((directory/'SCREENING_GATE.json').read_text())
        assert code==(2 if equal else 0)
        assert result['passing_checkpoint_count']==(0 if equal else 3)
        assert math.isclose(result['epoch120']['map_percent'],11.7)
        assert not result['epoch120']['strict_full_uad'] and not result['epoch120']['formal_target']

def main():
    assert validate_epoch_budget(record(120),fixed120_cooldown=True)==120
    for epoch in [4,30,40,60,119]:rejected(record(epoch),fixed120_cooldown=True)
    rejected(record(4),smoke=True,fixed120_cooldown=True)
    early=record(120);early['stopped_early']=True
    rejected(early,fixed120_cooldown=True)
    old=sequence([60,100]);new=sequence([21])
    assert old[:20]==new[:20]
    assert len(new)==120 and all(math.isclose(v,5e-7,rel_tol=1e-12) for v in new[20:])
    check_gate();check_gate(equal=True)
    print('CSF_FIXED120_COOLDOWN_CONTRACT_OK')

if __name__=='__main__':main()
