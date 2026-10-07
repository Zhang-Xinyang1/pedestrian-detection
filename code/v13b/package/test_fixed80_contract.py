"""Fixed80 must reject premature runs and preserve the V10 learning rate."""
import math,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent/'upstream'))
from solver.lr_scheduler import WarmupMultiStepLR
from verify_results_csf import validate_epoch_budget
def record(e):
 return {'final_epoch':e,'stopped_early':False,'early_stop_reason':None,'csf':{'final_epoch':e,'stopped_early':False,'early_stop_reason':None}}
assert validate_epoch_budget(record(80),fixed80_cooldown=True)==80
for e in [2,4,30,40,60,79,81,120]:
 try:validate_epoch_budget(record(e),fixed80_cooldown=True)
 except ValueError:pass
 else:raise AssertionError('Wrong fixed80 budget accepted')
for extra in [{'screening_40':True},{'fixed120_cooldown':True},{'smoke':True}]:
 try:validate_epoch_budget(record(80),fixed80_cooldown=True,**extra)
 except ValueError:pass
 else:raise AssertionError('Conflicting budget accepted')
p=torch.nn.Parameter(torch.zeros(1));opt=torch.optim.Adam([p],lr=5e-6)
s=WarmupMultiStepLR(opt,[21],.1,.1,10,'linear');values=[]
for e in range(1,81):
 s.step();values.append(opt.param_groups[0]['lr']);p.sum().backward();opt.step();opt.zero_grad()
assert len(values)==80 and all(math.isclose(x,5e-7,rel_tol=1e-12) for x in values[20:])
assert all(math.isclose(values[i],5e-6*(.1+.9*(i+1)/10),rel_tol=1e-12) for i in range(9))
print('V10_FIXED80_BUDGET_LR_PASS')
