"""A/B differ only in LR after epoch40; original first40 schedule is preserved."""
import json
import math
from pathlib import Path
from solver.lr_scheduler import WarmupMultiStepLR

BASE_LR=5e-6
TAIL_INITIAL=5e-7
TAIL_FINAL=1e-8

def tail_specification(arm,smoke=False):
    if arm not in ('A','B'):raise ValueError('Unknown LR arm')
    return {'arm':arm,'budget':4 if smoke else 100,'tail_start':2 if smoke else 40,
        'tail_end':4 if smoke else 100,'tail_initial':TAIL_INITIAL,'tail_final':TAIL_FINAL if arm=='B' else TAIL_INITIAL,
        'tail_mode':'constant' if arm=='A' else 'cosine','first40_production':'original V10 unchanged',
        'bias_factor':2,'smoke_schedule':'compressed tail transition for optimizer integration' if smoke else None,
        'single_variable':'learning rate after epoch40','all_other_losses_data_and_retrieval':'unchanged V10'}

def tail_lr(epoch,arm,tail_start=40,tail_end=100):
    if arm=='A':return TAIL_INITIAL
    progress=min(max((epoch-tail_start)/(tail_end-tail_start),0.),1.)
    return TAIL_FINAL+.5*(TAIL_INITIAL-TAIL_FINAL)*(1.+math.cos(math.pi*progress))

class MatchedTailScheduler(WarmupMultiStepLR):
    def __init__(self,optimizer,milestones,gamma,warmup_factor,warmup_iters,warmup_method,
                 arm='A',tail_start=40,tail_end=100,last_epoch=-1):
        if arm not in ('A','B') or tail_end<=tail_start:raise ValueError('Invalid LR tail')
        self.arm=arm;self.tail_start=tail_start;self.tail_end=tail_end
        super().__init__(optimizer,milestones,gamma,warmup_factor,warmup_iters,warmup_method,last_epoch)

    def get_lr(self):
        if self.last_epoch<=self.tail_start or (self.arm=='A' and self.tail_start==40):return super().get_lr()
        if self.arm=='A':return [base*self.gamma for base in self.base_lrs]
        value=tail_lr(self.last_epoch,self.arm,self.tail_start,self.tail_end)
        return [value*(base/BASE_LR) for base in self.base_lrs]

    def expected_at(self,epoch):
        if epoch<=self.tail_start:
            warmup=1.
            if epoch<self.warmup_iters:
                alpha=epoch/self.warmup_iters
                warmup=self.warmup_factor*(1.-alpha)+alpha
            factor=self.gamma if epoch>=21 else 1.
            return [base*warmup*factor for base in self.base_lrs]
        value=tail_lr(epoch,self.arm,self.tail_start,self.tail_end)
        return [value*(base/BASE_LR) for base in self.base_lrs]

def validate_optimizer_lrs(optimizer,scheduler,epoch):
    expected=scheduler.expected_at(epoch)
    actual=[float(group['lr']) for group in optimizer.param_groups]
    if scheduler.last_epoch!=epoch or len(actual)!=len(expected):raise ValueError('Scheduler epoch/group count differs')
    if not all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-16) for a,b in zip(actual,expected)):
        raise ValueError('Actual optimizer learning rates differ from declared arm')
    return {'epoch':epoch,'arm':scheduler.arm,'base_lr':actual[0],'group_count':len(actual),
        'lr_min':min(actual),'lr_max':max(actual),'optimizer_rates_match':True,
        'distinct_group_rates':sorted(set(actual))}

def verify_tail_history(directory,manifest,audit):
    directory=Path(directory);spec=manifest['info']['lr_tail_specification']
    if spec!=tail_specification(manifest['info']['lr_tail_arm'],manifest['info']['local_smoke']):raise ValueError('LR specification differs')
    history=json.loads((directory/'learning_rate_epoch_history.json').read_text())
    if history!=audit['csf']['learning_rate_history'] or len(history)!=audit['final_epoch']:raise ValueError('LR history differs')
    for epoch,row in enumerate(history,1):
        if epoch<=spec['tail_start']:
            alpha=min(epoch/10.,1.);factor=.1*(1-alpha)+alpha
            expected=BASE_LR*factor*(.1 if epoch>=21 else 1.)
        else:expected=tail_lr(epoch,spec['arm'],spec['tail_start'],spec['tail_end'])
        if row['epoch']!=epoch or row['arm']!=spec['arm'] or not row['optimizer_rates_match']:
            raise ValueError('LR epoch ledger differs')
        if not math.isclose(row['base_lr'],expected,rel_tol=1e-12,abs_tol=1e-16):raise ValueError('Base LR history differs')
        if not all(math.isclose(value,expected*ratio,rel_tol=1e-12,abs_tol=1e-16) for value,ratio in zip(row['distinct_group_rates'],[1,2])):
            raise ValueError('Bias LR ratio differs')
    return {'status':'PASS','arm':spec['arm'],'single_variable':spec['single_variable'],
        'actual_learning_rate_history_verified':True,'final_base_lr':history[-1]['base_lr'],
        'first40_production_recipe_unchanged':True,'specification':spec}
