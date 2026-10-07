"""Current payload requires fixed100; compressed smoke budgets remain separate."""
from verify_results_csf import validate_epoch_budget

def record(epoch,early=False):
    return {'final_epoch':epoch,'stopped_early':early,'early_stop_reason':None,
        'csf':{'final_epoch':epoch,'stopped_early':early,'early_stop_reason':None}}

assert validate_epoch_budget(record(100),fixed100_ab=True)==100
assert validate_epoch_budget(record(4),smoke=True,smoke_epochs=4)==4
for epoch,early in [(40,False),(80,False),(99,False),(120,False),(100,True)]:
    try:validate_epoch_budget(record(epoch,early),fixed100_ab=True)
    except ValueError:pass
    else:raise AssertionError('Wrong/incomplete budget accepted')
print('V13_FIXED100_EPOCH_BUDGET_OK')
