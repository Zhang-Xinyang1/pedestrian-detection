"""The new shared package changes only the declared tail LR in its AB arms."""
import hashlib
import json
from pathlib import Path
from lr_tail_schedule import tail_specification
from verify_results_csf import validate_epoch_budget

package=Path(__file__).resolve().parent
parent=json.loads((package.parent/'PARENT_SOURCES.json').read_text())
for name,digest in parent['unchanged_method_files'].items():
    assert hashlib.sha256((package/name).read_bytes()).hexdigest()==digest,'V10 method/data changed: '+name
def record(epoch):return {'final_epoch':epoch,'stopped_early':False,'early_stop_reason':None,'csf':{'final_epoch':epoch,'stopped_early':False,'early_stop_reason':None}}
assert validate_epoch_budget(record(100),fixed100_ab=True)==100
for epoch in [40,80,99,120]:
    try:validate_epoch_budget(record(epoch),fixed100_ab=True)
    except ValueError:pass
    else:raise AssertionError('Wrong production budget accepted')
a=tail_specification('A');b=tail_specification('B')
assert {k for k in a if a[k]!=b[k]}=={'arm','tail_final','tail_mode'}
print('V13_V10_METHOD_UNCHANGED_FIXED100_ONE_LR_VARIABLE_CONTRACT_PASS',flush=True)
