"""Check deployment isolation, approved base and explicit training-only extension."""
import ast
from pathlib import Path
from local_observed_auxiliary import LocalObservedAuxiliary

def main():
    root=Path(__file__).resolve().parent
    run=(root/'run_stage2_only.py').read_text(encoding='utf-8')
    loop=(root/'scene_stage2_csf.py').read_text(encoding='utf-8')
    pipe=(root.parent/'run_pipeline.py').read_text(encoding='utf-8')
    for path in root.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    assert 'load_state_dict(source_state, strict=True)' in run
    assert run.index('load_state_dict(source_state, strict=True)')<run.index('initialize_auxiliary(model')
    assert 'local_auxiliary.attach(model.image_encoder)' in run
    assert 'local_auxiliary.detach()' in run
    assert 'local_auxiliary_final.pt' in run and 'local_auxiliary_audit.json' in run
    assert 'step_succeeded,auxiliary_stats' in loop
    assert 'scnet_v9_local40_runs' in pipe and 'observed_local_identity_l2sp_fixed40_seed1' in pipe
    assert "'--prototype-method','identity_full'" in pipe
    assert "'--local-observed'" in pipe
    assert "'declared_budget':40,'main_checkpoint_epoch':40" in pipe
    assert 'ACTIVE.lock' in pipe and "lock.open('x'" in pipe
    assert 'scnet_v7_fixed120_runs' not in pipe
    assert 'RTX4090' not in pipe
    a=LocalObservedAuxiliary();spec=a.specification()
    assert spec['queries']==4 and spec['trainable_parameters']==1170432
    assert not spec['added_backbone_forward'] and not spec['missing_scene_completion']
    print('V9_LOCAL40_CONTRACT_OK',flush=True)

if __name__=='__main__':main()
