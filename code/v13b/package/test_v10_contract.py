"""Check independent deployment, fixed production and explicit two-epoch smoke."""
import ast
from pathlib import Path
from local_observed_auxiliary import LocalObservedAuxiliary
from verify_results_csf import validate_epoch_budget


def record(epoch):
    return dict(final_epoch=epoch,stopped_early=False,early_stop_reason=None,
                csf=dict(final_epoch=epoch,stopped_early=False,early_stop_reason=None))


def main():
    root=Path(__file__).resolve().parent
    for p in root.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    runner=(root/'run_stage2_only.py').read_text(encoding='utf-8')
    pipe=(root.parent/'run_pipeline.py').read_text(encoding='utf-8')
    assert 'scnet_v10_match40_runs' in pipe and 'matched_local_identity_l2sp_fixed40_seed1' in pipe
    assert "'--local-observed'" in pipe and "'--smoke-epochs','2'" in pipe
    assert "'declared_budget':40,'main_checkpoint_epoch':40" in pipe
    assert 'load_state_dict(source_state, strict=True)' in runner
    assert runner.index('load_state_dict(source_state, strict=True)')<runner.index('initialize_auxiliary(model')
    assert validate_epoch_budget(record(2),smoke=True,smoke_epochs=2)==2
    assert validate_epoch_budget(record(40),screening_40=True)==40
    for epoch in (1,3,4):
        try:validate_epoch_budget(record(epoch),smoke=True,smoke_epochs=2)
        except ValueError:pass
        else:raise AssertionError('wrong two-epoch smoke accepted')
    spec=LocalObservedAuxiliary().specification()
    assert spec['trainable_parameters']==1170432 and spec['local_correspondence']['extra_learned_parameters']==0
    assert spec['local_correspondence']['correspondence_stop_gradient']
    assert not spec['added_backbone_forward'] and not spec['missing_scene_completion']
    assert 'scnet_v7_fixed120_runs' not in pipe and 'scnet_v9_local40_runs' not in pipe
    print('V10_MATCH40_CONTRACT_OK smoke2=True production40=True',flush=True)


if __name__=='__main__':main()
