#!/bin/bash
#SBATCH --job-name=clipreid_csf_remaining30
#SBATCH --partition=vip_gpu_a800_scwc189
#SBATCH --qos=gpugpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=d1n41a16g01
#SBATCH --time=08:00:00
#SBATCH --output=/data/run01/scwc189/users/ZhangXinyang/logs/clipreid_csf_remaining30_%j.out
#SBATCH --error=/data/run01/scwc189/users/ZhangXinyang/logs/clipreid_csf_remaining30_%j.err
set -euo pipefail
ROOT=/data/run01/scwc189/users/ZhangXinyang
PACKAGE="$ROOT/jobs/clipreid_csf_screening_20261002_v3_remaining"
PYTHON="$ROOT/envs/uad/bin/python"
STAGE1="$ROOT/outputs/clipreid_scene_prompt_quality_gate_v6_memfix_178528_control/ViT-B-16_stage1_120.pth"
cd "$PACKAGE"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
sha256sum --check --strict SHA256SUMS
"$PYTHON" -B -c 'import torch; assert torch.cuda.device_count()==1; assert "A800" in torch.cuda.get_device_name(0); print("CSF_SINGLE_A800_CONFIRMED", torch.cuda.get_device_name(0), flush=True)'
PREFIX="$ROOT/outputs/clipreid_csf_remaining_screen30_20261002_v3_${SLURM_JOB_ID}"
for METHOD in six_scene csf; do
  OUTPUT="${PREFIX}_${METHOD}_seed1"
  test ! -e "$OUTPUT"
  echo "CSF_SCREENING_30_BEGIN method=$METHOD output=$OUTPUT"
  "$PYTHON" -B run_stage2_only.py \
    --data-root "$ROOT/datasets" --weights "$ROOT/pretrained/ViT-B-16.pt" \
    --stage1-checkpoint "$STAGE1" --output "$OUTPUT" --seed 1 \
    --csf-weight 0.1 --prototype-method "$METHOD" --screening-30
  set +e
  "$PYTHON" -B screen_against_uad.py "$OUTPUT"
  GATE_RC=$?
  set -e
  if [[ "$GATE_RC" -ne 0 && "$GATE_RC" -ne 2 ]]; then
    echo "CSF_SCREENING_GATE_ERROR method=$METHOD rc=$GATE_RC" >&2
    exit "$GATE_RC"
  fi
  "$PYTHON" -B verify_results_csf.py "$OUTPUT" --expected-seed 1 --screening-30
  if [[ "$GATE_RC" -eq 0 ]]; then
    echo "CSF_SCREENING_30_PASS method=$METHOD output=$OUTPUT"
  else
    echo "CSF_SCREENING_30_REJECT method=$METHOD output=$OUTPUT"
  fi
  echo "CSF_SCREENING_30_DONE method=$METHOD gate_rc=$GATE_RC output=$OUTPUT"
done
"$PYTHON" -B - "$PREFIX" <<'CSF_PAIR'
import hashlib, json, sys
from pathlib import Path
reference=None
results=[]
for method in ('six_scene','csf'):
    directory=Path(sys.argv[1]+'_'+method+'_seed1')
    def read(name): return json.loads((directory/name).read_text())
    manifest=read('manifest.json')
    audit=read('stage2_audit.json')
    marker=read('VERIFICATION.json')
    cfg=manifest['config'].copy()
    cfg['OUTPUT_DIR']='<arm>'
    info=manifest['info']
    assert marker['status']=='PASS' and not marker['smoke'] and marker['screening_30']
    assert marker['final_epoch']==30 and marker['seed']==1
    assert info['prototype_method']==audit['csf']['prototype_method']==method
    assert audit['final_epoch']==30 and not audit['stopped_early']
    assert len(audit['actual_batches_per_epoch'])==30
    record={'config':cfg,'seed':info['seed'],'source_stage1_sha256':info['source_stage1_checkpoint_sha256'],
        'initial_parameters':read('stage2_initial_parameters.json'),
        'sampler_plan_sha256':info['sampler_plan_sha256'],'candidate_policy':info['candidate_policy'],
        'weight':info['csf_loss_weight'],'temperature':info['csf_temperature'],
        'reconstruction_weight':info['csf_reconstruction_weight'],
        'source_hashes':manifest['adapter_sources'],'provenance':manifest['provenance'],
        'counts':manifest['actual_run_counts'],'train_scene_counts':manifest['train_scene_counts'],
        'query':manifest['eval_query'],'gallery':manifest['eval_gallery']}
    if reference is None:reference=record
    else:assert record==reference,'Paired protocol differs'
    gate=read('SCREENING_GATE.json')
    results.append({'method':method,'gate':gate['status'],'passing_checkpoints':gate['passing_checkpoints'],
        'optimizer_steps':audit['optimizer_steps'],'planned_steps':audit['planned_steps'],
        'mean_valid_anchor_fraction':audit['csf']['mean_valid_anchor_fraction'],
        'mean_candidate_identities':audit['csf']['mean_candidate_identities'],
        'final_map_percent':marker['map_percent'],'final_rank1_percent':marker['rank1_percent']})
digest=hashlib.sha256(json.dumps(reference,sort_keys=True).encode()).hexdigest()
print('CSF_REMAINING_30_PAIR_OK '+digest,flush=True)
print('CSF_REMAINING_30_RESULTS '+json.dumps(results,sort_keys=True),flush=True)
CSF_PAIR
echo "CSF_REMAINING_30_ALL_DONE prefix=$PREFIX"
