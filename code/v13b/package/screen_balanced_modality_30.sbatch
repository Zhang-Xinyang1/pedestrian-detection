#!/bin/bash
#SBATCH --job-name=clipreid_csf_balanced30
#SBATCH --partition=vip_gpu_a800_scwc189
#SBATCH --qos=gpugpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/data/run01/scwc189/users/ZhangXinyang/logs/clipreid_csf_balanced30_%j.out
#SBATCH --error=/data/run01/scwc189/users/ZhangXinyang/logs/clipreid_csf_balanced30_%j.err
set -euo pipefail
BASE=/data/run01/scwc189/users
if [[ -d "$BASE/ZhangXinyang" ]]; then ROOT="$BASE/ZhangXinyang"; else ROOT=$(find "$BASE" -maxdepth 1 -type d -name 'ZhangXinyang-*' -print -quit); fi
test -n "$ROOT"
PACKAGE="$ROOT/jobs/clipreid_csf_screening_20261002_v2_balanced_modality"
PYTHON="$ROOT/envs/uad/bin/python"
STAGE1="$ROOT/outputs/clipreid_scene_prompt_quality_gate_v6_memfix_178528_control/ViT-B-16_stage1_120.pth"
cd "$PACKAGE"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
sha256sum --check --strict SHA256SUMS
OUTPUT="$ROOT/outputs/clipreid_csf_balanced_modality_screen30_20261002_v2_${SLURM_JOB_ID}_seed1"
test ! -e "$OUTPUT"
"$PYTHON" -B run_stage2_only.py   --data-root "$ROOT/datasets" --weights "$ROOT/pretrained/ViT-B-16.pt"   --stage1-checkpoint "$STAGE1" --output "$OUTPUT" --seed 1   --csf-weight 0.1 --prototype-method balanced_modality --screening-30
set +e
"$PYTHON" -B screen_against_uad.py "$OUTPUT"
GATE_RC=$?
set -e
"$PYTHON" -B verify_results_csf.py "$OUTPUT" --expected-seed 1 --screening-30
if [[ "$GATE_RC" -eq 0 ]]; then
  echo "CSF_SCREENING_30_PASS method=balanced_modality output=$OUTPUT"
else
  echo "CSF_SCREENING_30_REJECT method=balanced_modality output=$OUTPUT"
fi
echo "CSF_SCREENING_30_DONE gate_rc=$GATE_RC output=$OUTPUT"
exit 0
