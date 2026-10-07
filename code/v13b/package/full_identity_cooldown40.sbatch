#!/bin/bash
#SBATCH --job-name=clipreid_csf_fullid40_v1
#SBATCH --partition=vip_gpu_a800_scwc189
#SBATCH --qos=gpugpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=d1n41a16g01
#SBATCH --time=08:00:00
#SBATCH --output=/data/run01/scwc189/users/ZhangXinyang/logs/clipreid_csf_fullid40_v1_%j.out
#SBATCH --error=/data/run01/scwc189/users/ZhangXinyang/logs/clipreid_csf_fullid40_v1_%j.err
set -euo pipefail
ROOT=/data/run01/scwc189/users/ZhangXinyang
PACKAGE="$ROOT/jobs/clipreid_csf_full_identity_cooldown40_20261003_v1"
PYTHON="$ROOT/envs/uad/bin/python"
STAGE1="$ROOT/outputs/clipreid_scene_prompt_quality_gate_v6_memfix_178528_control/ViT-B-16_stage1_120.pth"
cd "$PACKAGE"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
sha256sum --check --strict SHA256SUMS
"$PYTHON" -B -c 'import torch; assert torch.cuda.device_count()==1; assert "A800" in torch.cuda.get_device_name(0); print("L2SP_SINGLE_A800_CONFIRMED",torch.cuda.get_device_name(0),flush=True)'
"$PYTHON" -B test_l2sp.py --device cuda
"$PYTHON" -B test_l2sp_diagnostics_cuda.py
"$PYTHON" -B test_prototype_comparison_cuda.py
"$PYTHON" -B test_l2sp_epoch_accounting.py
SMOKE="$ROOT/outputs/clipreid_csf_fullid40_v1_smoke_${SLURM_JOB_ID}_seed1"
test ! -e "$SMOKE"
"$PYTHON" -B run_stage2_only.py \
  --data-root "$ROOT/datasets" --weights "$ROOT/pretrained/ViT-B-16.pt" \
  --stage1-checkpoint "$STAGE1" --output "$SMOKE" --seed 1 \
  --csf-weight 0.1 --prototype-method csf_full_identity --l2sp --local-smoke --smoke-six-scenes
"$PYTHON" -B verify_results_csf.py "$SMOKE" --expected-seed 1 --local-smoke
echo "L2SP_A800_SMOKE_VERIFIED output=$SMOKE"
OUTPUT="$ROOT/outputs/clipreid_csf_fullid40_v1_screen40_${SLURM_JOB_ID}_seed1"
test ! -e "$OUTPUT"
"$PYTHON" -B run_stage2_only.py \
  --data-root "$ROOT/datasets" --weights "$ROOT/pretrained/ViT-B-16.pt" \
  --stage1-checkpoint "$STAGE1" --output "$OUTPUT" --seed 1 \
  --csf-weight 0.1 --prototype-method csf_full_identity --l2sp --screening-40
set +e
"$PYTHON" -B screen_against_uad_40.py "$OUTPUT"
GATE_RC=$?
set -e
if [[ "$GATE_RC" -ne 0 && "$GATE_RC" -ne 2 ]]; then
  echo "L2SP_SCREENING_GATE_ERROR rc=$GATE_RC" >&2
  exit "$GATE_RC"
fi
"$PYTHON" -B verify_results_csf.py "$OUTPUT" --expected-seed 1 --screening-40
echo "L2SP_SCREEN40_DONE gate_rc=$GATE_RC output=$OUTPUT"
