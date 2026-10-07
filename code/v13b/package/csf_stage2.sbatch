#!/bin/bash
#SBATCH --job-name=clipreid_csf_proto_v3
#SBATCH --partition=vip_gpu_a800_scwc189
#SBATCH --qos=gpugpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-4%2
#SBATCH --output=/data/run01/scwc189/users/ZhangXinyang/logs/clipreid_csf_proto_v3_%A_%a.out
#SBATCH --error=/data/run01/scwc189/users/ZhangXinyang/logs/clipreid_csf_proto_v3_%A_%a.err
set -euo pipefail
ROOT=/data/run01/scwc189/users/ZhangXinyang
PACKAGE="$ROOT/jobs/clipreid_csf_prototype_comparison_20261002_v3"
PYTHON="$ROOT/envs/uad/bin/python"
STAGE1="$ROOT/outputs/clipreid_scene_prompt_quality_gate_v6_memfix_178528_control/ViT-B-16_stage1_120.pth"
cd "$PACKAGE"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
sha256sum --check --strict SHA256SUMS

METHOD=$(echo 'control identity balanced_modality six_scene csf' | cut -d ' ' -f "$((SLURM_ARRAY_TASK_ID + 1))")
OUTPUT="$ROOT/outputs/clipreid_csf_prototype_comparison_v3_$SLURM_ARRAY_JOB_ID"_"$METHOD"_seed1
"$PYTHON" -B run_stage2_only.py \
  --data-root "$ROOT/datasets" --weights "$ROOT/pretrained/ViT-B-16.pt" \
  --stage1-checkpoint "$STAGE1" --output "$OUTPUT" --seed 1 \
  --csf-weight 0.1 --prototype-method "$METHOD"
"$PYTHON" -B verify_results_csf.py "$OUTPUT" --expected-seed 1
echo "PROTOTYPE_STAGE2_JOB_OK method=$METHOD output=$OUTPUT"
