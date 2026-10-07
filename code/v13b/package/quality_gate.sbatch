#!/bin/bash
# Matched control / quality-gate array. Submit manually after server_preflight.sh.
#SBATCH -J clipreid_quality_gate
#SBATCH -p vip_gpu_a800_scwc189
#SBATCH --qos=gpugpu
#SBATCH -N1 -n1
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --array=0-1%1
#SBATCH -o /data/run01/scwc189/users/ZhangXinyang/logs/%x_%A_%a.out
#SBATCH -e /data/run01/scwc189/users/ZhangXinyang/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/data/run01/scwc189/users/ZhangXinyang
PACKAGE="$ROOT/jobs/clipreid_xmodal_alignment_20261002_v1"
: "${SLURM_ARRAY_JOB_ID:?Submit with sbatch}"
: "${SLURM_ARRAY_TASK_ID:?Expected array task}"
VARIANTS=(control quality_gate)
VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
cd "$PACKAGE"
sha256sum --check --strict SHA256SUMS
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
echo "SCENE_PROMPT_OPEN_FILE_LIMIT soft=$(ulimit -Sn) hard=$(ulimit -Hn)"
OUTPUT="$ROOT/outputs/clipreid_xmodal_v1_legacy_quality_gate_${SLURM_ARRAY_JOB_ID}_${VARIANT}"
test ! -e "$OUTPUT"
echo "SCENE_PROMPT_QUALITY_GATE_JOB_START variant=$VARIANT"
"$ROOT/envs/uad/bin/python" -B "$PACKAGE/run_official.py" \
  --data-root "$ROOT/datasets" --weights "$ROOT/pretrained/ViT-B-16.pt" \
  --prompt-mode modality --variant "$VARIANT" --i2t-weight 1.0 --output "$OUTPUT"
"$ROOT/envs/uad/bin/python" -B "$PACKAGE/verify_results.py" "$OUTPUT"
echo "SCENE_PROMPT_QUALITY_GATE_JOB_DONE output=$OUTPUT"
