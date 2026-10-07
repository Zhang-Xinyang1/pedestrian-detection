#!/bin/bash
#SBATCH -J clipreid_xmodal_v1
#SBATCH -p vip_gpu_a800_scwc189
#SBATCH --qos=gpugpu
#SBATCH -N1 -n1
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --array=0-2%3
#SBATCH -o /data/run01/scwc189/users/ZhangXinyang/logs/%x_%A_%a.out
#SBATCH -e /data/run01/scwc189/users/ZhangXinyang/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/data/run01/scwc189/users/ZhangXinyang
PACKAGE="$ROOT/jobs/clipreid_xmodal_alignment_20261002_v1"
test -n "$SLURM_ARRAY_TASK_ID"
case "$SLURM_ARRAY_TASK_ID" in
  0) WEIGHT=0.05; TAG=005 ;;
  1) WEIGHT=0.1; TAG=010 ;;
  2) WEIGHT=0.2; TAG=020 ;;
  *) echo "Unexpected task id" >&2; exit 2 ;;
esac
cd "$PACKAGE"
sha256sum --check --strict SHA256SUMS
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
printf -v OUTPUT '%s/outputs/clipreid_xmodal_alignment_v1_%s_w%s' "$ROOT" "$SLURM_ARRAY_JOB_ID" "$TAG"
test ! -e "$OUTPUT"
echo "XMODAL_JOB_START weight=$WEIGHT data_protocol=unchanged_official_P16K4"
"$ROOT/envs/uad/bin/python" -B "$PACKAGE/run_official.py" --data-root "$ROOT/datasets" --weights "$ROOT/pretrained/ViT-B-16.pt" --prompt-mode modality --variant control --i2t-weight 1.0 --xmodal-weight "$WEIGHT" --output "$OUTPUT"
"$ROOT/envs/uad/bin/python" -B "$PACKAGE/verify_results.py" "$OUTPUT"
echo "XMODAL_JOB_DONE weight=$WEIGHT output=$OUTPUT"
