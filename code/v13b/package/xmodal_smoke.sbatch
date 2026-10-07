#!/bin/bash
#SBATCH -J clipreid_xmodal_v1_smoke
#SBATCH -p vip_gpu_a800_scwc189
#SBATCH --qos=gpugpu
#SBATCH -N1 -n1
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -o /data/run01/scwc189/users/ZhangXinyang/logs/%x_%j.out
#SBATCH -e /data/run01/scwc189/users/ZhangXinyang/logs/%x_%j.err
set -euo pipefail
ROOT=/data/run01/scwc189/users/ZhangXinyang
PACKAGE="$ROOT/jobs/clipreid_xmodal_alignment_20261002_v1"
cd "$PACKAGE"
sha256sum --check --strict SHA256SUMS
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$ROOT/envs/uad/bin/python" -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; print('XMODAL_CUDA_OK',torch.cuda.get_device_name(0))"
"$ROOT/envs/uad/bin/python" -B "$PACKAGE/test_xmodal_alignment.py" --cuda
"$ROOT/envs/uad/bin/python" -B "$PACKAGE/test_pretrained.py" --weights "$ROOT/pretrained/ViT-B-16.pt"
TMP_OUTPUT="/tmp/clipreid_xmodal_v1_smoke_$SLURM_JOB_ID"
test ! -e "$TMP_OUTPUT"
"$ROOT/envs/uad/bin/python" -B "$PACKAGE/run_official.py" --data-root "$ROOT/datasets" --weights "$ROOT/pretrained/ViT-B-16.pt" --prompt-mode modality --variant control --i2t-weight 1.0 --xmodal-weight 0.1 --local-smoke --smoke-six-scenes --output "$TMP_OUTPUT"
"$ROOT/envs/uad/bin/python" -B "$PACKAGE/verify_results.py" "$TMP_OUTPUT" --local-smoke
echo "XMODAL_GPU_SMOKE_OK"
