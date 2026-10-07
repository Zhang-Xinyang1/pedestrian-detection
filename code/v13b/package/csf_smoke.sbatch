#!/bin/bash
#SBATCH --job-name=clipreid_csf_proto_v3_smoke
#SBATCH --partition=vip_gpu_a800_scwc189
#SBATCH --qos=gpugpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/data/run01/scwc189/users/ZhangXinyang/logs/clipreid_csf_proto_v3_smoke_%j.out
#SBATCH --error=/data/run01/scwc189/users/ZhangXinyang/logs/clipreid_csf_proto_v3_smoke_%j.err
set -euo pipefail
ROOT=/data/run01/scwc189/users/ZhangXinyang
PACKAGE="$ROOT/jobs/clipreid_csf_prototype_comparison_20261002_v3"
PYTHON="$ROOT/envs/uad/bin/python"
STAGE1="$ROOT/outputs/clipreid_scene_prompt_quality_gate_v6_memfix_178528_control/ViT-B-16_stage1_120.pth"
cd "$PACKAGE"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
sha256sum --check --strict SHA256SUMS

PREFIX="$ROOT/outputs/clipreid_csf_prototype_comparison_v3_smoke_$SLURM_JOB_ID"
"$PYTHON" -B -c 'import torch; assert torch.cuda.device_count()==1; assert "A800" in torch.cuda.get_device_name(0); print("PROTOTYPE_A800_CONFIRMED")'
"$PYTHON" -B test_prototype_comparison_cuda.py
for METHOD in control identity balanced_modality six_scene csf; do
  OUTPUT="$PREFIX"_"$METHOD"
  "$PYTHON" -B run_stage2_only.py \
    --data-root "$ROOT/datasets" --weights "$ROOT/pretrained/ViT-B-16.pt" \
    --stage1-checkpoint "$STAGE1" --output "$OUTPUT" --seed 1 \
    --csf-weight 0.1 --prototype-method "$METHOD" --local-smoke --smoke-six-scenes
  "$PYTHON" -B verify_results_csf.py "$OUTPUT" --expected-seed 1 --local-smoke
done
"$PYTHON" -B verify_comparison_pair.py "$PREFIX"
echo "PROTOTYPE_COMPARISON_A800_SMOKE_OK"
