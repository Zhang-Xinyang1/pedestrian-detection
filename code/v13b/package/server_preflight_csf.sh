#!/bin/bash
set -euo pipefail
ROOT=/data/run01/scwc189/users/ZhangXinyang
PACKAGE="$ROOT/jobs/clipreid_csf_prototype_comparison_20261002_v3"
PYTHON="$ROOT/envs/uad/bin/python"
STAGE1="$ROOT/outputs/clipreid_scene_prompt_quality_gate_v6_memfix_178528_control/ViT-B-16_stage1_120.pth"

cd "$PACKAGE"
sha256sum --check --strict SHA256SUMS
for script in server_preflight_csf.sh csf_smoke.sbatch csf_stage2.sbatch; do
  bash -n "$script"
done
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4

"$PYTHON" -B -m py_compile factorized_prototype.py scene_stage2_csf.py run_stage2_only.py verify_results_csf.py
"$PYTHON" -B test_factorized_prototype.py
"$PYTHON" -B test_prototype_comparison.py
"$PYTHON" -B test_stage2_csf_contract.py
"$PYTHON" -B test_stage2_only_contract.py
"$PYTHON" -B test_verify_csf_contract.py
"$PYTHON" -B test_fixed_epoch_budget.py
"$PYTHON" -B test_data_protocol.py
"$PYTHON" -B test_stage1_checkpoint_contract.py --checkpoint "$STAGE1"

for METHOD in control identity balanced_modality six_scene csf; do
  "$PYTHON" -B run_stage2_only.py \
    --data-root "$ROOT/datasets" --weights "$ROOT/pretrained/ViT-B-16.pt" \
    --stage1-checkpoint "$STAGE1" \
    --output "$ROOT/outputs/csf_v3_preflight_no_write_$METHOD" \
    --seed 1 --csf-weight 0.1 --prototype-method "$METHOD" --check-only
done
echo "CSF_SERVER_PREFLIGHT_OK"
