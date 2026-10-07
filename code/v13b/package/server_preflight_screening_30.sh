#!/bin/bash
set -euo pipefail
BASE=/data/run01/scwc189/users
if [[ -d "$BASE/ZhangXinyang" ]]; then ROOT="$BASE/ZhangXinyang"; else ROOT=$(find "$BASE" -maxdepth 1 -type d -name 'ZhangXinyang-*' -print -quit); fi
test -n "$ROOT"
PACKAGE="$ROOT/jobs/clipreid_csf_screening_20261002_v2_balanced_modality"
PYTHON="$ROOT/envs/uad/bin/python"
STAGE1="$ROOT/outputs/clipreid_scene_prompt_quality_gate_v6_memfix_178528_control/ViT-B-16_stage1_120.pth"
cd "$PACKAGE"
sha256sum --check --strict SHA256SUMS
bash -n screen_balanced_modality_30.sbatch
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
"$PYTHON" -B -m py_compile run_stage2_only.py verify_results_csf.py screen_against_uad.py
"$PYTHON" -B test_factorized_prototype.py
"$PYTHON" -B test_prototype_comparison.py
"$PYTHON" -B test_stage2_csf_contract.py
"$PYTHON" -B test_stage2_only_contract.py
"$PYTHON" -B test_verify_csf_contract.py
"$PYTHON" -B test_data_protocol.py
"$PYTHON" -B test_stage1_checkpoint_contract.py --checkpoint "$STAGE1"
"$PYTHON" -B run_stage2_only.py   --data-root "$ROOT/datasets" --weights "$ROOT/pretrained/ViT-B-16.pt"   --stage1-checkpoint "$STAGE1" --output "$ROOT/outputs/csf_screen30_preflight_no_write_balanced_modality"   --seed 1 --csf-weight 0.1 --prototype-method balanced_modality --screening-30 --check-only
echo "CSF_SCREENING_30_PREFLIGHT_OK"
