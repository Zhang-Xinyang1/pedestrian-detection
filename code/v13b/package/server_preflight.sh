#!/bin/bash
set -euo pipefail
ROOT=/data/run01/scwc189/users/ZhangXinyang
PACKAGE="$ROOT/jobs/clipreid_xmodal_alignment_20261002_v1"
cd "$PACKAGE"
sha256sum --check --strict SHA256SUMS
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
"$ROOT/envs/uad/bin/python" -B "$PACKAGE/test_data_protocol.py"
"$ROOT/envs/uad/bin/python" -B "$PACKAGE/test_xmodal_alignment.py"
"$ROOT/envs/uad/bin/python" -B "$PACKAGE/test_prompts.py"
"$ROOT/envs/uad/bin/python" -B "$PACKAGE/test_adapter.py"
"$ROOT/envs/uad/bin/python" -B "$PACKAGE/test_i2t.py"
"$ROOT/envs/uad/bin/python" -B "$PACKAGE/test_quality_gate.py"
"$ROOT/envs/uad/bin/python" -B "$PACKAGE/test_worker_ipc.py"
"$ROOT/envs/uad/bin/python" -B "$PACKAGE/run_official.py" --data-root "$ROOT/datasets" --weights "$ROOT/pretrained/ViT-B-16.pt" --prompt-mode modality --variant control --i2t-weight 1.0 --xmodal-weight 0.05 --output "$ROOT/outputs/clipreid_xmodal_v1_preflight_w005" --check-only
"$ROOT/envs/uad/bin/python" -B "$PACKAGE/run_official.py" --data-root "$ROOT/datasets" --weights "$ROOT/pretrained/ViT-B-16.pt" --prompt-mode modality --variant control --i2t-weight 1.0 --xmodal-weight 0.1 --output "$ROOT/outputs/clipreid_xmodal_v1_preflight_w010" --check-only
"$ROOT/envs/uad/bin/python" -B "$PACKAGE/run_official.py" --data-root "$ROOT/datasets" --weights "$ROOT/pretrained/ViT-B-16.pt" --prompt-mode modality --variant control --i2t-weight 1.0 --xmodal-weight 0.2 --output "$ROOT/outputs/clipreid_xmodal_v1_preflight_w020" --check-only
echo "XMODAL_SERVER_PREFLIGHT_OK (data/sampler unchanged; no training or output writes)"
