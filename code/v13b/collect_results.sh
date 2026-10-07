#!/bin/sh
set -eu
PROJECT_ROOT="${PROJECT_ROOT:-/work/home/luhanning/prvc}"
ARM="${2:-}"
[ "${1:-}" = '--arm' ] || { echo 'usage: collect_results.sh --arm A|B' >&2; exit 2; }
case "$ARM" in
 A) NAME=V13A_v10_constant_tail_fixed100_seed1 ;;
 B) NAME=V13B_v10_cosine_tail_fixed100_seed1 ;;
 *) echo 'arm must be A or B' >&2; exit 2 ;;
esac
RUN_REL="scnet_v13_lr100_ab_runs/$NAME"
RUN="$PROJECT_ROOT/$RUN_REL"
[ -d "$RUN" ] || { echo 'V13 run not found' >&2; exit 2; }
TAG="SCNET_V13${ARM}_EVIDENCE_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
find "$RUN/smoke" "$RUN/production100" -maxdepth 1 -type f -name '*.pth' -exec sha256sum {} \; > "$PROJECT_ROOT/${TAG}_checkpoint_sha256.txt"
tar --exclude='*.pth' --exclude='*.pyc' --exclude='__pycache__' --exclude='wheels' -czf "$PROJECT_ROOT/$TAG.tar.gz" -C "$PROJECT_ROOT" "$RUN_REL" SCNET_REPAIR_V13_LR100_AB_20261005 "${TAG}_checkpoint_sha256.txt"
sha256sum "$PROJECT_ROOT/$TAG.tar.gz"
printf 'Evidence: %s\n' "$PROJECT_ROOT/$TAG.tar.gz"
printf 'Final model: %s\n' "$RUN/production100/ViT-B-16_100.pth"
printf 'Full state: %s\n' "$RUN/production100/training_state_latest.pth"
