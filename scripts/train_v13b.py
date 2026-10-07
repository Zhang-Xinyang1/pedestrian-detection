"""Invoke the unchanged V13B code using paths relative to this repository."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'code/v13b'
PACKAGE = RELEASE / 'package'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['check', 'smoke', 'train', 'pipeline'], default='check')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--site', type=Path, help='Existing dependencies for the original pipeline')
    args = parser.parse_args()
    output = (args.output or ROOT / 'outputs' / ('v13b_' + args.mode)).resolve()
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONNOUSERSITE='1')
    environment['PYTHONPATH'] = os.pathsep.join([str(PACKAGE), str(PACKAGE / 'upstream')])
    if args.mode == 'pipeline':
        command = [sys.executable, '-u', '-B', str(RELEASE / 'run_pipeline.py'),
            '--project-root', str(ROOT), '--data-root', str(ROOT / 'datasets'),
            '--runtime-root', str(output), '--arm', 'B']
        if args.site:
            command += ['--site', str(args.site.resolve())]
    else:
        if args.mode != 'check' and output.exists():
            parser.error('Output already exists; use a new --output to preserve the previous run.')
        command = [sys.executable, '-u', '-B', str(PACKAGE / 'run_stage2_only.py'),
            '--data-root', str(ROOT / 'datasets'), '--weights', str(ROOT / 'pretrained/ViT-B-16.pt'),
            '--stage1-checkpoint', str(ROOT / 'pretrained/ViT-B-16_stage1_120.pth'),
            '--output', str(output), '--seed', '1', '--csf-weight', '0.1',
            '--prototype-method', 'identity_full', '--l2sp', '--local-observed', '--lr-tail-arm', 'B']
        if args.mode == 'smoke':
            command += ['--local-smoke', '--smoke-six-scenes', '--smoke-epochs', '4']
        else:
            command += ['--fixed100-ab']
            if args.mode == 'check':
                command += ['--check-only']
    print('V13B mode:', args.mode, flush=True)
    result = subprocess.run(command, cwd=PACKAGE, env=environment)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

