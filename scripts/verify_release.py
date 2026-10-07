"""Verify the original code, downloaded artifacts, and assembled release."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-state', action='store_true', help='Additionally load and compare full model/optimizer state; needs PyTorch')
    args = parser.parse_args()
    code = ROOT / 'code/v13b'
    source_files = json.loads((code / 'PAYLOAD_SHA256.json').read_text(encoding='utf-8'))
    for name, digest in source_files.items():
        if sha(code / name) != digest:
            raise ValueError('Original source SHA256 differs: ' + name)
    print('V13B_SOURCE_SHA256_PASS', len(source_files), flush=True)
    prod = ROOT / 'artifacts/V13B/production100'
    original = json.loads((prod / 'ARTIFACT_SHA256.json').read_text(encoding='utf-8'))
    omitted = []
    included = 0
    for name, digest in original.items():
        path = prod / name
        if not path.is_file():
            if name in {f'ViT-B-16_{epoch}.pth' for epoch in [10, 20, 30, 40, 60, 70, 80, 90]}:
                omitted.append(name)
                continue
            raise FileNotFoundError('Required original artifact missing: ' + name)
        if sha(path) != digest:
            raise ValueError('Original artifact SHA256 differs: ' + name)
        included += 1
    marker = json.loads((prod / 'VERIFICATION.json').read_text())
    if sha(prod / 'ARTIFACT_SHA256.json') != marker['artifact_manifest_sha256']:
        raise ValueError('Original artifact manifest differs')
    assert marker['status'] == 'PASS' and marker['final_epoch'] == 100
    assert sha(prod / 'ViT-B-16_100.pth') == marker['checkpoint_sha256']
    assert sha(prod / 'training_state_latest.pth') == marker['training_state']['sha256']
    print('V13B_INCLUDED_ARTIFACT_SHA256_PASS', included, flush=True)
    print('Intermediate checkpoint models preserved on original server:', ', '.join(omitted), flush=True)
    manifest_path = ROOT / 'RELEASE_MANIFEST.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('RELEASE_MANIFEST.json is required')
    release = json.loads(manifest_path.read_text(encoding='utf-8'))
    for name, item in release['files'].items():
        path = ROOT / name
        if not path.is_file() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Release file differs: ' + name)
    print('CVPR_RELEASE_SHA256_PASS', len(release['files']), flush=True)
    if args.model_state:
        import torch
        import numpy as np
        torch.set_num_threads(4)
        state = torch.load(str(prod / 'training_state_latest.pth'), map_location='cpu', weights_only=False, mmap=True)
        main_model = torch.load(str(prod / 'ViT-B-16_100.pth'), map_location='cpu', weights_only=False, mmap=True)
        audit = json.loads((prod / 'stage2_audit.json').read_text())
        assert state['schema'] == 'V13_full_epoch_boundary_v1' and state['epoch'] == 100 and state['lr_tail_arm'] == 'B'
        assert state['model'].keys() == main_model.keys()
        for name, tensor in main_model.items():
            if not torch.equal(tensor, state['model'][name]):
                raise ValueError('Complete state/model tensor differs: ' + name)
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError('Nonfinite model tensor: ' + name)
        for name, digest in audit['final_parameters'].items():
            if hashlib.sha256(state['model'][name].contiguous().numpy().tobytes()).hexdigest() != digest:
                raise ValueError('Formal parameter audit differs: ' + name)
        assert state['ledger']['actual_batches_per_epoch'] == audit['actual_batches_per_epoch']
        assert state['ledger']['successful_updates'] == audit['optimizer_steps']
        assert state['ledger']['learning_rate_history'] == audit['csf']['learning_rate_history']
        assert state['scheduler']['last_epoch'] == 100
        assert state['optimizer']['state'] and state['AMP']
        assert {'python', 'numpy', 'torch_cpu', 'torch_cuda'} <= state['rng'].keys()
        assert int(state['global_memory']['successful_updates']) == audit['optimizer_steps']
        print('V13B_FULL_TRAINING_STATE_PASS', json.dumps(dict(epoch=100, tensor_count=len(main_model),
            successful_updates=audit['optimizer_steps'], components=list(state))), flush=True)
    print('CVPR_RELEASE_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
