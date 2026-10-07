"""Evaluate an archived V13B model with the original retrieval protocol."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'code/v13b/package'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', type=int, choices=[50, 100], default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-images', type=int, help='Engineering forward check only; does not report retrieval scores')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or (args.max_images is not None and args.max_images < 1):
        parser.error('Invalid batch size, worker count, or max-images')
    sys.path.insert(0, str(PACKAGE))
    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Subset
    from yacs.config import CfgNode
    import run_official as base
    from datasets.bases import ImageDataset
    from whu_adapter import WHUMARS
    from whu_metrics import retrieval_metrics, diagnostic_matrices
    if not torch.cuda.is_available():
        raise RuntimeError('The original CLIP-ReID model constructor requires CUDA')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    directory = ROOT / 'artifacts/V13B/production100'
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    cfg = CfgNode(manifest['config'])
    cfg.DATASETS.ROOT_DIR = str(ROOT / 'datasets')
    cfg.freeze()
    dataset = WHUMARS(ROOT / 'datasets')
    assert (len(dataset.train), len(dataset.query), len(dataset.gallery)) == (92133, 6405, 93609)
    assert dataset.train_signature == manifest['info']['train_metadata_sha256']
    weights = ROOT / 'pretrained/ViT-B-16.pt'
    expected_url = base.models.clip._MODELS['ViT-B-16']

    def offline_download(url, *unused, **unused_keywords):
        if url != expected_url:
            raise ValueError('Unexpected CLIP weight URL')
        return str(weights)

    base.models.clip._download = offline_download
    model = base.ScenePromptModel(dataset.num_train_pids, dataset.num_train_cams, dataset.num_train_vids,
        cfg, quality_gate=False)
    base.attach_scene_prompts(model, weights, 'modality')
    checkpoint = directory / f'ViT-B-16_{args.epoch}.pth'
    expected = json.loads((directory / f'checkpoint_audit_stage2_{args.epoch:03d}.json').read_text())['sha256']
    if base.sha(checkpoint) != expected:
        raise ValueError('Checkpoint SHA256 differs')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state, strict=True)
    del state
    model.cuda().eval()
    transform = transforms.Compose([transforms.Resize(cfg.INPUT.SIZE_TEST), transforms.ToTensor(),
        transforms.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)])
    images = ImageDataset(dataset.query + dataset.gallery, transform)
    if args.max_images:
        images = Subset(images, range(min(args.max_images, len(images))))
    loader = DataLoader(images, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        collate_fn=base.loaders.val_collate_fn)
    features = []
    with torch.no_grad():
        for number, (batch, _, _, _, _, _) in enumerate(loader, 1):
            feature = model(batch.cuda(), cam_label=None, view_label=None).float().cpu()
            if feature.shape[1] != 1280 or not torch.isfinite(feature).all():
                raise ValueError('Invalid V13B descriptor')
            features.append(feature)
            if number % 200 == 0:
                print('Evaluation batches:', number, '/', len(loader), flush=True)
    features = torch.cat(features)
    if args.max_images:
        print('V13B_FORWARD_CHECK_PASS', json.dumps(dict(epoch=args.epoch, images=len(features), feature_dim=1280)), flush=True)
        return
    nq = len(dataset.query)
    result = retrieval_metrics(features[:nq], features[nq:], dataset.query_meta, dataset.gallery_meta)
    result['matrices'] = diagnostic_matrices(features[:nq], features[nq:], dataset.query_meta, dataset.gallery_meta)
    result.update(epoch=args.epoch, protocol='WHU_all_same_camera_excluded', feature_dim=1280, local_smoke=False)
    output = args.output or ROOT / 'outputs' / f'eval_v13b_{args.epoch:03d}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('V13B_EVALUATION', json.dumps({k: v for k, v in result.items() if k != 'matrices'}), flush=True)
    print('Saved:', output, flush=True)


if __name__ == '__main__':
    main()

