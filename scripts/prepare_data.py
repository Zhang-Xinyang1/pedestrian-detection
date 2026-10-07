"""Safely unpack the exact local dataset archive and validate image hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    if path.stat().st_size < 1024 * 1024:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'datasets/WHU-MARS-v13b.zip')
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--verify-images', action='store_true')
    args = parser.parse_args()
    data = ROOT / 'datasets'
    manifest = json.loads((data / 'DATASET_MANIFEST.json').read_text(encoding='utf-8'))
    if args.extract:
        if sha(args.archive) != manifest['archive_sha256']:
            raise ValueError('Dataset archive SHA256 differs')
        with zipfile.ZipFile(args.archive) as archive:
            for entry in archive.infolist():
                target = (data / entry.filename).resolve()
                if data.resolve() not in target.parents or entry.is_dir():
                    raise ValueError('Unsafe dataset archive path')
                if target.exists():
                    raise FileExistsError('Refusing to overwrite existing data: ' + str(target))
            archive.extractall(data)
    counts = {split: {modality: len(list((data / 'WHU-MARS' / split / modality).glob('*.jpg')))
        for modality in ['RGB', 'IR', 'Thermal']} for split in manifest['counts']}
    if counts != manifest['counts']:
        raise ValueError('Dataset split counts differ from the V13B dataset')
    if args.verify_images:
        image_manifest = data / manifest['image_manifest']
        if sha(image_manifest) != manifest['image_manifest_sha256']:
            raise ValueError('Image checksum manifest differs')
        for number, line in enumerate(image_manifest.read_text(encoding='utf-8').splitlines(), 1):
            digest, name = line.split('  ', 1)
            path = (data / name).resolve()
            if data.resolve() not in path.parents or sha(path) != digest:
                raise ValueError('Image SHA256 differs: ' + name)
            if number % 30000 == 0:
                print('Images verified:', number, flush=True)
    print('V13B_DATASET_PASS', json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
