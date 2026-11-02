"""Download the exact feature files, or verify existing files without network access."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[1]


def verify_file(path, record):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing feature file: {path}. Run scripts/download_features.py first.')
    if path.stat().st_size != record['size_bytes']:
        raise ValueError(f'Unexpected size for {path}; expected {record["size_bytes"]} bytes.')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != record['sha256']:
        raise ValueError(f'SHA256 mismatch for {path}. Keep this file separate and obtain the recorded features.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, default=ROOT)
    parser.add_argument('--datasets', nargs='+', choices=['C16', 'C17', 'BRACS'], default=['C16', 'C17', 'BRACS'])
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--verify-only', action='store_true', help='Check size and SHA256 locally; never download.')
    action.add_argument('--dry-run', action='store_true', help='Print immutable URLs and destination paths only.')
    args = parser.parse_args(argv)
    root = args.project_dir.resolve()
    manifest = json.loads((ROOT / 'data_manifest.json').read_text(encoding='utf-8'))
    base = f'https://huggingface.co/datasets/LeonYe/ECE1512-WSI-Feature/resolve/{manifest["huggingface_revision"]}'
    for dataset in args.datasets:
        record = manifest['datasets'][dataset]
        target = root / record['path']
        url = f'{base}/{record["remote_path"]}'
        if args.dry_run:
            print(f'{dataset}: {url}\n  -> {target} ({record["size_bytes"]} bytes)', flush=True)
            continue
        if target.exists() or args.verify_only:
            verify_file(target, record)
            print(f'{dataset}: size and SHA256 verified: {target}', flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + '.part')
        print(f'{dataset}: downloading {record["size_bytes"] / 1e9:.2f} GB', flush=True)
        digest, count, last_log = hashlib.sha256(), 0, time.monotonic()
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with partial.open('wb') as handle:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    handle.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
                    if time.monotonic() - last_log > 30:
                        print(f'{dataset}: {count / record["size_bytes"]:.0%}', flush=True)
                        last_log = time.monotonic()
        if count != record['size_bytes'] or digest.hexdigest() != record['sha256']:
            raise ValueError(f'Download failed size/SHA256 verification: {partial}')
        partial.replace(target)
        print(f'{dataset}: downloaded and verified: {target}', flush=True)


if __name__ == '__main__':
    main()
