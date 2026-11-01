"""Download only the three raw WSI assets requested by the supplied notebook."""
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
FILES = {'test_067.tif': 2415492970, 'test_068.tif': 2530427892, 'test_068.xml': 76525}

if __name__ == '__main__':
    for name, expected in FILES.items():
        target = ROOT / 'slide' / name
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            if target.stat().st_size != expected:
                raise RuntimeError(f'Existing file has unexpected size: {target}')
            print(f'Already downloaded: {name}', flush=True)
            continue
        partial = target.with_suffix(target.suffix + '.part')
        url = f'https://huggingface.co/datasets/LeonYe/ECE1512-WSI-Feature/resolve/main/slide/{name}'
        print(f'Downloading {name} ({expected / 1e9:.2f} GB)', flush=True)
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            count, last_log = 0, time.monotonic()
            with partial.open('wb') as handle:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    handle.write(chunk)
                    count += len(chunk)
                    if time.monotonic() - last_log > 30:
                        print(f'{name}: {count / expected:.0%}', flush=True)
                        last_log = time.monotonic()
        if count != expected:
            raise RuntimeError(f'Incomplete download: {name}, {count} != {expected}')
        partial.replace(target)
        print(f'Completed {name}', flush=True)
