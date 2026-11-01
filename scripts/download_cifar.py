"""Resumable ranged download from the official CIFAR server, verified by MD5."""
import concurrent.futures
import hashlib
import json
import time
import requests
from pathlib import Path
from torchvision.datasets import CIFAR100


URL = 'https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz'
SIZE = 169001437
CHUNK = 2 * 1024 * 1024


def fetch(i):
    start, end = i * CHUNK, min((i + 1) * CHUNK, SIZE) - 1
    path = Path('data/parts') / f'{i:04d}.part'
    if path.exists() and path.stat().st_size == end - start + 1:
        return i
    for attempt in range(8):
        try:
            with requests.get(URL, headers={'Range': f'bytes={start}-{end}'}, timeout=(15,120)) as response:
                if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
                    raise RuntimeError('Server range response mismatch')
                data = response.content
            if len(data) != end - start + 1:
                raise RuntimeError('Incomplete range')
            path.write_bytes(data)
            return i
        except Exception:
            if attempt == 7:
                raise
            time.sleep(attempt + 1)


def main():
    Path('data/parts').mkdir(parents=True, exist_ok=True)
    target = Path('data') / CIFAR100.filename
    if target.exists() and hashlib.md5(target.read_bytes()).hexdigest() == CIFAR100.tgz_md5:
        print('Archive already verified', flush=True)
        return
    n = (SIZE + CHUNK - 1) // CHUNK
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for completed, future in enumerate(concurrent.futures.as_completed([pool.submit(fetch, i) for i in range(n)]), 1):
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
            print(json.dumps(dict(parts_completed=completed, total=n)), flush=True)
    if errors:
        raise RuntimeError(f'{len(errors)} ranges failed; rerun to resume: {errors[0]}')
    temp = target.with_suffix('.verified.tmp')
    with temp.open('wb') as f:
        for i in range(n):
            f.write((Path('data/parts') / f'{i:04d}.part').read_bytes())
    checksum = hashlib.md5(temp.read_bytes()).hexdigest()
    if checksum != CIFAR100.tgz_md5:
        raise RuntimeError(f'Archive MD5 mismatch: {checksum}')
    temp.replace(target)
    (Path('data') / 'download_provenance.json').write_text(json.dumps(dict(
        url=URL, bytes=SIZE, md5=checksum, verified=True), indent=2))
    print('Official archive MD5 verified', flush=True)


if __name__ == '__main__':
    main()
