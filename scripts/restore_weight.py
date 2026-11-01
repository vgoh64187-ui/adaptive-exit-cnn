"""Prepare one hash-checked inference run from delivered weights and metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def restore(name,weight_dir,destination):
    manifest=json.loads((ROOT/'output/weights/manifest.json').read_text(encoding='utf-8'))
    records=[r for r in manifest if r['run']==name]
    if len(records)!=1:raise ValueError('Unknown run name; see output/weights/manifest.json')
    r=records[0];source=Path(weight_dir)/r['file']
    if not source.is_file() or sha256(source)!=r['sha256']:
        raise RuntimeError(f'Missing or incorrect checkpoint: {source}')
    metadata=json.loads((ROOT/'output/weights/metadata.json').read_text(encoding='utf-8'))[name]
    policy=metadata['policies.json']
    if policy['weights_sha256']!=r['sha256']:raise RuntimeError('Policy and checkpoint do not match')
    dest=Path(destination)/name;dest.mkdir(parents=True,exist_ok=True)
    target=dest/'best.pt'
    if target.exists() and sha256(target)!=r['sha256']:
        raise RuntimeError(f'Refusing to overwrite an unrelated checkpoint: {target}')
    if not target.exists():shutil.copy2(source,target)
    if sha256(target)!=r['sha256']:raise RuntimeError('Copied checkpoint checksum failed')
    for item in ['config.json','policies.json','profile.json','completed.json']:
        path=dest/item;data=metadata[item]
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=data:raise RuntimeError(f'Refusing to overwrite {path}')
        if not path.exists():path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'run':name,'directory':str(dest),'weights_sha256':r['sha256']},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');p.add_argument('--weights-dir',default=str(ROOT/'output/weights'))
    p.add_argument('--destination',default=str(ROOT/'output/inference_runs'));args=p.parse_args()
    restore(args.run,args.weights_dir,args.destination)
