"""Render Markdown LaTeX expressions to reproducible high-resolution equation assets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
from matplotlib.mathtext import math_to_image
from matplotlib.font_manager import FontProperties
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True)
    args=p.parse_args();source=Path(args.source);raw=source.read_bytes();text=raw.decode('utf-8')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    matplotlib.rcParams['mathtext.fontset']='stix'
    expressions=[(m.group(1),True) for m in re.finditer(r'\\\[(.*?)\\\]',text,re.S)]
    expressions += [(m.group(1),False) for m in re.finditer(r'\\\((.*?)\\\)',text,re.S)]
    entries={}
    for expression,display in expressions:
        expression=' '.join(expression.split())
        key=hashlib.sha256((str(display)+expression).encode()).hexdigest()[:20]
        size=11.5 if display else 10.5
        target=out/f'{key}.png'
        rendered=re.sub(r'\\(?:bigg|Bigg|big|Big)[lr]?', '', expression)
        depth=math_to_image('$'+rendered+'$',target,prop=FontProperties(size=size),dpi=360,format='png',color='black')
        with Image.open(target) as im:width,height=im.size
        entries[key]=dict(expression=expression,display=display,path=str(target.resolve()),
                          width_pt=width*72/360,height_pt=height*72/360,depth_pt=depth)
    (out/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(raw).hexdigest(),
        expressions=entries),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(expressions=len(entries),manifest=str(out/'manifest.json'))))


if __name__=='__main__':main()
