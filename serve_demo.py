"""Serve a local interactive classifier with the same real dynamic path as the CLI.

No cloud calls or extra web frameworks. Development weights require an explicit
flag and are labelled in the UI and every output record.
"""
import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import datasets, transforms
from demo import classify
from inference import load_run, load_policy
from src.data import MEAN, STD
from src.models import profile_macs


class Application:
    def __init__(self, run, data='data', device='cpu', development=False):
        self.run=Path(run);self.device=device;self.development=development
        config=json.loads((self.run/'config.json').read_text())
        if not development and (config['smoke_only'] or not (self.run/'completed.json').exists()):
            raise RuntimeError('Use completed trained weights, or explicitly pass --development')
        self.model,self.config,self.weights_sha=load_run(run,device)
        self.costs=profile_macs(self.model)
        self.dataset=datasets.CIFAR100(data,train=True,download=False)
        self.split=json.loads((Path(data)/'split_cifar100.json').read_text())
        if self.split['sha256']!=config['split_sha256']:
            raise RuntimeError('Application dataset split differs from training')
        self.indices=sorted(self.split['selection'])
        self.policies=json.loads((self.run/'policies.json').read_text())
        if self.policies['weights_sha256']!=self.weights_sha:
            raise RuntimeError('Application weights/policies mismatch')
        self.transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),
                                          transforms.Normalize(MEAN,STD)])

    def info(self):
        budgets={str(consensus).lower(): sorted({p['budget'] for p in self.policies['policies']
                 if p['criterion']=='budget' and 'policy' in p and p['policy']['consensus']==consensus})
                 for consensus in [False,True]}
        return dict(run=self.run.name,model=self.config['model'],sample_count=len(self.indices),
                    budgets=budgets,has_exits=bool(self.model.exits),device=self.device,
                    development=self.development,smoke_only=self.config['smoke_only'])

    def sample(self,index):
        if not 0<=index<len(self.indices):raise ValueError('Sample index outside validation selection set')
        original=self.indices[index]
        image,label=self.dataset[original]
        return image,dict(type='validation_selection',index=original,selection_position=index),dict(
                         class_id=label,label=self.dataset.classes[label])

    @staticmethod
    def image_url(image):
        buffer=io.BytesIO();image.save(buffer,format='PNG')
        return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')

    def predict(self,body):
        truth=None
        if body.get('image_data'):
            value=body['image_data']
            if not isinstance(value,str) or not value.startswith('data:image/') or ';base64,' not in value:
                raise ValueError('Expected a base64 image')
            raw=base64.b64decode(value.split(',',1)[1],validate=True)
            with Image.open(io.BytesIO(raw)) as source:
                image=ImageOps.exif_transpose(source).convert('RGB')
            source_info=dict(type='uploaded_image')
        else:
            image,source_info,truth=self.sample(int(body.get('sample_index',0)))
        if not isinstance(body.get('consensus',False),bool) or not isinstance(body.get('full_depth',False),bool):
            raise ValueError('Invalid mode fields')
        full_depth=body.get('full_depth',False) or not self.model.exits
        if full_depth:
            policy,selected=None,None
        else:
            budget=float(body.get('budget',.75))
            policy,selected=load_policy(self.run,self.weights_sha,budget,body.get('consensus',False))
        result=classify(self.model,self.transform(image),policy,self.dataset.classes,self.device,full_depth)
        macs=self.costs['backbone_macs'] if full_depth else self.costs['exit_macs'][result['exit']-1]
        result.update(source=source_info,ground_truth=truth,weights_sha256=self.weights_sha,
                      run=self.run.name,model=self.config['model'],device=self.device,
                      cumulative_macs=macs,cost_ratio=macs/self.costs['backbone_macs'],
                      full_depth=full_depth,selected_policy=selected,
                      image_data=self.image_url(image.resize((32,32))),
                      smoke_only=self.config['smoke_only'],development=self.development,
                      probability_mode='raw softmax' if full_depth else 'validation temperature scaled',
                      timing_note='Single observation; not a benchmark; excludes image preprocessing',
                      scope='100 CIFAR classes; validation average budget does not bound each image')
        return result


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,body,content_type='application/json; charset=utf-8'):
            raw=json.dumps(body,ensure_ascii=False).encode('utf-8') if isinstance(body,(dict,list)) else body
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store')
            self.end_headers();self.wfile.write(raw)

        def do_GET(self):
            url=urlparse(self.path)
            if url.path=='/':
                self.reply(200,(Path(__file__).parent/'app'/'index.html').read_bytes(),'text/html; charset=utf-8')
            elif url.path=='/api/info':self.reply(200,app.info())
            elif url.path=='/api/sample':
                try:
                    image,source,truth=app.sample(int(parse_qs(url.query).get('index',['0'])[0]))
                    self.reply(200,dict(image_data=app.image_url(image),source=source,ground_truth=truth))
                except (ValueError,IndexError) as e:self.reply(400,dict(error=str(e)))
            else:self.reply(404,dict(error='Not found'))

        def do_POST(self):
            if self.path!='/api/predict':
                self.reply(404,dict(error='Not found'));return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=8*1024*1024:raise ValueError('Request must be between 1 byte and 8 MB')
                body=json.loads(self.rfile.read(length))
                if not isinstance(body,dict):raise ValueError('Expected JSON object')
                self.reply(200,app.predict(body))
            except (ValueError,TypeError,KeyError,UnidentifiedImageError,Image.DecompressionBombError) as e:
                self.reply(400,dict(error=str(e)))

        def log_message(self,format,*args):
            print(f'HTTP {format % args}',flush=True)
    return Handler


def main():
    p=argparse.ArgumentParser(description='本地预算可调图像分类界面')
    p.add_argument('run');p.add_argument('--data',default='data')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--port',type=int,default=8765);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--development',action='store_true')
    args=p.parse_args();torch.set_num_threads(args.threads)
    app=Application(args.run,args.data,args.device,args.development)
    server=HTTPServer(('127.0.0.1',args.port),handler_for(app))
    print(json.dumps(dict(url=f'http://127.0.0.1:{server.server_port}',run=args.run,
                          development=args.development)),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
