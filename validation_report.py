"""Export completed-run validation evidence, explicitly separate from test results."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from inference import file_sha256


def export(run,output):
    run=Path(run)
    config=json.loads((run/'config.json').read_text())
    completed=json.loads((run/'completed.json').read_text())
    if config['smoke_only'] or completed['epochs']!=config['epochs']:
        raise RuntimeError('Only completed non-smoke runs can enter validation review')
    metadata=json.loads((run/'validation_predictions.metadata.json').read_text())
    cache=run/'validation_predictions.npz'
    weights=file_sha256(run/'best.pt')
    if metadata['weights_sha256']!=weights or metadata['prediction_sha256']!=file_sha256(cache):
        raise RuntimeError('Stale validation artifacts')
    data=np.load(cache)
    labels=data['labels']
    if len(labels)!=5000 or metadata['n']!=5000:raise RuntimeError('Incomplete validation population')
    logits=sorted([key for key in data.files if key.startswith('logits_')])
    accuracies=[dict(output=i+1,correct=int((data[key].argmax(1)==labels).sum()),
                     n=len(labels),accuracy=float((data[key].argmax(1)==labels).mean())) for i,key in enumerate(logits)]
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    sources={f:file_sha256(run/f) for f in ['config.json','history.jsonl','best.pt','policies.json',
                                          'validation_predictions.npz','validation_predictions.metadata.json']}
    result=dict(run=run.name,head=config['head'],kd=config['kd'],seed=config['seed'],
                completed_epochs=completed['epochs'],selected_epoch=checkpoint['epoch']+1,
                validation_outputs=accuracies,validation_precision='FP32',
                training_selection_score=completed['best_validation_score'],
                training_selection_precision='AMP on CUDA; see training environment',
                scope='5000 validation images used for model development, NOT final official test accuracy',
                profile=json.loads((run/'profile.json').read_text()),
                split_sha256=config['split_sha256'],sources=sources)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    target=out/f'{run.name}.json';target.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(path=str(target),selected_epoch=result['selected_epoch'],validation_outputs=accuracies)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',default='results/validation')
    args=p.parse_args();export(args.run,args.output)
