"""Generate experiment settings table from real configuration and split files."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest=Path('configs/main_queue.json');split_path=Path('data/split_cifar100.json')
    run_paths=[Path('runs/baseline_seed42/config.json'),Path('runs/gap_seed42/config.json')]
    spec=json.loads(manifest.read_text());split=json.loads(split_path.read_text())
    configs=[json.loads(p.read_text()) for p in run_paths]
    keys=['epochs','batch_size','workers','lr','weight_decay','temperature','width']
    if any(a[k]!=configs[0][k] for a in configs for k in keys):raise RuntimeError('Completed run settings disagree')
    if any(configs[0][k]!=spec[k] for k in ['epochs','batch_size','workers']):raise RuntimeError('Manifest and actual runs disagree')
    if any(c['split_sha256']!=split['sha256'] for c in configs):raise RuntimeError('Split differs from run configuration')
    groups=defaultdict(list)
    for job in spec['runs']:groups[(job['head'],job['kd'])].append(job['seed'])
    expected={('baseline',0.),('gap',0.),('mlp',0.),('local_global',0.),('local_global',.5)}
    if set(groups)!=expected or any(sorted(x)!=[17,42,73] for x in groups.values()):raise RuntimeError('Incomplete experimental matrix')
    c=configs[0]
    rows=[('数据划分（训练/温度校准/策略选择）',f"{len(split['train']):,} / {len(split['calibration']):,} / {len(split['selection']):,}"),
          ('训练随机种子','17 / 42 / 73'),('模型组数与训练次数',f"{len(groups)} 组 / {len(spec['runs'])} 次"),
          ('训练轮次 / 批量大小',f"{c['epochs']} / {c['batch_size']}"),('SGD 初始学习率 / 权重衰减',f"{c['lr']} / {c['weight_decay']}"),
          ('动量与学习率调度','Nesterov 0.9；前 5 轮预热，随后余弦衰减'),
          ('自蒸馏温度 / 系数',f"{c['temperature']:g} / 0.5（仅融合+蒸馏组）"),
          ('模型选择准则','单出口：验证准确率；多出口：各出口验证准确率均值'),
          ('混合精度','CUDA 训练启用；验证预测缓存与正式推理采用 FP32')]
    out=Path('paper/generated');out.mkdir(parents=True,exist_ok=True)
    table='表 4 数据划分与模型训练设置\n\n| 项目 | 设置 |\n| --- | --- |\n'+''.join(f'| {a} | {b} |\n' for a,b in rows)
    target=out/'training_settings.md';target.write_text(table,encoding='utf-8')
    sources=[manifest,split_path,*run_paths,Path('train.py'),Path('src/data.py')]
    record=dict(scope='predeclared settings verified against completed seed42 run configs; no test performance',
                sources={str(p):sha(p) for p in sources},training_source_sha256=c['source_sha256'],
                split_sha256=split['sha256'],entries=[dict(item=a,value=b) for a,b in rows],output_sha256=sha(target))
    (out/'training_settings.provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(path=str(target),entries=len(rows),runs_declared=len(spec['runs']))))


if __name__=='__main__':main()
