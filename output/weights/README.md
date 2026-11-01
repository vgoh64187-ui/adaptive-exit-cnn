# 实验权重交付与核验

本目录包含 15 个完整训练运行的 `best.pt` 副本，文件名为 `<运行名>.pt`，清单 `manifest.json` 记录每个文件的 SHA-256 和字节数。它们对应 `results/final/summary.json` 的 `provenance` 中各运行 `checkpoint_sha256`，已逐项比较一致。

仓库忽略 `.pt` 文件；本目录中的权重为本机交付件，不会因克隆 GitHub 仓库自动下载。`metadata.json` 保存对应配置、策略与结构统计。原始运行目录 `runs/<运行名>/` 还包含日志和缓存；以下恢复包支持推理，冻结协议的完整复算仍需原始运行记录。复现实验可按仓库 README 重新训练并重新冻结。

PowerShell 校验示例：

```powershell
$items = Get-Content output/weights/manifest.json -Raw | ConvertFrom-Json
$items | ForEach-Object {
    $actual = (Get-FileHash (Join-Path output/weights $_.file) -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $_.sha256) { throw "权重校验失败：$($_.file)" }
}
```

恢复并运行一个模型（先在仓库根目录按 README 安装依赖）：

```powershell
.venv/Scripts/python.exe scripts/download_cifar.py
.venv/Scripts/python.exe -m src.data
.venv/Scripts/python.exe scripts/restore_weight.py lg_kd_seed42
.venv/Scripts/python.exe demo.py output/inference_runs/lg_kd_seed42 --sample-index 3 --budget .75
.venv/Scripts/python.exe serve_demo.py output/inference_runs/lg_kd_seed42 --device cpu
```

权重不在默认位置时，为恢复命令追加 `--weights-dir 路径`。命令会核对权重 SHA-256 和策略绑定，发现目标目录已有不同文件时停止。可用运行名见 `manifest.json`。
