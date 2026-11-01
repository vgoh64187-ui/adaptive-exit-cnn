# 实验权重交付与核验

本目录包含 15 个完整训练运行的 `best.pt` 副本，文件名为 `<运行名>.pt`，清单 `manifest.json` 记录每个文件的 SHA-256 和字节数。它们对应 `results/final/summary.json` 的 `provenance` 中各运行 `checkpoint_sha256`，已逐项比较一致。

仓库忽略 `.pt` 文件；本目录中的权重为本机交付件，不会因克隆 GitHub 仓库自动下载。原始运行目录 `runs/<运行名>/` 还包含配置、策略、日志和缓存；单独复制权重不足以满足冻结协议的完整复算。推理示例需要将相应权重放回配套运行目录并保留其 `config.json` 与 `policies.json`；复现实验可按仓库 README 重新训练并重新冻结。

PowerShell 校验示例：

```powershell
$items = Get-Content output/weights/manifest.json -Raw | ConvertFrom-Json
$items | ForEach-Object {
    $actual = (Get-FileHash (Join-Path output/weights $_.file) -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $_.sha256) { throw "权重校验失败：$($_.file)" }
}
```
