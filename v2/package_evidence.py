"""Package audited, lightweight V2 evidence without weights or per-image caches."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

root = Path.cwd()
dest = root / "release/v2-evidence"
staging = root / "release/v2-evidence.staging"
lock_path = root / "runs_v2/v2_screening_lock.json"
audit_path = root / "results/v2/v2_full_audit.json"
summary_path = root / "results/v2/final_comparison.json"
paper_audit_path = root / "results/v2/paper_verification.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if dest.exists() or staging.exists():
    raise RuntimeError("Evidence destination or staging directory exists; inspect it first")
lock = json.loads(lock_path.read_text(encoding="utf-8"))
audits = json.loads(audit_path.read_text(encoding="utf-8"))
summary = json.loads(summary_path.read_text(encoding="utf-8"))
paper_audit = json.loads(paper_audit_path.read_text(encoding="utf-8"))
if lock.get("status") != "frozen" or len(lock["runs"]) != 7:
    raise RuntimeError("Expected the full seven-run frozen screening protocol")
if summary["locked_v2_protocol_sha256"] != sha(lock_path):
    raise RuntimeError("Comparison summary is not bound to the frozen protocol")
if paper_audit["summary_sha256"] != sha(summary_path):
    raise RuntimeError("Word verification is not bound to the summary")
if paper_audit["source_sha256"] != sha(root / "paper/report_v2.md"):
    raise RuntimeError("Paper source changed after Word verification")
if paper_audit["word_sha256"] != sha(root / "output/基于ResNet18的提前退出CNN课程论文-V2实验版.docx"):
    raise RuntimeError("Word document changed after verification")
if {row["run"] for row in audits["runs"]} != set(lock["runs"]):
    raise RuntimeError("Audit does not cover every frozen V2 run")
for source, expected in lock["sources"].items():
    if sha(root / source) != expected:
        raise RuntimeError("Frozen source changed: " + source)
for row in audits["runs"]:
    run = root / "runs_v2" / row["run"]
    if row["protocol_sha256"] != sha(lock_path) or row["metrics_sha256"] != sha(run / "test_metrics.json"):
        raise RuntimeError("Test audit no longer matches " + row["run"])
    if row["prediction_sha256"] != sha(run / "test_predictions.npz"):
        raise RuntimeError("Test prediction cache changed: " + row["run"])
    if json.loads((run / "profile.json").read_text()) != json.loads((run / "test_metrics.json").read_text())["profile"]:
        raise RuntimeError("Stored profile does not match evaluated profile: " + row["run"])
    for name, expected in lock["runs"][row["run"]].items():
        if sha(run / name) != expected:
            raise RuntimeError("Frozen artifact changed: " + row["run"] + "/" + name)

paths = [lock_path, root / "runs_v2/screening_selection.json", audit_path, summary_path,
         root / "data/split_cifar100.json",
         paper_audit_path, root / "results/v2/run_resume_events.json",
         root / "results/v2/v1_full_audit.json", root / "results/v2/v1_paper_table_audit.json",
         root / "results/v2/v1_revision_table_audit.json", root / "results/v2/v1_figure_data_audit.json",
         root / "results/v2/routing_validation.json", root / "results/v2/phase1_validation_report.md",
         root / "figures/v2/routing_validation_exploratory.data.json",
         root / "figures/v2/v1_routing_reliability_valid_seeds.data.json",
         root / "figures/v2/v2_budget_accuracy.data.json", root / "figures/v2/v2_paired_effects.data.json",
         root / "figures/v2/v2_exit_risk.data.json"]
selected = summary["selection"]
paths += [root / "results/v2" / (selected + "_latency_interleaved.json"),
          root / "results/v2/final_feature_analysis.json"]
names = ("config.json", "environment.json", "profile.json", "history.jsonl", "completed.json",
         "validation_predictions.metadata.json", "policies.json", "test_predictions.metadata.json",
         "test_metrics.json")
for name in lock["runs"]:
    paths.extend(root / "runs_v2" / name / item for item in names)
if any(not p.is_file() for p in paths):
    raise RuntimeError("Missing required evidence: " + ", ".join(str(p) for p in paths if not p.is_file()))
staging.mkdir(parents=True)
files = {}
for path in paths:
    relative = path.relative_to(root)
    target = staging / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    original, copied = sha(path), sha(target)
    if original != copied:
        raise RuntimeError("Evidence copy mismatch: " + str(relative))
    files[relative.as_posix()] = copied
readme = ("# V2 实验记录包\n\n"
          "本目录保留冻结协议、验证选型、七次运行的配置与训练日志、策略、测试指标和审核记录。"
          "根目录内路径与项目原路径对应。各文件的 SHA-256 见 manifest.json。\n\n"
          "本包含固定数据划分索引，不含 CIFAR-100 原始图像、模型权重、优化器检查点和逐图预测缓存；"
          "仅凭本包不能重新计算逐图指标或逐字节还原训练。复核单样本结果需要另外取得与冻结散列匹配的文件；"
          "从头训练可依照项目 README 和 v2/README.md 执行，随机增强与设备差异可能改变结果。\n\n"
          "V2 使用此前已查看的同一官方测试集，结果为探索性补充。验证预算与测试实际平均计算比应分别解释。\n")
(staging / "README.md").write_text(readme, encoding="utf-8")
files["README.md"] = sha(staging / "README.md")
manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "audited lightweight evidence; no weights, datasets or per-image predictions",
            "files": files}
(staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
staging.rename(dest)
print(json.dumps({"output": str(dest), "files": len(files), "manifest_sha256": sha(dest / "manifest.json")}))
