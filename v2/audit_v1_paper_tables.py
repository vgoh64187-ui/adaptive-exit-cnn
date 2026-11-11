"""Check V1 tables carried into the revised paper against their generated sources."""
import hashlib
import json
import re
from pathlib import Path

root = Path.cwd()
report_path = root / "paper/report_logic_revision.md"
report = report_path.read_text(encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(text):
    matches = re.findall(r"(?m)^\|[^\n]+$", text)
    return [line.strip() for line in matches if not re.fullmatch(r"[|:\-\s]+", line)]


def table(number):
    match = re.search(rf"(?m)^表 {number} [^\n]+\n\s*((?:\|[^\n]+\n)+)", report)
    if not match:
        raise RuntimeError(f"Missing V1 table {number}")
    return rows(match.group(1))


sources = {
    1: "paper/generated/training_settings.md",
    2: "paper/generated/architecture_table.md",
    3: "paper/generated/budget_accuracy_table.md",
    4: "paper/generated/exit_accuracy_table.md",
}
for number, name in sources.items():
    if table(number) != rows((root / name).read_text(encoding="utf-8")):
        raise RuntimeError(f"Paper table {number} differs from generated table")

ledger_path = root / "paper/generated/table_provenance.json"
ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
loaded = {}
for name, expected in ledger["sources"].items():
    path = root / name.replace("\\", "/")
    if digest(path) != expected:
        raise RuntimeError(f"Table source changed: {name}")
    loaded[name] = json.loads(path.read_text(encoding="utf-8"))
for name, expected in ledger["output_sha256"].items():
    if digest(root / name.replace("\\", "/")) != expected:
        raise RuntimeError(f"Generated table changed: {name}")
for record in ledger["bindings"]:
    value = loaded[record["source"]]
    for part in record["pointer"].strip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    formatted = f"{value * record['scale']:,.{record['decimal_places']}f}"
    if value != record["raw"] or formatted != record["formatted"]:
        raise RuntimeError(f"Table binding changed: {record['pointer']}")

settings_path = root / "paper/generated/training_settings.provenance.json"
settings = json.loads(settings_path.read_text(encoding="utf-8"))
if digest(root / sources[1]) != settings["output_sha256"]:
    raise RuntimeError("Generated training settings changed")
expected_settings = [f"| {entry['item']} | {entry['value']} |" for entry in settings["entries"]]
if table(1)[1:] != expected_settings:
    raise RuntimeError("Training table differs from setting provenance")
for name, expected in settings["sources"].items():
    if digest(root / name.replace("\\", "/")) != expected:
        raise RuntimeError(f"Training setting source changed: {name}")
result = {
    "scope": "V1 paper tables 1-4 and generated-table provenance; excludes narrative, figures and V2",
    "paper_sha256": digest(report_path),
    "table_provenance_sha256": digest(ledger_path),
    "settings_provenance_sha256": digest(settings_path),
    "tables_verified": list(sources),
    "numeric_bindings_verified": len(ledger["bindings"]),
}
output = root / "results/v2/v1_paper_table_audit.json"
if output.exists():
    raise RuntimeError("Audit output already exists; inspect before replacing")
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
