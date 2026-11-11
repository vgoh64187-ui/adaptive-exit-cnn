"""Generate and render the Word paper after audited V2 analysis completes."""
import hashlib
import subprocess
import sys
import time
from pathlib import Path

root=Path.cwd()
source=root/'paper/report_v2.md'
ready=root/'figures/v2/v2_feature_correction.png'
while not (source.exists() and ready.exists() and (root/'results/v2/final_comparison.json').exists()):
    time.sleep(30)
before=hashlib.sha256(source.read_bytes()).hexdigest()
time.sleep(3)
if hashlib.sha256(source.read_bytes()).hexdigest()!=before:
    raise RuntimeError('Paper source changed while starting build')
output=root/'output/基于ResNet18的提前退出CNN课程论文-V2实验版.docx'
if output.exists():
    raise RuntimeError('Output exists; inspect before rebuilding')
bundle=Path('C:/Users/13246/.cache/codex-runtimes/codex-primary-runtime/dependencies')
docpython=bundle/'python/python.exe'
node=bundle/'node/bin/node.exe'
skill=Path('C:/Users/13246/.codex/plugins/cache/openai-primary-runtime/documents/26.1104.22356/skills/documents')
render=root/'tmp/v2_final_render'
subprocess.run([str(node),str(skill/'container_tools/mark_artifact_operation_started.mjs'),
                '--operation-kind','create','--expected-output-count','1','--output-format','docx'],check=True)
env=__import__('os').environ.copy()
env['PYTHONPATH']=str(root/'tmp/word_math_deps')
subprocess.run([str(docpython),'v2/build_word_omml.py',str(source),str(output)],check=True,env=env)
subprocess.run([str(docpython),'v2/verify_paper.py',str(output)],check=True)
subprocess.run([str(docpython),str(skill/'render_docx.py'),str(output),
                '--output_dir',str(render),'--emit_pdf'],check=True)
print('Word built and rendered; visual page review remains',flush=True)
