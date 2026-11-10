"""Synchronize the narrative revision in the delivered Word back to Markdown."""
from pathlib import Path
from docx import Document
import re

root=Path.cwd()
source=root/"paper/report_revision.md"
target=root/"paper/report_logic_revision.md"
word=root/"output/基于ResNet18的提前退出CNN课程论文-逻辑修订版.docx"
s=source.read_text(encoding="utf-8")
d=Document(word)
paragraphs=[p.text for p in d.paragraphs]
def wordtext(prefix):
    matches=[p for p in paragraphs if p.startswith(prefix)]
    if len(matches)!=1:
        raise RuntimeError(f"Expected one Word paragraph for {prefix}: {len(matches)}")
    return matches[0]
def replace_line(prefix,replacement):
    global s
    pattern=r"^"+re.escape(prefix)+r".*$"
    s,n=re.subn(pattern,lambda m:replacement,s,count=1,flags=re.MULTILINE)
    if n!=1:
        raise RuntimeError("Missing source prefix "+prefix)
for old,new in [
    ("摘要：针对固定深度","摘要：针对经典卷积"),
    ("图像分类网络通常为","ResNet-18 是利用"),
    ("残差网络利用恒等映射","基线采用固定计算路径"),
    ("本文围绕残差 CNN 的中间出口","本文在 ResNet-18 的第二"),
    ("以下分类和计算量结果来自","本节首先给出完整单出口"),
]:
    replace_line(old,wordtext(new))
for a,b in [
    ("## 2 局部—全局协同提前退出网络","## 2 经典网络基线与结构改进"),
    ("### 2.1 残差骨干与多出口结构","### 2.1 ResNet-18 基线与固定深度问题"),
    ("### 2.2 局部—全局门控出口","### 2.2 面向中间分类的局部—全局出口改进"),
    ("## 3 实验设置","## 3 实验设计与评价方法"),
    ("## 4 实验结果与分析","## 4 基线与改进模型的实验结果"),
    ("## 4.1 按预算分配计算","## 4.1 基线性能与预算下的改进收益"),
    ("## 4.2 出口特征与蒸馏","## 4.2 改进出口与蒸馏的作用"),
]:
    if s.count(a)!=1: raise RuntimeError("Unexpected heading "+a)
    s=s.replace(a,b,1)
fig1=re.search(r"^!\[图 1 .*?\]\([^\n]+\)\n",s,re.M)
if not fig1: raise RuntimeError("Figure 1 missing")
original=fig1.group(0)
s=s[:fig1.start()]+s[fig1.end():]
newcaption=wordtext("图 1 基于 ResNet-18")
newfigure="!["+newcaption+"](../figures/mechanism.png)\n"
idx=s.index("### 2.2 面向中间分类的局部—全局出口改进")
s=s[:idx]+wordtext("单出口基线只在第四阶段")+"\n\n"+newfigure+"\n"+s[idx:]
for number,heading,intro in [(3,"## 4.1 基线性能与预算下的改进收益",wordtext("单出口 ResNet-18 在官方测试集上")),
                             (4,"## 4.2 改进出口与蒸馏的作用","")]:
    match=re.search(r"^表 "+str(number)+r" [^\n]+\n\n(?:\|[^\n]+\n)+",s,re.M)
    if not match: raise RuntimeError(f"Table {number} missing")
    table=match.group(0)
    s=s[:match.start()]+s[match.end():]
    insertion=heading+"\n\n"+(intro+"\n\n" if intro else "")+table
    if s.count(heading)!=1: raise RuntimeError("Heading mismatch")
    s=s.replace(heading,insertion,1)
for number,prefix in [(2,"图 2 不同出口结构"),(3,"图 3 固定出口")]:
    match=re.search(r"^!\[图 "+str(number)+r" [^\]]+\]\(([^\n]+)\)$",s,re.M)
    if not match: raise RuntimeError("Missing figure "+str(number))
    s=s[:match.start()]+"!["+wordtext(prefix)+"]("+match.group(1)+")"+s[match.end():]
if target.exists():
    raise RuntimeError("Refusing to overwrite synchronized source")
target.write_text(s,encoding="utf-8")
print(target)
print("tables",len(re.findall(r"^表 \d+ ",s,re.M)),"figures",len(re.findall(r"^!\[图 \d+ ",s,re.M)))
