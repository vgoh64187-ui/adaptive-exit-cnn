import sys
from pathlib import Path
import re,json,hashlib
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from PIL import Image
from lxml import etree
from latex2mathml.converter import convert
root=Path.cwd()
if len(sys.argv) != 3:
 raise SystemExit('Usage: python v2/build_word_omml.py SOURCE OUTPUT')
source=Path(sys.argv[1])
lines=source.read_text(encoding='utf-8').splitlines()
xsl_candidates=[
 Path('C:/Program Files/Microsoft Office/root/Office16/MML2OMML.XSL'),
 Path('C:/Program Files (x86)/Microsoft Office/root/Office16/MML2OMML.XSL')]
xsl_path=next((path for path in xsl_candidates if path.exists()),None)
if not xsl_path:
 raise RuntimeError('Microsoft Office MathML to OMML stylesheet is required')
transform=etree.XSLT(etree.parse(str(xsl_path)))
def append_equation(paragraph,expression):
 normalized=' '.join(expression.split()).replace(chr(92)+'arg'+chr(92)+'max',chr(92)+'operatorname{argmax}')
 mathml=etree.fromstring(convert(normalized).encode('utf-8'))
 for node in mathml.xpath('//*[text()="𝒮"]'):
  node.text='S'
  node.set('mathvariant','script')
 text=''.join(mathml.itertext())
 if chr(92) in text:
  raise RuntimeError('Unconverted LaTeX command in '+normalized)
 result=transform(mathml)
 equation=result.getroot()
 ns={"m":"http://schemas.openxmlformats.org/officeDocument/2006/math"}
 for expr in equation.xpath('.//m:nary/m:e',namespaces=ns):
  if len(expr)==0:
   run=etree.SubElement(expr,'{'+ns["m"]+'}r')
   value=etree.SubElement(run,'{'+ns["m"]+'}t')
   value.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
   value.text=' '
 paragraph._p.append(equation)
d=Document(); sec=d.sections[0];sec.page_width=Cm(21);sec.page_height=Cm(29.7);sec.left_margin=sec.right_margin=Cm(2.5);sec.top_margin=sec.bottom_margin=Cm(2.2)
for name in ['Normal','Title','Heading 1','Heading 2']:
 s=d.styles[name];s.font.name='宋体';s._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体');s.font.color.rgb=RGBColor(0,0,0)
title_pr=d.styles['Title']._element.get_or_add_pPr()
for border in title_pr.findall(qn('w:pBdr')): title_pr.remove(border)
d.styles['Normal'].font.size=Pt(10.5);d.styles['Normal'].paragraph_format.line_spacing=1.35
for name,size in [('Title',16),('Heading 1',13),('Heading 2',11.5)]:d.styles[name].font.size=Pt(size)
def inline(p,s):
 for part in re.split(r'(\\\(.*?\\\)|\*\*.*?\*\*)',s):
  if part.startswith('\\(') and part.endswith('\\)'):
   append_equation(p,part[2:-2])
  elif part.startswith('**') and part.endswith('**'):p.add_run(part[2:-2]).bold=True
  else:p.add_run(part.replace('`',''))
i=0
while i<len(lines):
 line=lines[i].strip();i+=1
 if not line:continue
 if line=='\\[':
  expr=[]
  while i<len(lines) and lines[i].strip()!='\\]':expr.append(lines[i]);i+=1
  i+=1;p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_together=True;append_equation(p,' '.join(expr));continue
 if line.startswith('# '):p=d.add_paragraph(style='Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;inline(p,line[2:]);continue
 if line.startswith('## '):
  p=d.add_paragraph(style='Heading 1')
  inline(p,line[3:]);continue
 if line.startswith('### '):p=d.add_paragraph(style='Heading 2');inline(p,line[4:]);continue
 m=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
 if m:
  path=(source.parent/m.group(2)).resolve()
  with Image.open(path) as im:w,h=im.size
  width=min(15.7, 13.8*w/h);p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True;p.add_run().add_picture(str(path),width=Cm(width))
  p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(7);p.add_run(m.group(1)).font.size=Pt(9);continue
 if line.startswith('|'):
  rows=[line]
  while i<len(lines) and lines[i].strip().startswith('|'):rows.append(lines[i].strip());i+=1
  cells=[[c.strip() for c in row.strip('|').split('|')] for row in rows if not re.fullmatch(r'[\s|:\-]+',row)]
  t=d.add_table(rows=0,cols=len(cells[0]));t.style='Table Grid';t.autofit=True
  for j,row in enumerate(cells):
   rr=t.add_row()
   rr._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"));
   if j==0:
    trPr=rr._tr.get_or_add_trPr();trPr.append(OxmlElement('w:tblHeader'))
   for cell,text in zip(rr.cells,row):
    p=cell.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(1);p.paragraph_format.line_spacing=1.1;run=p.add_run(text);run.font.size=Pt(9);run.bold=(j==0)
    if len(cells)<=9 and j<len(cells)-1:p.paragraph_format.keep_with_next=True
  d.add_paragraph().paragraph_format.space_after=Pt(2);continue
 if line.startswith('> '):continue
 p=d.add_paragraph();p.paragraph_format.space_after=Pt(5)
 next_line=next((item.strip() for item in lines[i:] if item.strip()),'')
 if re.match(r'^表 \d+ ',line) and next_line.startswith('|'):
  p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
 elif not line.startswith('[') and not line.startswith('关键词'):p.paragraph_format.first_line_indent=Cm(.7)
 inline(p,line)
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER;fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');footer._p.append(fld)
out=Path(sys.argv[2]);out.parent.mkdir(parents=True,exist_ok=True);d.save(out);print(out)
