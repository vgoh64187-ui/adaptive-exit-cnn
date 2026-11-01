"""Typeset a Chinese paper from editable Markdown and verified equation assets.

Use the bundled document Python runtime. This renderer does not invent results;
final numerical text must be assembled and audited against experiment outputs.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether, PageBreak
import reportlab.lib.textsplit as textsplit
import reportlab.platypus.paragraph as paragraph_engine

# ReportLab's default Japanese set omits several full-width Chinese punctuation
# marks. Extend only this process's line-breaking rules, not installed files.
for module in [textsplit,paragraph_engine]:
    module.ALL_CANNOT_START+='，；：！？％）”’》〉'

WIDTH=160*mm


def build(source,math_dir,output,draft=False):
    source=Path(source);raw=source.read_bytes();text=raw.decode('utf-8')
    math=json.loads((Path(math_dir)/'manifest.json').read_text(encoding='utf-8'))
    if math['source_sha256']!=hashlib.sha256(raw).hexdigest():raise RuntimeError('Equation manifest source changed')
    for name,filename in [('CN','simsun.ttc'),('Hei','simhei.ttf'),('Latin','times.ttf'),('LatinItalic','timesi.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/filename)))
    styles=dict(body=ParagraphStyle('body',fontName='CN',fontSize=11,leading=17,firstLineIndent=22,
                                   alignment=TA_JUSTIFY,wordWrap='CJK',spaceAfter=5,autoLeading='max',allowWidows=0,allowOrphans=0),
                title=ParagraphStyle('title',fontName='Hei',fontSize=18,leading=25,alignment=TA_CENTER,spaceAfter=15),
                h1=ParagraphStyle('h1',fontName='Hei',fontSize=13,leading=19,spaceBefore=11,spaceAfter=6,keepWithNext=True),
                h2=ParagraphStyle('h2',fontName='Hei',fontSize=11.5,leading=17,spaceBefore=8,spaceAfter=5,keepWithNext=True),
                caption=ParagraphStyle('caption',fontName='CN',fontSize=10,leading=14,alignment=TA_CENTER,spaceAfter=9),
                table_caption=ParagraphStyle('table_caption',fontName='CN',fontSize=10,leading=14,alignment=TA_CENTER,
                                             spaceAfter=7,keepWithNext=True),
                note=ParagraphStyle('note',fontName='CN',fontSize=9,leading=13,wordWrap='CJK',spaceAfter=7),
                cell=ParagraphStyle('cell',fontName='CN',fontSize=9.5,leading=13,alignment=TA_CENTER,wordWrap='CJK'))

    def equation(expression,display):
        expression=' '.join(expression.split())
        key=hashlib.sha256((str(display)+expression).encode()).hexdigest()[:20]
        return math['expressions'][key]

    def inline(value):
        def plain(token):
            parts=re.split(r'([A-Za-z0-9][A-Za-z0-9 \t.,:;!?()/+%=–-]*)',token)
            return ''.join(('<font name="Latin">'+html.escape(part)+'</font>')
                           if re.match(r'^[A-Za-z0-9]',part) else html.escape(part) for part in parts)
        tokens=re.split(r'(\\\(.*?\\\)|\*\*.*?\*\*)',value)
        result=[]
        for token in tokens:
            if token.startswith('\\(') and token.endswith('\\)'):
                asset=equation(token[2:-2],False)
                result.append(f'<img src="{html.escape(asset["path"],quote=True)}" width="{asset["width_pt"]}" height="{asset["height_pt"]}" valign="-{asset["depth_pt"]}"/>')
            elif token.startswith('**') and token.endswith('**'):
                result.append('<font name="Hei">'+html.escape(token[2:-2])+'</font>')
            else:result.append(plain(token))
        return ''.join(result)

    lines=text.splitlines();story=[];i=0;eq_number=0
    if draft:story.append(Paragraph('排版检查稿',styles['note']))
    while i<len(lines):
        line=lines[i].strip();i+=1
        if not line:continue
        if line=='\\[':
            expressions=[]
            while i<len(lines) and lines[i].strip()!='\\]':expressions.append(lines[i]);i+=1
            if i==len(lines):raise ValueError('Unclosed display equation')
            i+=1;asset=equation('\n'.join(expressions),True);eq_number+=1
            scale=min(1.,(WIDTH-35)/asset['width_pt'])
            picture=Image(asset['path'],width=asset['width_pt']*scale,height=asset['height_pt']*scale)
            table=Table([[picture,Paragraph(f'({eq_number})',styles['cell'])]],colWidths=[WIDTH-30,30])
            table.setStyle(TableStyle([('ALIGN',(0,0),(0,0),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                                      ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
            story.append(table);continue
        if line.startswith('# '):story.append(Paragraph(inline(line[2:]),styles['title']));continue
        if line.startswith('## '):
            if line=='## 参考文献':story.append(PageBreak())
            story.append(Paragraph(inline(line[3:]),styles['h1']));continue
        if line.startswith('### '):story.append(Paragraph(inline(line[4:]),styles['h2']));continue
        match=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
        if match:
            path=(source.parent/match.group(2)).resolve()
            image=Image(str(path));scale=min(WIDTH/image.imageWidth,145*mm/image.imageHeight)
            image.drawWidth=image.imageWidth*scale;image.drawHeight=image.imageHeight*scale
            story.append(KeepTogether([image,Spacer(1,5),Paragraph(inline(match.group(1)),styles['caption'])]));continue
        if line.startswith('|'):
            rows=[line]
            while i<len(lines) and lines[i].strip().startswith('|'):rows.append(lines[i].strip());i+=1
            cells=[[c.strip() for c in r.strip('|').split('|')] for r in rows if not re.fullmatch(r'[\s|:\-]+',r)]
            n=len(cells[0]);table=Table([[Paragraph(inline(c),styles['cell']) for c in row] for row in cells],
                                       colWidths=[WIDTH/n]*n,repeatRows=1)
            table.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),1,colors.black),
                 ('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),1,colors.black),
                 ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
            story.extend([table,Spacer(1,9)]);continue
        if line.startswith('> '):
            if draft:story.append(Paragraph(inline(line[2:]),styles['note']))
            continue
        paragraph=[line]
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','!','>','\\[')):
            paragraph.append(lines[i].strip());i+=1
        value=''.join(paragraph)
        style=styles['note'] if re.match(r'^\[\d+\]',value) else styles['body']
        if re.match(r'^表\s*\d+',value):style=styles['table_caption']
        story.append(Paragraph(inline(value),style))

    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('Latin',9);canvas.drawCentredString(A4[0]/2,14*mm,str(doc.page));canvas.restoreState()
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=25*mm,leftMargin=25*mm,
                          topMargin=22*mm,bottomMargin=22*mm,title=lines[0].lstrip('# '),author='')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(json.dumps(dict(path=str(output),source_sha256=hashlib.sha256(raw).hexdigest(),equations=eq_number,draft=draft)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--math-dir',required=True)
    p.add_argument('--output',required=True);p.add_argument('--draft',action='store_true');a=p.parse_args()
    build(a.source,a.math_dir,a.output,a.draft)
