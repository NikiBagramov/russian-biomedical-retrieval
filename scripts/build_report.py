"""Course-template report structure; figures and tables come from measured artifacts."""
import json,html,re
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,Image,KeepTogether,PageBreak
from PIL import Image as PILImage
ROOT=Path(__file__).resolve().parents[1]

def main():
    config=json.loads((ROOT/'project.json').read_text(encoding='utf-8'));paper=json.loads((ROOT/'report/paper.json').read_text(encoding='utf-8'))
    fonts=Path('C:/Windows/Fonts')
    if fonts.exists():names=['times.ttf','timesbd.ttf','timesi.ttf']
    else:fonts=Path('/usr/share/fonts/truetype/dejavu');names=['DejaVuSerif.ttf','DejaVuSerif-Bold.ttf','DejaVuSerif-Italic.ttf']
    for n,f in zip(['Body','BodyBold','BodyItalic'],names):pdfmetrics.registerFont(TTFont(n,str(fonts/f)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBold')
    styles=getSampleStyleSheet()
    for name,kw in {'Paper':dict(fontName='Body',fontSize=10.5,leading=14.3,alignment=TA_JUSTIFY,spaceAfter=8),'TitlePaper':dict(fontName='BodyBold',fontSize=18,leading=22,alignment=TA_CENTER,spaceAfter=14),'Authors':dict(fontName='Body',fontSize=11,leading=15,alignment=TA_CENTER,spaceAfter=12),'SectionPaper':dict(fontName='BodyBold',fontSize=13.5,leading=18,spaceBefore=12,spaceAfter=8,keepWithNext=True),'CellPaper':dict(fontName='Body',fontSize=8.4,leading=10.8),'CaptionPaper':dict(fontName='BodyItalic',fontSize=9,leading=12,spaceAfter=10)}.items():styles.add(ParagraphStyle(name=name,**kw))
    story=[];md=[]
    def safe(text):return html.escape(str(text).replace('\u2011','-').replace('\u2013','-').replace('\u2014','-'))
    def p(text,style='Paper'):
        escaped=safe(text)
        escaped=re.sub(r'https://[^\s<>]+',lambda m:'<link href="'+m.group().rstrip('.,;)')+'">'+m.group().rstrip('.,;)')+'</link>'+m.group()[len(m.group().rstrip('.,;)')):],escaped)
        story.append(Paragraph(escaped,styles[style]));md.append(str(text)+'\n')
    p(paper['title'],'TitlePaper');p('Никита Баграмов и Альфия Исхакова','Authors');p(config['course']+' | '+paper['date'],'Authors')
    for block in paper['blocks']:
        kind=block['type']
        if kind=='p':p(block['text'])
        elif kind=='h':p(block['text'],'SectionPaper')
        elif kind=='page':story.append(PageBreak())
        elif kind=='table':
            content=[block['headers']]+block['rows'];cells=[[Paragraph(safe(v),styles['CellPaper']) for v in row] for row in content];fractions=block.get('widths',[1]*len(content[0]));widths=[509*v/sum(fractions) for v in fractions]
            table=Table(cells,colWidths=widths,repeatRows=1,hAlign='CENTER');table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf0f2')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#52646c')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d5dde0')),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.extend([table,Spacer(1,9)]);md.append('| '+' | '.join(map(str,block['headers']))+' |\n| '+' | '.join(['---']*len(block['headers']))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in block['rows'])+'\n')
        elif kind=='figure':
            path=ROOT/'artifacts/figures'/block['file'];im=PILImage.open(path);width=495;height=min(430,width*im.height/im.width);width=height*im.width/im.height
            story.append(KeepTogether([Image(str(path),width=width,height=height),Paragraph(safe(block['caption']),styles['CaptionPaper'])]));md.append('!['+block['caption']+'](../artifacts/figures/'+block['file']+')\n')
        else:raise ValueError(kind)
    out=ROOT/'output/pdf';out.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.setFont('Body',8);canvas.setFillColor(colors.HexColor('#53636d'));canvas.drawString(43,26,'Bagramov & Iskhakova | '+config['slug']);canvas.drawRightString(A4[0]-43,26,str(doc.page))
    doc=SimpleDocTemplate(str(out/(config['slug']+'_report.pdf')),pagesize=A4,rightMargin=43,leftMargin=43,topMargin=38,bottomMargin=47,title=paper['title'],author=', '.join(config['authors']))
    doc.build(story,onFirstPage=footer,onLaterPages=footer);(ROOT/'report/report.md').write_text('\n'.join(md),encoding='utf-8',newline='\n');print('Report PDF created')
if __name__=='__main__':main()
