from common import ROOT,read,dump
from labels import LABELS
from datetime import datetime

class Paper:
    def __init__(self,title):self.title=title;self.blocks=[]
    def h(self,text):self.blocks.append(dict(type='h',text=text))
    def p(self,text):self.blocks.append(dict(type='p',text=text))
    def table(self,headers,rows,widths=None):self.blocks.append(dict(type='table',headers=headers,rows=rows,widths=widths or [1]*len(headers)))
    def figure(self,file,caption):self.blocks.append(dict(type='figure',file=file,caption=caption))
    def references(self,refs):
        self.h('References')
        for i,reference in enumerate(refs,1):self.p(f'[{i}] {reference}')
    def finish(self):
        c=read(ROOT/'project.json')
        self.p('Воспроизводимость. Протокол: PROTOCOL.md; provenance, dev-lock, предсказания: artifacts/. verify_results.py пересчитывает метрики; run_pipeline.py --reproduce --download создаёт новый эксперимент. Генераторы отчёта находятся в scripts/.')
        dump(ROOT/'report/paper.json',dict(title=self.title,date=datetime.now().strftime('%d.%m.%Y'),blocks=self.blocks))

def fmt(v,n=4):return '-' if v is None else f'{v:.{n}f}'
def label(n):return LABELS.get(n,n)
def interval(values,n=4):return '['+', '.join(fmt(x,n) for x in values)+']'
