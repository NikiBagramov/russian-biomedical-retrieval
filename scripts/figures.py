"""Scientific figures built only from actual exported numerical results."""
from common import ROOT,read,jsonl
from labels import LABELS
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    slug=read(ROOT/'project.json')['slug'];r=read(ROOT/'artifacts/results.json');out=ROOT/'artifacts/figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.constrained_layout.use':True})
    def save(fig,name):fig.savefig(out/name,dpi=180,facecolor='white');plt.close(fig)
    def bar(values,title,xlabel,name,highlight):
        ordered=sorted(values,key=values.get);fig,ax=plt.subplots(figsize=(8.5,max(3.,len(ordered)*.3)));y=np.arange(len(ordered));ax.barh(y,[values[n] for n in ordered],color=['#167b77' if n==highlight else '#8997a5' for n in ordered]);ax.set_yticks(y,[LABELS.get(n,n) for n in ordered]);ax.set_title(title);ax.set_xlabel(xlabel);ax.set_xlim(0,max(values.values())*1.1)
        for i,n in enumerate(ordered):ax.text(values[n]+.005,i,f'{values[n]:.3f}',va='center',fontsize=8)
        save(fig,name)
    if slug=='russian-paraphrase-semantics':
        metrics=r['methods'];bar({n:v['paraphraser']['roc_auc'] for n,v in metrics.items()},'ParaPhraser: strict semantic equivalence','Official test ROC AUC','main.png','selected_blend')
        names=['tfidf','ruroberta512','frozen_logreg','lora_ensemble','selected_blend'];datasets=['paraphraser','rupaws_wiki','rupaws_qqp'];fig,axes=plt.subplots(1,3,figsize=(10,3.9))
        for ax,d in zip(axes,datasets):
            a=[metrics[n][d]['roc_auc'] for n in names];b=[metrics[n][d+'_sentence_disjoint']['roc_auc'] for n in names];x=np.arange(len(names));ax.bar(x-.18,a,.36,label='Official');ax.bar(x+.18,b,.36,label='Sentence-disjoint');ax.set_title(d);ax.set_ylim(.4,1);ax.set_xticks(x,[LABELS[n] for n in names],rotation=70,ha='right')
        axes[0].set_ylabel('ROC AUC');axes[-1].legend(fontsize=7);save(fig,'robustness.png')
        fig,ax=plt.subplots(figsize=(8.3,3.7));comparisons=r['bootstrap']['comparisons']
        for i,(n,v) in enumerate(comparisons.items()):
            lo,hi=v['ci95'];ax.plot([lo,hi],[i,i],color='#167b77',linewidth=2);ax.scatter([v['delta']],[i],color='#167b77')
        ax.axvline(0,color='black',linestyle='--',linewidth=.8);ax.set_yticks(range(len(comparisons)),[LABELS[n] for n in comparisons]);ax.set_xlabel('Selected blend minus comparator: ROC AUC');ax.set_title('Paired group bootstrap: 95% intervals');save(fig,'uncertainty.png')
    elif slug=='russian-conservative-spellcheck':
        metrics=r['metrics'];bar({n:v['MultidomainGold']['SpellRuEval']['f1'] for n,v in metrics.items()},'MultidomainGold: paper-comparable spelling correction','Official test SpellRuEval F1','main.png','selected')
        fig,ax=plt.subplots(figsize=(8.4,4.7));names=['sage95','sage_large','ordinary_seed7','ordinary_matched_seed7','identity_seed7','medoid','selected']
        for n in names:
            v=metrics[n]['MultidomainGold']['SpellRuEval'];ax.scatter(v['recall'],v['precision'],s=40);ax.annotate(LABELS[n],(v['recall'],v['precision']),fontsize=7,xytext=(3,3),textcoords='offset points')
        ax.set_xlabel('Spelling recall');ax.set_ylabel('Spelling precision');ax.set_title('Precision / recall trade-off');save(fig,'robustness.png')
        controls=read(ROOT/'artifacts/clean_controls.json')['methods'];fig,ax=plt.subplots(figsize=(8.5,4.3));names=list(controls);x=np.arange(len(names))
        for j,c in enumerate(['MultidomainGold','RUSpellRU']):ax.bar(x+(j-.5)*.36,[controls[n][c]['exact_preservation'] for n in names],.36,label=c)
        ax.set_xticks(x,[LABELS.get(n,n) for n in names],rotation=45,ha='right');ax.set_ylim(0,1.05);ax.set_ylabel('Fraction of clean references left unchanged');ax.set_title('Separate clean-input stress test, after selection lock');ax.legend();save(fig,'uncertainty.png')
    else:
        metrics=r['metrics'];bar({n:v['ndcg_cut_10'] for n,v in metrics.items()},'RusBEIR rus-NFCorpus: full official test','nDCG@10 (TREC linear gain)','main.png','selected')
        fig,ax=plt.subplots(figsize=(8.5,3.6));comp=r['bootstrap']['comparisons']
        for i,(n,v) in enumerate(comp.items()):ax.plot(v['ci95'],[i,i],color='#167b77',linewidth=2);ax.scatter([v['delta']],[i],color='#167b77')
        ax.axvline(0,color='black',linestyle='--',linewidth=.8);ax.set_yticks(range(len(comp)),[LABELS[n] for n in comp]);ax.set_xlabel('Selected method minus comparator: nDCG@10');ax.set_title('Paired query bootstrap: 95% intervals');save(fig,'uncertainty.png')
        per=read(ROOT/'artifacts/per_query_metrics.json');delta=[per['selected'][q]['ndcg_cut_10']-per['bm25_bge_reranker512'][q]['ndcg_cut_10'] for q in per['selected']];fig,ax=plt.subplots(figsize=(8.2,3.7));ax.hist(delta,bins=30,color='#167b77');ax.axvline(0,color='black',linestyle='--',linewidth=.8);ax.set_xlabel('Query nDCG@10 difference: selected - BM25+BGE');ax.set_ylabel('Queries');ax.set_title('Gains and regressions across all 323 queries');save(fig,'robustness.png')
    print('Three figures created from measured results')
if __name__=='__main__':main()
