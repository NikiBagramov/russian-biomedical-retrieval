from common import ROOT,read
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    r=read(ROOT/'experiments/phase2/artifacts/final_results.json');m=r['metrics'];out=ROOT/'artifacts/figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.constrained_layout.use':True})
    labels={'bm25':'BM25 local','bm25_bge512':'BM25 + BGE local','phase1_qwen06':'RRF + Qwen 0.6B','phase1_selected':'Previous selected','phase2_retrieval':'New retrieval fusion','phase2_selected':'New system + Qwen 4B'}
    names=sorted(m,key=lambda n:m[n]['ndcg_cut_10']);fig,ax=plt.subplots(figsize=(8.3,3.3));ax.barh(range(len(names)),[m[n]['ndcg_cut_10'] for n in names],color=['#167b77' if n=='phase2_selected' else '#8997a5' for n in names]);ax.set_yticks(range(len(names)),[labels[n] for n in names]);ax.axvline(.3483,color='#a55b2e',linestyle='--',label='Published BM25+BGE');ax.set_xlim(0,.42);ax.set_xlabel('Full official test nDCG@10');ax.legend(loc='lower right',fontsize=8)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,fontsize=8)
    for i,n in enumerate(names):ax.text(m[n]['ndcg_cut_10']+.003,i,f'{m[n]["ndcg_cut_10"]:.4f}',va='center',fontsize=8)
    fig.savefig(out/'phase2_main.png',dpi=180);plt.close(fig)
    comp=r['bootstrap']['comparisons'];fig,ax=plt.subplots(figsize=(8.2,2.5))
    for i,(n,v) in enumerate(comp.items()):ax.plot(v['ci95'],[i,i],color='#167b77',linewidth=2);ax.scatter([v['delta']],[i],color='#167b77')
    ax.axvline(0,color='black',linestyle='--',linewidth=.8);ax.set_yticks(range(len(comp)),[labels[n] for n in comp]);ax.set_xlabel('New system minus comparator: nDCG@10');fig.savefig(out/'phase2_uncertainty.png',dpi=180);plt.close(fig)
    per=r['per_query'];delta=[per['phase2_selected'][q]['ndcg_cut_10']-per['phase1_selected'][q]['ndcg_cut_10'] for q in per['phase2_selected']];fig,ax=plt.subplots(figsize=(8.2,2.6));ax.hist(delta,bins=30,color='#167b77');ax.axvline(0,color='black',linestyle='--',linewidth=.8);ax.set_xlabel('Per-query nDCG@10 difference versus previous selected');ax.set_ylabel('Queries');fig.savefig(out/'phase2_regressions.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
