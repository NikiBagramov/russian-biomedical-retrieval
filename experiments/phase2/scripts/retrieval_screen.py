"""Cheap candidate-pool and fusion controls; never loads test relevance."""
from study import *
from itertools import product
from sparse import run_from_matrix

def main():
    ids,queries,docs,rels=inputs();index=read(BASE/'cache/index.json');ix=[index['query_ids'].index(q) for q in ids]
    aliases=['bm25','e5_large','qwen_embedding']
    matrices={n:np.load(BASE/'cache'/('scores_'+n+'.npy'))[ix].astype(np.float64) for n in aliases}
    baseline=read(BASE/'cache/run_selected.json')
    trial=[];best=None
    # A small fixed grid: original versus three retrievers, two score scalings.
    for mode in ['minmax','zscore']:
        scaled={}
        for n,x in matrices.items():
            if mode=='minmax':scaled[n]=(x-x.min(1,keepdims=True))/np.maximum(x.max(1,keepdims=True)-x.min(1,keepdims=True),1e-12)
            else:scaled[n]=(x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-12)
        for b,e,q in product([.25,.5,1.],[0.,.5,1.],[.5,1.]):
            m=b*scaled['bm25']+e*scaled['e5_large']+q*scaled['qwen_embedding']
            run=run_from_matrix(m,ids,index['doc_ids']);score=evaluate(rels,run)[0]['ndcg_cut_10'];item=dict(mode=mode,weights=[b,e,q],ndcg10=score);trial.append(item)
            if best is None or score>best[0]:best=(score,item,run)
    measure('score_fusion_dev',best[2],rels,dict(selected=best[1],fixed_grid_trials=trial))
    # Upper bound diagnoses whether deeper reranking can improve the candidate set.
    bounds={}
    for n,run in [('phase1_rrf',read(BASE/'cache/run_rrf.json')),('score_fusion',best[2])]:
        for depth in [20,50,100]:
            oracle={q:ranking({d:rels[q].get(d,0) for d in sorted(run[q],key=lambda d:(-run[q][d],d))[:depth]}) for q in ids}
            bounds[n+str(depth)]=evaluate(rels,oracle)[0]['ndcg_cut_10']
    dump(ROOT/'artifacts/candidate_oracle_dev.json',dict(note='Uses dev labels for diagnosis only; oracle rankings never used as model predictions',bounds=bounds))
    print('Development candidate-set oracle bounds',bounds,flush=True)
if __name__=='__main__':main()
