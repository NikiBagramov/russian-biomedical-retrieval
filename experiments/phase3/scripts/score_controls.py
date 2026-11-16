"""Fixed dev-only score/tie controls using completed cached screen pairs."""
from study import *
from itertools import product

assert not (ROOT/'artifacts/evaluation_lock.json').exists()
grid=dict(depths=[20,50,100],tie_breakers=['doc_id','retrieval'],score_modes=['rank_rrf','zlogit'],retrieval_weights=[0.,.05,.1,.25,.5,1.],policy='Complete fixed64 screen only; final promotion requires all324dev')
dump(ROOT/'artifacts/score_controls_grid.json',grid)
ids,queries,docs,rels=inputs(True);pool=read(ROOT/'cache/run_score_fusion_dev.json');pairs=read(ROOT/'cache/pair_scores_qwen4_2048.json')
trials=[];best=None
for depth,tie,mode,weight in product(grid['depths'],grid['tie_breakers'],grid['score_modes'],grid['retrieval_weights']):
    run={}
    for q in ids:
        ordered=sorted(pool[q],key=lambda d:(-pool[q][d],d));head=ordered[:depth];prior={d:i for i,d in enumerate(ordered)}
        assert all(q+'\t'+d in pairs for d in head)
        values={d:pairs[q+'\t'+d] for d in head}
        ranked=sorted(head,key=lambda d:(-values[d],prior[d] if tie=='retrieval' else d))
        if mode=='rank_rrf':combined={d:1/(60+i+1)+weight/(60+prior[d]+1) for i,d in enumerate(ranked)}
        else:
            x=np.array([np.log(np.clip(values[d],1e-7,1-1e-7)/(1-np.clip(values[d],1e-7,1-1e-7))) for d in ranked]);x=(x-x.mean())/max(x.std(),1e-12)
            y=np.array([-prior[d] for d in ranked],dtype=float);y=(y-y.mean())/max(y.std(),1e-12)
            combined={d:float(a+weight*b) for d,a,b in zip(ranked,x,y)}
        head=sorted(head,key=lambda d:(-combined[d],prior[d] if tie=='retrieval' else d));run[q]={d:float(len(ordered)-i) for i,d in enumerate(head+ordered[depth:])}
    score=evaluate(rels,run)[0]['ndcg_cut_10'];item=dict(depth=depth,tie=tie,mode=mode,weight=weight,ndcg10=score);trials.append(item)
    if best is None or score>best[0]:best=(score,item,run)
dump(ROOT/'artifacts/score_controls_trials.json',dict(grid=grid,trials=trials,best=best[1],source_sha256=sha(__file__)))
measure('score_controls_best_screen',best[2],rels,best[1]);print('BEST',best[1],flush=True)
