"""The exact fixed final grid executed before the recorded phase-two test."""
from study import *
from expand_queries import rrf

ids,queries,docs,rels=inputs();qwen=read(ROOT/'cache/run_qwen4_fusion_d20_l2048_full.json');old=read(BASE/'cache/run_selected.json');pool=read(ROOT/'cache/run_score_fusion_dev.json');old={q:old[q] for q in ids};trials=[];best=None
assert not (ROOT/'artifacts/evaluation_lock.json').exists(),'Use a fresh reproduction directory'
for w in [0.,.25,.5,.75,1.]:
    for v in [0.,.1,.25]:
        run=rrf([qwen,old,pool],[1.,w,v]);score=evaluate(rels,run)[0]['ndcg_cut_10'];item=dict(qwen4_weight=1.,phase1_weight=w,retrieval_weight=v,dev_ndcg10=score);trials.append(item)
        if best is None or score>best[0]:best=(score,item,run)
dump(ROOT/'artifacts/selection_trials_phase2.json',dict(grid_predeclared_before_test=[dict(phase1_weight=[0.,.25,.5,.75,1.]),dict(retrieval_weight=[0.,.1,.25])],trials=trials,selected=best[1]));dump(ROOT/'cache/run_phase2_selected_dev.json',best[2]);print('DEV FINAL CANDIDATE',best[1],flush=True)
