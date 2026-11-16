"""Apply the already defined ranking-control family to all three Giga screens."""
from itertools import product
from study import *
from ranking_math import rerank

assert not (ROOT/'artifacts/evaluation_lock.json').exists()
ids,_,_,qrels=inputs(True);scores=read(ROOT/'cache/pair_scores_qwen4_2048.json')
trials=[]
for name in ['giga_pool1_dev','giga_pool2_dev','giga_pool3_dev']:
    pool=read(ROOT/'cache'/('run_'+name+'.json'))
    configs=[dict(pool=name,depth=0,tie='doc_id',mode='retrieval',weight=0.)]
    configs += [dict(pool=name,depth=20,tie=t,mode=m,weight=w) for t,m,w in product(['doc_id','retrieval'],['rank_rrf','zlogit'],[0.,.05,.1,.25,.5,1.])]
    for config in configs:
        run={q:rerank(pool[q],scores,q,config) for q in ids};metrics,_=evaluate(qrels,run)
        trials.append(dict(config=config,metrics=metrics))
best=max(trials,key=lambda t:t['metrics']['ndcg_cut_10'])
dump(ROOT/'artifacts/giga_screen_controls_trials.json',dict(split='fixed64dev',trials=trials,best=best,code_sha256=sha(__file__),ranking_source_sha256=sha(ROOT/'scripts/ranking_math.py')))
print('GIGA SCREEN CONTROLS',best,flush=True)
