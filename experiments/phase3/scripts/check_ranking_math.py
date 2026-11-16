"""Validate shared ranking rules against measured controls on real dev data."""
from study import *
from ranking_math import order,rerank

ids,_,_,qrels=inputs(True);pool=read(ROOT/'cache/run_score_fusion_dev.json');pairs=read(ROOT/'cache/pair_scores_qwen4_2048.json')
trials=read(ROOT/'artifacts/score_controls_trials.json')['trials'];maximum=0.
for trial in trials:
    run={q:rerank(pool[q],pairs,q,trial) for q in ids};value=evaluate(qrels,run)[0]['ndcg_cut_10']
    maximum=max(maximum,abs(value-trial['ndcg10']))
assert maximum<1e-12
ids,_,_,_=inputs();old=read(ROOT/'cache/run_qwen4_fusion_d20_l2048_full.json')
config=dict(depth=20,tie='doc_id',mode='rank_rrf',weight=0.)
assert all(order(rerank(pool[q],pairs,q,config))==order(old[q]) for q in ids)
record=dict(passed=True,screen_trials=len(trials),maximum_ndcg_difference=maximum,baseline_dev_rankings=len(ids),source_sha256=sha(ROOT/'scripts/ranking_math.py'),conformance_source_sha256=sha(__file__))
dump(ROOT/'artifacts/ranking_math_conformance.json',record);print('RANKING CONFORMANCE',record,flush=True)
