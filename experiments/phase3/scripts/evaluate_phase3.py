"""One relevance evaluation of the fixed run, with paired query uncertainty."""
from study import *
from common import write_jsonl
from frozen_candidates import assert_sources

assert not (ROOT/'artifacts/final_results.json').exists(),'Final relevance evaluation already recorded'
lock_path=ROOT/'artifacts/evaluation_lock.json';lock=read(lock_path);assert_sources(lock)
manifest=read(ROOT/'artifacts/test_inference.json');run_path=ROOT/'cache/run_phase3_test.json'
assert sha(lock_path)==manifest['lock_sha256'] and sha(run_path)==manifest['run_sha256']
assert sha(ROOT/'cache/frozen_phase3_test_candidates.json')==manifest['candidate_sha256']
qrels=read(BASE/'data/processed/test_qrels.json');assert len(qrels)==323
run=read(run_path);assert set(run)==set(qrels)
metrics,per=evaluate(qrels,run);pool_metrics,pool_per=evaluate(qrels,read(ROOT/'cache/frozen_phase3_test_candidates.json'))
prior=read(BASE/'experiments/phase2/artifacts/final_results.json');old=prior['per_query']['phase2_selected'];assert set(per)==set(old)
ids=list(qrels);rng=np.random.default_rng(10);sample=rng.integers(0,len(ids),size=(2000,len(ids)))
delta=np.array([per[q]['ndcg_cut_10']-old[q]['ndcg_cut_10'] for q in ids]);boot=dict(seed=10,repeats=2000,delta=float(delta.mean()),ci95=np.quantile(delta[sample].mean(1),[.025,.975]).tolist())
value=metrics['ndcg_cut_10'];baseline=prior['metrics']['phase2_selected']['ndcg_cut_10'];assert abs(baseline-lock['baseline_test_ndcg10'])<1e-12
rows=[]
for name,ranking in [('phase3_selected',run),('phase3_retrieval',read(ROOT/'cache/frozen_phase3_test_candidates.json'))]:
    for q in ids:
        for rank,d in enumerate(sorted(ranking[q],key=lambda d:(-ranking[q][d],d)),1):rows.append(dict(method=name,query_id=q,doc_id=d,rank=rank,relevance=qrels[q].get(d,0)))
write_jsonl(ROOT/'artifacts/final_predictions.jsonl',rows)
dump(ROOT/'artifacts/test_relevance_summary.json',{q:dict(relevant_documents=sum(g>0 for g in rel.values()),ideal_grades=sorted(rel.values(),reverse=True)[:10]) for q,rel in qrels.items()})
record=dict(metrics=metrics,per_query=per,retrieval_metrics=pool_metrics,retrieval_per_query=pool_per,baseline_phase2=prior['metrics']['phase2_selected'],baseline_phase2_per_query=old,paired_bootstrap=boot,
 relative_gain_vs_own=value/baseline-1,absolute_gain_vs_own=value-baseline,target_5pct_met=value>=lock['target_test_ndcg10'],target_test_ndcg10=lock['target_test_ndcg10'],
 published_reference=dict(model='Giga-Embeddings-instruct',ndcg10=.36834,source='https://huggingface.co/spaces/sergeyzh/rusBEIR-lite/blob/ef8a174bbc1c1d344066bc98fd22813196ae0ef6/results.jsonl',comparison='External aggregate; not paired, not independently reproduced'),relative_gain_vs_published=value/.36834-1,
 lock_sha256=sha(lock_path),test_qrels_sha256=sha(BASE/'data/processed/test_qrels.json'),test_reuse_disclosed=True)
dump(ROOT/'artifacts/final_results.json',record)
print('PHASE3 FINAL TEST',value,'relative gain over own',record['relative_gain_vs_own'],'target met',record['target_5pct_met'],flush=True)
print('PAIRED BOOTSTRAP',boot,flush=True)
