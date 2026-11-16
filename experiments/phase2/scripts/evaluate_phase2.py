"""One final relevance evaluation with paired uncertainty and the published anchor."""
from study import *
from common import write_jsonl

assert not (ROOT/'artifacts/final_results.json').exists(),'Final evaluation already recorded'
lock=read(ROOT/'artifacts/evaluation_lock.json');manifest=read(ROOT/'artifacts/test_inference.json');path=ROOT/'cache/run_phase2_test.json'
assert sha(path)==manifest['run_sha256'] and sha(ROOT/'artifacts/evaluation_lock.json')==manifest['lock_sha256']
qrels=read(BASE/'data/processed/test_qrels.json');assert len(qrels)==323
pool=read(ROOT/'cache/frozen_candidate_pool.json')
runs={'phase2_selected':read(path),'phase2_retrieval':{q:pool[q] for q in qrels}}
for name,filename in [('phase1_selected','selected'),('bm25','bm25'),('bm25_bge512','bm25_bge_reranker512'),('phase1_qwen06','rrf_qwen_reranker2048')]:runs[name]=read(BASE/'cache'/('run_'+filename+'.json'))
metrics={};per={};predictions=[]
for name,run in runs.items():
    metrics[name],per[name]=evaluate(qrels,run)
    for q in qrels:
        for rank,d in enumerate(sorted(run[q],key=lambda d:(-run[q][d],d)),1):predictions.append(dict(method=name,query_id=q,doc_id=d,rank=rank,relevance=qrels[q].get(d,0)))
ids=list(qrels);rng=np.random.default_rng(10);sample=rng.integers(0,len(ids),size=(2000,len(ids)));ours=np.array([per['phase2_selected'][q]['ndcg_cut_10'] for q in ids]);comparisons={}
for name in ['phase1_selected','bm25_bge512','phase1_qwen06']:
    delta=ours-np.array([per[name][q]['ndcg_cut_10'] for q in ids]);values=delta[sample].mean(1);comparisons[name]=dict(delta=float(delta.mean()),ci95=np.quantile(values,[.025,.975]).tolist())
anchor=.3483;value=metrics['phase2_selected']['ndcg_cut_10']
record=dict(metrics=metrics,per_query=per,bootstrap=dict(seed=10,repeats=2000,comparisons=comparisons),published_reference=dict(source='https://arxiv.org/html/2504.12879v1',method='BM25+BGE',ndcg10=anchor,comparison='published aggregate; configuration/resources differ, not paired statistical evidence'),relative_gain_vs_published=value/anchor-1,absolute_gain_vs_published=value-anchor,target_5pct_met=value>=anchor*1.05,target_15pct_met=value>=anchor*1.15,lock_sha256=sha(ROOT/'artifacts/evaluation_lock.json'),test_qrels_sha256=sha(BASE/'data/processed/test_qrels.json'))
write_jsonl(ROOT/'artifacts/final_predictions.jsonl',predictions);dump(ROOT/'artifacts/final_results.json',record)
print('PHASE2 FINAL TEST nDCG10',value,'relative gain versus published',record['relative_gain_vs_published'],flush=True)
print('PAIRED COMPARISONS',comparisons,flush=True)
