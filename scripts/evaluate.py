"""Evaluate all frozen methods on the complete official test once."""
import csv,math
from common import ROOT,read,dump,write_jsonl,sha
from metrics import evaluate
import numpy as np

def main():
    lock=read(ROOT/'artifacts/evaluation_lock.json')
    for name,digest in lock['run_hashes'].items():assert sha(ROOT/'cache'/name)==digest,'Predictions changed after selection'
    qrels=read(ROOT/'data/processed/test_qrels.json');methods={};per={};predictions=[]
    for filename,digest in lock['run_hashes'].items():
        name=filename[4:-5];run=read(ROOT/'cache'/filename);methods[name],per[name]=evaluate(qrels,run)
        for q in qrels:
            ordered=sorted(run[q],key=lambda d:(-run[q][d],d))
            predictions.extend(dict(method=name,query_id=q,doc_id=d,rank=i+1,score=run[q][d],relevance=qrels[q].get(d,0)) for i,d in enumerate(ordered))
    assert 'selected' in methods
    rng=np.random.default_rng(10);ids=list(qrels);samples=rng.integers(0,len(ids),size=(2000,len(ids)));bootstrap={}
    ours=np.array([per['selected'][q]['ndcg_cut_10'] for q in ids]);references=['bm25','e5_large','bm25_bge_reranker512','rrf_bge_reranker512','rrf_qwen_reranker2048']
    for name in references:
        delta=ours-np.array([per[name][q]['ndcg_cut_10'] for q in ids]);distribution=delta[samples].mean(1)
        bootstrap[name]=dict(delta=float(delta.mean()),ci95=np.quantile(distribution,[.025,.975]).tolist(),probability_delta_nonpositive=float(np.mean(distribution<=0)))
    ownci=np.quantile(ours[samples].mean(1),[.025,.975]).tolist()
    summary=dict(test_queries=len(qrels),metrics=methods,selected_test_ndcg95=ownci,bootstrap=dict(seed=10,repeats=2000,unit='query',comparisons=bootstrap),lock_sha256=sha(ROOT/'artifacts/evaluation_lock.json'),paper_reference=dict(source='https://arxiv.org/html/2504.12879v1',metric='nDCG@10',BM25=.3233,multilingual_E5_large=.3096,BM25_BGE=.3483,note='Published Elasticsearch baseline is not byte-identical to our CPU BM25 adaptation; separate reference values, not fresh runs.'))
    dump(ROOT/'artifacts/results.json',summary);dump(ROOT/'artifacts/per_query_metrics.json',per);write_jsonl(ROOT/'artifacts/predictions.jsonl',predictions)
    dump(ROOT/'artifacts/test_relevance_summary.json',{q:dict(relevant_documents=sum(v>0 for v in rel.values()),ideal_grades=sorted(rel.values(),reverse=True)[:10]) for q,rel in qrels.items()})
    with (ROOT/'artifacts/results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['method']+list(next(iter(methods.values()))));writer.writeheader();writer.writerows(dict(method=k,**v) for k,v in methods.items())
    print('TEST EVALUATION COMPLETE',methods,flush=True)
if __name__=='__main__':main()
