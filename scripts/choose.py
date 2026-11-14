"""Development-only comparison and fusion; freeze before opening test relevance."""
import itertools
from common import ROOT,read,dump,sha,development_allowed
from neural import rrf
from metrics import evaluate

def main():
    development_allowed();qrels=read(ROOT/'data/processed/development_qrels.json')['dev'];runs={p.stem[4:]:read(p) for p in sorted((ROOT/'cache').glob('run_*.json')) if p.stem!='run_selected'}
    required=['bm25','bm25_raw','bm25_twofield','e5_small','e5_large','qwen_embedding','rrf','bm25_bge_reranker512','e5_large_bge_reranker512','rrf_bge_reranker512','rrf_qwen_reranker512','rrf_qwen_reranker2048']
    assert set(required)<=set(runs),set(required)-set(runs)
    table={name:evaluate(qrels,run)[0] for name,run in runs.items()};trials=[];best=None
    for length,w,original in itertools.product([512,2048],[0.,.25,.5,.75,1.],[0.,.1,.25]):
        run=rrf([runs['rrf_qwen_reranker'+str(length)],runs['rrf_bge_reranker512'],runs['rrf']],weights=[w,1-w,original]);score=evaluate(qrels,run)[0]['ndcg_cut_10']
        item=dict(length=length,qwen_weight=w,bge_weight=1-w,retrieval_weight=original,dev_ndcg10=score);trials.append(item)
        if best is None or score>best[0]:best=(score,item,run)
    dump(ROOT/'cache/run_selected.json',best[2]);table['selected_fusion']=evaluate(qrels,best[2])[0]
    record=dict(primary='ndcg_cut_10',selected=best[1],selection='Only the official dev qrels; test qrels not loaded',methods=table,trials=trials,source_sha256={'scripts/'+n:sha(ROOT/'scripts'/n) for n in ['common.py','prepare.py','sparse.py','neural.py','metrics.py','choose.py']},run_hashes={p.name:sha(p) for p in sorted((ROOT/'cache').glob('run_*.json'))},development_qrels_sha256=sha(ROOT/'data/processed/development_qrels.json'))
    dump(ROOT/'artifacts/selection.json',record);dump(ROOT/'artifacts/evaluation_lock.json',record)
    print('Selection locked',best[1],flush=True)
if __name__=='__main__':main()
