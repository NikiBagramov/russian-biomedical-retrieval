"""CPU baselines using the author's current BM25s tokenization and fixed Lucene settings."""
import re,time
from common import ROOT,jsonl,dump,sha,development_allowed
import numpy as np
import bm25s

def score_field(documents,queries):
    model=bm25s.BM25(method='lucene',k1=1.2,b=.75)
    tokens=[re.findall(r'\w+',s.lower()) for s in documents]
    model.index(tokens,show_progress=False)
    inds,values=model.retrieve([re.findall(r'\w+',s.lower()) for s in queries],k=len(documents),show_progress=False)
    result=np.zeros((len(queries),len(documents)),dtype=np.float32)
    for i in range(len(queries)):result[i,inds[i]]=values[i]
    return result

def run_from_matrix(matrix,query_ids,doc_ids,k=100):
    # Stable lexical doc-id tie break, mirrored in score perturbations only at ties.
    order_ids=np.array(sorted(range(len(doc_ids)),key=lambda j:doc_ids[j]))
    out={}
    for i,q in enumerate(query_ids):
        order=order_ids[np.argsort(-matrix[i,order_ids],kind='stable')]
        order=[j for j in order if doc_ids[j]!=q][:k]
        # Ranking scores, rather than tiny floating epsilons, ensure TREC ties cannot reverse order.
        out[q]={doc_ids[j]:float(k-r) for r,j in enumerate(order)}
    return out

def main():
    development_allowed();started=time.perf_counter()
    corpus=jsonl(ROOT/'data/processed/corpus.jsonl');queries=[r for r in jsonl(ROOT/'data/processed/queries.jsonl') if r['split']!='train']
    query_ids=[r['_id'] for r in queries];doc_ids=[r['_id'] for r in corpus]
    texts=[r['processed_title']+' '+r['processed_text'] for r in corpus];qs=[r['processed_text'] for r in queries]
    variants={};times={}
    for name,docs,qq in [('bm25',texts,qs),('bm25_raw',[r['title']+' '+r['text'] for r in corpus],[r['text'] for r in queries])]:
        s=time.perf_counter();variants[name]=score_field(docs,qq);times[name]=time.perf_counter()-s
    s=time.perf_counter();title=score_field([r['processed_title'] for r in corpus],qs);body=score_field([r['processed_text'] for r in corpus],qs)
    variants['bm25_twofield']=np.maximum(title,body)+.5*np.minimum(title,body);times['bm25_twofield']=time.perf_counter()-s
    for name,matrix in variants.items():
        np.save(ROOT/'cache'/('scores_'+name+'.npy'),matrix)
        dump(ROOT/'cache'/('run_'+name+'.json'),run_from_matrix(matrix,query_ids,doc_ids))
    dump(ROOT/'cache/index.json',dict(query_ids=query_ids,doc_ids=doc_ids))
    dump(ROOT/'artifacts/sparse_runtime.json',dict(methods=times,total_seconds=time.perf_counter()-started,settings=dict(method='lucene',k1=1.2,b=.75,top_k=100),code_sha256=sha(__file__)))
    print('CPU sparse rankings complete',times,flush=True)
if __name__=='__main__':main()
