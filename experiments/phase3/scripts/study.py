"""Continuation utilities: development labels only, immutable phase-one files."""
from common import ROOT,read,dump,jsonl,sha,seed,model_path
import numpy as np
from metrics import evaluate

BASE=ROOT.parents[1]
PORTFOLIO=BASE.parent

def inputs(screen=False):
    protocol=read(ROOT/'artifacts/protocol.json')
    ids=protocol['screen_query_ids'] if screen else protocol['full_dev_query_ids']
    rels=read(ROOT/'data/dev_qrels.json')
    queries={r['_id']:r['text'] for r in jsonl(ROOT/'data/queries.jsonl')}
    docs={r['_id']:r['title']+'\n'+r['text'] for r in jsonl(ROOT/'data/corpus.jsonl')}
    return ids,queries,docs,{q:rels[q] for q in ids}

def ranking(values):
    ordered=sorted(values,key=lambda d:(-values[d],d))
    return {d:float(len(ordered)-i) for i,d in enumerate(ordered)}

def measure(name,run,rels,metadata):
    metrics,per=evaluate(rels,run)
    dump(ROOT/'cache'/('run_'+name+'.json'),run)
    dump(ROOT/'artifacts'/('result_'+name+'.json'),dict(name=name,split='dev_screen' if len(rels)==64 else 'dev_full',queries=len(rels),metrics=metrics,per_query=per,metadata=metadata))
    print(name,len(rels),'nDCG10',round(metrics['ndcg_cut_10'],6),flush=True)
    return metrics
