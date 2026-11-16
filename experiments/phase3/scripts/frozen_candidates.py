"""Deterministic candidate construction from frozen dev choices, without qrels."""
from common import ROOT,read,jsonl,sha
from sparse import run_from_matrix
import numpy as np
BASE=ROOT.parents[1];PORTFOLIO=BASE.parent

def stage_config(pool):
    if pool=='score_fusion_dev':
        item=read(ROOT/'artifacts/result_score_fusion_dev.json')['metadata']['selected']
        return dict(kind='legacy_fusion',mode=item['mode'],weights=item['weights'])
    item=read(ROOT/'artifacts'/('result_'+pool+'.json'))['metadata']
    return dict(kind='giga_fusion',mode='zscore',weights=item['weights'],prompt=item['prompt'],length=1024)

def scale(x,mode):
    if mode=='zscore':return (x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-12)
    if mode=='minmax':return (x-x.min(1,keepdims=True))/np.maximum(x.max(1,keepdims=True)-x.min(1,keepdims=True),1e-12)
    raise ValueError(mode)

def candidate_pool(stage,ids,split):
    index=read(BASE/'cache/index.json');positions=[index['query_ids'].index(q) for q in ids]
    assert len(index['doc_ids'])==3633
    parts=[scale(np.load(BASE/'cache'/('scores_'+name+'.npy'))[positions].astype(np.float64),stage['mode']) for name in ['bm25','e5_large','qwen_embedding']]
    b,e,q=stage['weights'][:3];matrix=b*parts[0]+e*parts[1]+q*parts[2]
    if stage['kind']=='giga_fusion':
        tag='giga_nf4_l1024'+('_test' if split=='test' else '')
        embedded=np.load(ROOT/'cache'/('embeddings_'+tag+'.npz'))
        assert embedded['doc_ids'].tolist()==index['doc_ids'] and embedded['query_ids'].tolist()==ids
        raw=np.load(ROOT/'cache'/('scores_'+tag+'_'+stage['prompt']+'.npy')).astype(np.float64)
        assert raw.shape==matrix.shape
        matrix+=stage['weights'][3]*scale(raw,stage['mode'])
    return run_from_matrix(matrix,ids,index['doc_ids'])

def test_inputs():
    rows=[r for r in jsonl(BASE/'data/processed/queries.jsonl') if r['split']=='test']
    ids=[r['_id'] for r in rows];assert len(ids)==323
    assert not set(ids)&set(read(ROOT/'artifacts/protocol.json')['full_dev_query_ids'])
    return ids,{r['_id']:r['text'] for r in rows},{r['_id']:r['title']+'\n'+r['text'] for r in jsonl(ROOT/'data/corpus.jsonl')}

def assert_sources(lock):
    for name,digest in lock['source_sha256'].items():assert sha(ROOT/'scripts'/name)==digest,name
    assert sha(BASE/'cache/index.json')==lock['index_sha256']
    assert sha(BASE/'data/processed/queries.jsonl')==lock['queries_sha256']
    assert sha(ROOT/'data/corpus.jsonl')==lock['corpus_sha256']
    for name,digest in lock['base_score_sha256'].items():assert sha(BASE/'cache'/name)==digest,name
    for name,digest in lock['giga_input_sha256'].items():assert sha(ROOT/'cache'/name)==digest,name
    assert sha(ROOT/'artifacts/selection_phase3.json')==lock['selection_sha256']
