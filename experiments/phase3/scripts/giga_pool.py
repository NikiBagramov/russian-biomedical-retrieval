"""Fixed full-dev first-stage fusion grid; no test labels or generation."""
from study import *
from sparse import run_from_matrix
from itertools import product
import hashlib,json

assert not (ROOT/'artifacts/evaluation_lock.json').exists()
spec=dict(prompts=['generic','biomedical'],bm25_weights=[0.,.25,.5],e5_weights=[0.,.25,.5],qwen_weights=[0.,.25,.5],giga_weight=1.,normalization='per-query z-score over all3633docs',policy='Full324 official dev; screen rerank top3 distinct pools')
dump(ROOT/'artifacts/giga_pool_grid.json',spec)
ids,queries,docs,rels=inputs();index=read(BASE/'cache/index.json');positions=[index['query_ids'].index(q) for q in ids]
arrays={n:np.load(BASE/'cache'/f'scores_{n}.npy')[positions].astype(np.float64) for n in ['bm25','e5_large','qwen_embedding']}
scale=lambda x:(x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-12)
scaled={n:scale(v) for n,v in arrays.items()};embedding=np.load(ROOT/'cache/embeddings_giga_nf4_l1024.npz')
assert embedding['query_ids'].tolist()==ids and embedding['doc_ids'].tolist()==index['doc_ids']
trials=[];top=[]
for prompt in spec['prompts']:
    giga=np.load(ROOT/'cache'/f'scores_giga_nf4_l1024_{prompt}.npy').astype(np.float64);giga=scale(giga)
    for b,e,q in product(spec['bm25_weights'],spec['e5_weights'],spec['qwen_weights']):
        matrix=giga+b*scaled['bm25']+e*scaled['e5_large']+q*scaled['qwen_embedding']
        run=run_from_matrix(matrix,ids,index['doc_ids']);value=evaluate(rels,run)[0]['ndcg_cut_10']
        signature=hashlib.sha256(json.dumps({q:sorted(run[q],key=lambda d:(-run[q][d],d)) for q in ids},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        item=dict(prompt=prompt,weights=[b,e,q,1.],ndcg10=value,ranking_sha256=signature);trials.append(item)
        if all(existing[1]['ranking_sha256']!=signature for existing in top):top.append((value,item,run))
        top.sort(key=lambda x:-x[0]);top=top[:3]
dump(ROOT/'artifacts/giga_pool_trials.json',dict(grid=spec,trials=trials,code_sha256=sha(__file__)))
for i,(value,item,run) in enumerate(top,1):measure('giga_pool'+str(i)+'_dev',run,rels,item)
print('GIGA TOP3',[(v,c) for v,c,r in top],flush=True)
