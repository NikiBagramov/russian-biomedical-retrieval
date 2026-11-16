"""Fixed phase-two candidate construction and inference; no relevance is loaded."""
import time,gc
from study import ROOT,BASE,PORTFOLIO,read,dump,jsonl,sha,seed,ranking
from sparse import run_from_matrix
from qwen_probe import TASK,PREFIX,SUFFIX
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig

def candidate_pool():
    config=read(ROOT/'artifacts/result_score_fusion_dev.json')['metadata']['selected']
    index=read(BASE/'cache/index.json');scaled={}
    for name in ['bm25','e5_large','qwen_embedding']:
        x=np.load(BASE/'cache'/('scores_'+name+'.npy')).astype(np.float64)
        if config['mode']=='minmax':scaled[name]=(x-x.min(1,keepdims=True))/np.maximum(x.max(1,keepdims=True)-x.min(1,keepdims=True),1e-12)
        else:scaled[name]=(x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-12)
    b,e,q=config['weights'];matrix=b*scaled['bm25']+e*scaled['e5_large']+q*scaled['qwen_embedding']
    run=run_from_matrix(matrix,index['query_ids'],index['doc_ids'])
    development=read(ROOT/'cache/run_score_fusion_dev.json');assert all(run[q]==v for q,v in development.items())
    return run

def score_pairs(ids,queries,docs,pool,checkpoint,depth=20,force=False):
    seed(7);scores=read(checkpoint) if checkpoint.exists() and not force else {};heads={q:sorted(pool[q],key=lambda d:(-pool[q][d],d))[:depth] for q in ids}
    pending=[(q,d) for q in ids for d in heads[q] if q+'\t'+d not in scores]
    started=time.perf_counter()
    if pending:
        path=PORTFOLIO/'cache/models/qwen_reranker4';tok=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='left')
        model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),device_map={'':0}).eval();model.config.use_cache=False
        prefix=tok.encode(PREFIX,add_special_tokens=False);suffix=tok.encode(SUFFIX,add_special_tokens=False);budget=2048-len(prefix)-len(suffix);yes=tok.convert_tokens_to_ids('yes');no=tok.convert_tokens_to_ids('no')
        with torch.inference_mode():
            for start in range(0,len(pending),2):
                pairs=pending[start:start+2];texts=[f'<Instruct>: {TASK}\n<Query>: {queries[q]}\n<Document>: {docs[d]}' for q,d in pairs]
                raw=tok(texts,add_special_tokens=False,truncation=False)['input_ids'];enc=tok.pad({'input_ids':[prefix+x[:budget]+suffix for x in raw]},padding=True,return_tensors='pt').to('cuda')
                values=model(**enc,logits_to_keep=1,use_cache=False).logits[:,-1,[no,yes]].float().softmax(-1)[:,1].cpu().tolist()
                for (q,d),v in zip(pairs,values):assert np.isfinite(v);scores[q+'\t'+d]=float(v)
                if start%200==0:dump(checkpoint,scores);print('FROZEN INFERENCE',start,'/',len(pending),'seconds',round(time.perf_counter()-started,1),flush=True)
        dump(checkpoint,scores);del model;gc.collect();torch.cuda.empty_cache()
    run={}
    for q in ids:
        head=sorted(heads[q],key=lambda d:(-scores[q+'\t'+d],d));tail=[d for d in sorted(pool[q],key=lambda d:(-pool[q][d],d)) if d not in heads[q]];ordered=head+tail
        run[q]={d:float(len(ordered)-i) for i,d in enumerate(ordered)}
    return run,scores,dict(new_pairs=len(pending),seconds_this_resume=time.perf_counter()-started)

if __name__=='__main__':
    assert not (ROOT/'artifacts/evaluation_lock.json').exists()
    ids=read(ROOT/'artifacts/protocol.json')['screen_query_ids'][:2];queries={r['_id']:r['text'] for r in jsonl(ROOT/'data/queries.jsonl')};docs={r['_id']:r['title']+'\n'+r['text'] for r in jsonl(ROOT/'data/corpus.jsonl')};pool=candidate_pool()
    run,scores,timing=score_pairs(ids,queries,docs,pool,ROOT/'cache/parity_probe.json',force=True)
    previous=read(ROOT/'cache/pair_scores_qwen4_2048.json');difference=max(abs(v-previous[k]) for k,v in scores.items())
    assert difference<1e-5,difference
    assert all(run[q]==read(ROOT/'cache/run_qwen4_fusion_d20_l2048_full.json')[q] for q in ids)
    dump(ROOT/'artifacts/inference_parity.json',dict(passed=True,development_queries=ids,pairs=len(scores),max_abs_difference=difference,source_sha256=sha(__file__),timing=timing))
    print('FROZEN INFERENCE PARITY PASSED',difference,flush=True)
