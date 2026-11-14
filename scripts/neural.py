"""Frozen neural competitors and retrieval/reranker adaptations; no qrels are read."""
import argparse,time,gc,math
from common import ROOT,jsonl,read,dump,model_path,seed,sha,development_allowed
from sparse import run_from_matrix
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer,AutoModelForSequenceClassification,AutoModelForCausalLM

TASK='Given a biomedical search query, retrieve relevant scientific passages that answer the query'
PREFIX='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

def encode(alias,texts,queries=False,batch_size=8):
    seed(7);qwen=alias=='qwen_embedding';tok=AutoTokenizer.from_pretrained(model_path(alias),padding_side='left' if qwen else 'right',local_files_only=True)
    model=AutoModel.from_pretrained(model_path(alias),dtype=torch.bfloat16 if qwen else torch.float32,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    prepared=[('Instruct: '+TASK+'\nQuery: '+s if queries else s) if qwen else ('query: ' if queries else 'passage: ')+s for s in texts]
    out=[];truncated=0;started=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(texts),batch_size):
            batch=prepared[start:start+batch_size]
            raw=tok(batch,truncation=False)['input_ids'];truncated+=sum(len(x)>512 for x in raw)
            enc=tok(batch,padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
            h=model(**enc).last_hidden_state.float()
            if qwen:pooled=h[:,-1,:]  # all batches left padded
            else:pooled=(h*enc.attention_mask.unsqueeze(-1)).sum(1)/enc.attention_mask.sum(1,keepdim=True)
            out.append(torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy())
            if start%400==0:print(alias,'queries' if queries else 'corpus',start,'/',len(texts),flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
    return np.concatenate(out),dict(seconds=time.perf_counter()-started,truncated=truncated,rows=len(texts),max_tokens=512)

def rrf(runs,k=60,weights=None):
    if weights is None:weights=[1.]*len(runs)
    out={}
    assert all(set(run)==set(runs[0]) for run in runs)
    for q in runs[0]:
        values={}
        for run,w in zip(runs,weights):
            for i,d in enumerate(sorted(run[q],key=lambda d:(-run[q][d],d))):values[d]=values.get(d,0)+w/(k+i+1)
        ranked=sorted(values,key=lambda d:(-values[d],d))[:100]
        out[q]={d:float(len(ranked)-j) for j,d in enumerate(ranked)}
    return out

def rerank(alias,length,corpus,queries,runs):
    tok=AutoTokenizer.from_pretrained(model_path(alias),padding_side='left' if alias=='qwen_reranker' else 'right',local_files_only=True)
    kind=AutoModelForCausalLM if alias=='qwen_reranker' else AutoModelForSequenceClassification
    model=kind.from_pretrained(model_path(alias),dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    cache=ROOT/'cache'/f'pair_scores_{alias}_{length}.json';scores=read(cache) if cache.exists() else {}
    pending=list(dict.fromkeys((q,d) for run in runs.values() for q,v in run.items() for d in sorted(v,key=lambda x:(-v[x],x))[:20] if q+'\t'+d not in scores))
    prefix=tok.encode(PREFIX,add_special_tokens=False);suffix=tok.encode(SUFFIX,add_special_tokens=False)
    yes=tok.convert_tokens_to_ids('yes');no=tok.convert_tokens_to_ids('no');started=time.perf_counter();truncated=0
    batch_size=2 if length>512 else 8
    with torch.inference_mode():
        for start in range(0,len(pending),batch_size):
            pairs=pending[start:start+batch_size]
            if alias=='bge_reranker':
                raw=[(queries[q][:8192],corpus[d][:8192]) for q,d in pairs]
                enc=tok(raw,padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda');p=model(**enc).logits.flatten().float().cpu().tolist()
            else:
                texts=[f'<Instruct>: {TASK}\n<Query>: {queries[q]}\n<Document>: {corpus[d]}' for q,d in pairs]
                tokenized=tok(texts,padding=False,truncation=False,add_special_tokens=False)['input_ids'];budget=length-len(prefix)-len(suffix)
                truncated+=sum(len(x)>budget for x in tokenized)
                enc=tok.pad({'input_ids':[prefix+x[:budget]+suffix for x in tokenized]},padding=True,return_tensors='pt').to('cuda')
                logits=model(**enc,logits_to_keep=1).logits[:,-1,[no,yes]].float();p=logits.softmax(-1)[:,1].cpu().tolist()
            for (q,d),v in zip(pairs,p):scores[q+'\t'+d]=float(v)
            if start%400==0:dump(cache,scores);print(alias,length,start,'/',len(pending),flush=True)
    dump(cache,scores);outputs={}
    for name,run in runs.items():
        out={}
        for q,values in run.items():
            ranked=sorted(values,key=lambda d:(-values[d],d));head=sorted(ranked[:20],key=lambda d:(-scores[q+'\t'+d],d));ordered=head+ranked[20:]
            out[q]={d:float(len(ordered)-i) for i,d in enumerate(ordered)}
        outputs[name+'_'+alias+str(length)]=out
    del model;gc.collect();torch.cuda.empty_cache()
    return outputs,dict(seconds=time.perf_counter()-started,new_pairs=len(pending),cached_pairs=len(scores),truncated_new_pairs=truncated,max_length=length,batch_size=batch_size)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['dense','rerank','all'],default='all');a=ap.parse_args();development_allowed();seed(7)
    corpus=jsonl(ROOT/'data/processed/corpus.jsonl');qs=[r for r in jsonl(ROOT/'data/processed/queries.jsonl') if r['split']!='train'];ix=read(ROOT/'cache/index.json')
    assert ix['query_ids']==[r['_id'] for r in qs] and ix['doc_ids']==[r['_id'] for r in corpus]
    runtime=read(ROOT/'artifacts/neural_runtime.json') if (ROOT/'artifacts/neural_runtime.json').exists() else {}
    if a.stage in ['all','dense']:
        for alias in ['e5_small','e5_large','qwen_embedding']:
            out=ROOT/'cache'/('run_'+alias+'.json')
            if out.exists():continue
            docs,dt=encode(alias,[r['title']+'\n'+r['text'] for r in corpus]);queries,qt=encode(alias,[r['text'] for r in qs],True)
            np.savez_compressed(ROOT/'cache'/('embeddings_'+alias+'.npz'),corpus=docs,queries=queries)
            matrix=queries@docs.T;np.save(ROOT/'cache'/('scores_'+alias+'.npy'),matrix);dump(out,run_from_matrix(matrix,ix['query_ids'],ix['doc_ids']))
            runtime[alias]=dict(corpus=dt,queries=qt);dump(ROOT/'artifacts/neural_runtime.json',runtime)
        dump(ROOT/'cache/run_rrf.json',rrf([read(ROOT/'cache/run_bm25.json'),read(ROOT/'cache/run_qwen_embedding.json')]))
    if a.stage in ['all','rerank']:
        pool={n:read(ROOT/'cache'/('run_'+n+'.json')) for n in ['bm25','e5_large','rrf']}
        docs={r['_id']:r['title']+'\n'+r['text'] for r in corpus};queries={r['_id']:r['text'] for r in qs}
        for alias,length,pools in [('bge_reranker',512,pool),('qwen_reranker',512,{'rrf':pool['rrf']}),('qwen_reranker',2048,{'rrf':pool['rrf']})]:
            key=alias+str(length)
            if key in runtime:continue
            outputs,timing=rerank(alias,length,docs,queries,pools)
            for name,run in outputs.items():dump(ROOT/'cache'/('run_'+name+'.json'),run)
            runtime[key]=timing;dump(ROOT/'artifacts/neural_runtime.json',runtime)
    print('Neural stage complete',a.stage,flush=True)
if __name__=='__main__':main()
