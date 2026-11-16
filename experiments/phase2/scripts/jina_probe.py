"""Pinned 2026 listwise architecture with explicitly bounded context blocks."""
import argparse,time,gc
from study import *
import torch
from transformers import AutoModel

def score(model,query,texts,doc_tokens=512,block_tokens=4096,block_docs=16):
    query,texts,lengths,_=model._truncate_texts(query,texts,256,doc_tokens)
    doc_embeds=[];query_embeds=[];weights=[];block=[];used=0;blocks=0
    def flush():
        nonlocal block,used,blocks
        if not block:return
        out=model._compute_single_batch(query,block,instruction=None)
        doc_embeds.extend(out.doc_embeds[0].float().cpu().numpy())
        query_embeds.append(out.query_embeds[0].float().cpu().numpy())
        weights.append(float(((1+out.scores.float())/2).max().cpu()))
        del out
        block=[];used=0;blocks+=1
    with torch.inference_mode():
        for text,length in zip(texts,lengths):
            if block and (used+length+32>block_tokens or len(block)>=block_docs):flush()
            block.append(text);used+=length+32
        flush()
    query_embedding=np.average(np.array(query_embeds),axis=0,weights=weights)
    values=model._calculate_cosine_scores(query_embedding,np.array(doc_embeds))[0]
    assert len(values)==len(texts) and np.isfinite(values).all()
    return values,blocks

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');ap.add_argument('--depth',type=int,default=20);ap.add_argument('--pool',choices=['rrf','fusion'],default='rrf');ap.add_argument('--smoke',action='store_true');a=ap.parse_args();seed(7)
    ids,queries,docs,rels=inputs(not a.full)
    if a.smoke:ids=ids[:2];rels={q:rels[q] for q in ids}
    pool=read(BASE/'cache/run_rrf.json') if a.pool=='rrf' else read(ROOT/'cache/run_score_fusion_dev.json')
    name=f'jina35_{a.pool}_d{a.depth}_'+('full' if a.full else 'screen')+('_smoke' if a.smoke else '')
    checkpoint=ROOT/'cache'/('scores_'+name+'.json');scores=read(checkpoint) if checkpoint.exists() else {}
    path=PORTFOLIO/'cache/models/jina_reranker35';assert (ROOT/'artifacts/model_jina35.json').exists(),'Pinned download is not complete'
    model=AutoModel.from_pretrained(path,local_files_only=True,trust_remote_code=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    # The repository's source was inspected; _ensure_tokenizer loads this local directory.
    model._ensure_tokenizer();model.config.use_cache=False
    started=time.perf_counter();blocks=0
    for i,q in enumerate(ids):
        if q not in scores:
            head=sorted(pool[q],key=lambda d:(-pool[q][d],d))[:a.depth]
            values,b=score(model,queries[q],[docs[d] for d in head]);blocks+=b
            scores[q]={d:float(v) for d,v in zip(head,values)};dump(checkpoint,scores)
        if i%8==0:print(name,i,'/',len(ids),'seconds',round(time.perf_counter()-started,1),flush=True)
    run={}
    for q in ids:
        head=sorted(scores[q],key=lambda d:(-scores[q][d],d));tail=[d for d in sorted(pool[q],key=lambda d:(-pool[q][d],d)) if d not in scores[q]]
        run[q]={d:float(len(head+tail)-i) for i,d in enumerate(head+tail)}
    measure(name,run,rels,dict(model='jinaai/jina-reranker-v3.5',revision=read(ROOT/'artifacts/model_jina35.json')['revision'],depth=a.depth,doc_token_cap=512,query_token_cap=256,block_doc_cap=16,block_token_budget=4096,block_query_fusion='official max((1+score)/2)-weighted query-embedding average',adaptation='bounded context blocks instead of 131K full-list pass for 6GB GPU',seconds=time.perf_counter()-started,blocks_new=blocks,peak_gpu_mb=torch.cuda.max_memory_allocated()/1048576,code_sha256=sha(__file__)))
    del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
