"""Pinned Giga embeddings; inference never reads relevance labels."""
import argparse,time,gc
from common import ROOT,read,dump,jsonl,sha,seed
import numpy as np
import torch
from transformers import AutoConfig,AutoModel,AutoTokenizer,BitsAndBytesConfig

BASE=ROOT.parents[1];PORTFOLIO=BASE.parent
TASKS={
 'generic':'Given a web search query, retrieve relevant passages that answer the query',
 'biomedical':'Given a biomedical search query, retrieve relevant scientific passages that answer the query',
}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--test',action='store_true');ap.add_argument('--length',type=int,default=1024);ap.add_argument('--batch',type=int,default=2);a=ap.parse_args();seed(7)
    assert (ROOT/'artifacts/evaluation_lock.json').exists()==a.test
    assert not (a.test and a.smoke)
    path=PORTFOLIO/'cache/models/giga_embedding';provenance=read(ROOT/'artifacts/model_giga.json');review=read(ROOT/'artifacts/giga_source_review_inputs.json')
    assert provenance['revision']==review['revision']
    assert all(sha(path/name)==digest for name,digest in review['sha256'].items())
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=True,padding_side='right')
    config=AutoConfig.from_pretrained(path,local_files_only=True,trust_remote_code=True)
    model=AutoModel.from_pretrained(path,config=config,local_files_only=True,trust_remote_code=True,dtype=torch.bfloat16,
        attn_implementation='eager',quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),device_map={'':0}).eval()
    # The released outer wrapper does not advertise SDPA; its text encoder does.
    assert model.model._supports_sdpa
    model.model.config._attn_implementation='sdpa';model.model.config.use_cache=False
    ids=read(BASE/'cache/index.json')['doc_ids'];source={r['_id']:r['title']+'\n'+r['text'] for r in jsonl(ROOT/'data/corpus.jsonl')}
    query_ids=read(ROOT/'artifacts/protocol.json')['full_dev_query_ids'];queries={r['_id']:r['text'] for r in jsonl(ROOT/'data/queries.jsonl')}
    if a.test:
        assert a.length==1024
        queries={r['_id']:r['text'] for r in jsonl(BASE/'data/processed/queries.jsonl') if r['split']=='test'}
        query_ids=[q for q in read(BASE/'cache/index.json')['query_ids'] if q in queries];assert len(query_ids)==323
    if a.smoke:ids=ids[:2];query_ids=query_ids[:2]
    vectors={};timings={};started=time.perf_counter()
    groups={'corpus':[source[d] for d in ids]}
    if a.test:
        previous=np.load(ROOT/'cache/embeddings_giga_nf4_l1024.npz');assert previous['doc_ids'].tolist()==ids
        vectors['corpus']=previous['corpus'];groups={}
    groups.update({name:[f'Instruct: {task}\nQuery: {queries[q]}' for q in query_ids] for name,task in TASKS.items()})
    for group,texts in groups.items():
        output=[];tick=time.perf_counter();truncated=0
        with torch.inference_mode():
            for start in range(0,len(texts),a.batch):
                batch=texts[start:start+a.batch];raw=tokenizer(batch,truncation=False)['input_ids'];truncated+=sum(len(t)>a.length for t in raw)
                encoded=tokenizer(batch,padding=True,truncation=True,max_length=a.length,return_tensors='pt').to('cuda')
                embeds=model(**encoded,return_embeddings=True,use_cache=False).float()
                embeds=torch.nn.functional.normalize(embeds,p=2,dim=-1)
                assert torch.isfinite(embeds).all() and embeds.shape[1]==2048
                output.append(embeds.cpu().numpy())
                if start%200==0:print('GIGA',group,start,'/',len(texts),'seconds',round(time.perf_counter()-tick,1),flush=True)
        vectors[group]=np.concatenate(output);timings[group]=dict(rows=len(texts),truncated=truncated,seconds=time.perf_counter()-tick)
    tag=f'giga_nf4_l{a.length}'+('_smoke' if a.smoke else '')+('_test' if a.test else '')
    np.savez_compressed(ROOT/'cache'/('embeddings_'+tag+'.npz'),doc_ids=np.array(ids),query_ids=np.array(query_ids),**vectors)
    for name in TASKS:
        scores=vectors[name]@vectors['corpus'].T
        np.save(ROOT/'cache'/f'scores_{tag}_{name}.npy',scores)
    dump(ROOT/'artifacts'/f'encoding_{tag}.json',dict(model=provenance,review=review,quantization='NF4 double quant; BF16 compute; released latent-attention pooling followed by FP32 L2',max_length=a.length,batch=a.batch,text_encoder_attention='sdpa',prompts=TASKS,groups=timings,peak_gpu_mb=torch.cuda.max_memory_allocated()/1048576,seconds=time.perf_counter()-started,code_sha256=sha(__file__)))
    print('GIGA ENCODING COMPLETE',tag,flush=True);del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
