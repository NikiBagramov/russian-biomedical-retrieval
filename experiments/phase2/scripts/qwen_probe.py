"""Pointwise alternative: larger released Qwen reranker with NF4 for 6GB VRAM."""
import argparse,time,gc
from study import *
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig

TASK='Given a biomedical search query, retrieve relevant scientific passages that answer the query'
PREFIX='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--depth',type=int,default=50);ap.add_argument('--length',type=int,default=2048);ap.add_argument('--size',choices=['4','06'],default='4');ap.add_argument('--pool',choices=['rrf','fusion'],default='fusion');a=ap.parse_args();seed(7)
    ids,queries,docs,rels=inputs(not a.full)
    if a.smoke:ids=ids[:2];rels={q:rels[q] for q in ids}
    pool=read(BASE/'cache/run_rrf.json') if a.pool=='rrf' else read(ROOT/'cache/run_score_fusion_dev.json')
    name=f'qwen{a.size}_{a.pool}_d{a.depth}_l{a.length}_'+('full' if a.full else 'screen')+('_smoke' if a.smoke else '')
    path=PORTFOLIO/'cache/models'/('qwen_reranker4' if a.size=='4' else 'qwen_reranker')
    if a.size=='4':assert (ROOT/'artifacts/model_qwen4.json').exists(),'Verified download not ready'
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='left')
    options=dict(local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa')
    if a.size=='4':options.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),device_map={'':0})
    model=AutoModelForCausalLM.from_pretrained(path,**options)
    if a.size!='4':model=model.to('cuda')
    model.eval();model.config.use_cache=False
    checkpoint=ROOT/'cache'/f'pair_scores_qwen{a.size}_{a.length}.json'
    scores=read(checkpoint) if checkpoint.exists() else (read(BASE/'cache'/f'pair_scores_qwen_reranker_{a.length}.json') if a.size=='06' and (BASE/'cache'/f'pair_scores_qwen_reranker_{a.length}.json').exists() else {})
    prefix=tok.encode(PREFIX,add_special_tokens=False);suffix=tok.encode(SUFFIX,add_special_tokens=False);budget=a.length-len(prefix)-len(suffix);yes=tok.convert_tokens_to_ids('yes');no=tok.convert_tokens_to_ids('no')
    heads={q:sorted(pool[q],key=lambda d:(-pool[q][d],d))[:a.depth] for q in ids}
    pending=[(q,d) for q in ids for d in heads[q] if q+'\t'+d not in scores]
    started=time.perf_counter();truncated=0;batch_size=2
    with torch.inference_mode():
        for start in range(0,len(pending),batch_size):
            pairs=pending[start:start+batch_size]
            texts=[f'<Instruct>: {TASK}\n<Query>: {queries[q]}\n<Document>: {docs[d]}' for q,d in pairs]
            raw=tok(texts,add_special_tokens=False,truncation=False)['input_ids'];truncated+=sum(len(x)>budget for x in raw)
            enc=tok.pad({'input_ids':[prefix+x[:budget]+suffix for x in raw]},padding=True,return_tensors='pt').to('cuda')
            logits=model(**enc,logits_to_keep=1,use_cache=False).logits[:,-1,[no,yes]].float();values=logits.softmax(-1)[:,1].cpu().tolist()
            for (q,d),value in zip(pairs,values):assert np.isfinite(value);scores[q+'\t'+d]=float(value)
            if start%200==0:dump(checkpoint,scores);print(name,start,'/',len(pending),'seconds',round(time.perf_counter()-started,1),flush=True)
    dump(checkpoint,scores);run={}
    for q in ids:
        head=sorted(heads[q],key=lambda d:(-scores[q+'\t'+d],d));tail=[d for d in sorted(pool[q],key=lambda d:(-pool[q][d],d)) if d not in head];ordered=head+tail
        run[q]={d:float(len(ordered)-i) for i,d in enumerate(ordered)}
    measure(name,run,rels,dict(model='Qwen3-Reranker-'+('4B' if a.size=='4' else '0.6B'),quantization='NF4 double quantization, BF16 computation' if a.size=='4' else 'BF16',depth=a.depth,max_length=a.length,new_pairs=len(pending),truncated_new_pairs=truncated,seconds=time.perf_counter()-started,peak_gpu_mb=torch.cuda.max_memory_allocated()/1048576,code_sha256=sha(__file__)))
    del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
