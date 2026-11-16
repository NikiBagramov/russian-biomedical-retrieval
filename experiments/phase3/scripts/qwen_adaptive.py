"""Same Qwen pointwise scoring with bounded, length-grouped inference batches."""
import argparse,time,gc,re
from study import *
from qwen_probe import TASK,PREFIX,SUFFIX
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parity',action='store_true');ap.add_argument('--full',action='store_true');ap.add_argument('--legacy',action='store_true');ap.add_argument('--depth',type=int,default=100);ap.add_argument('--pool',default='score_fusion_dev');a=ap.parse_args();seed(7)
    assert re.fullmatch(r'[a-zA-Z0-9_]+',a.pool)
    assert not (ROOT/'artifacts/evaluation_lock.json').exists()
    ids,queries,docs,rels=inputs(not a.full);pool=read(ROOT/'cache'/('run_'+a.pool+'.json'))
    if a.parity:ids=ids[:2];rels={q:rels[q] for q in ids};a.depth=20
    checkpoint=ROOT/'cache/pair_scores_qwen4_2048.json';scores=read(checkpoint);previous=scores.copy() if a.parity else None
    heads={q:sorted(pool[q],key=lambda d:(-pool[q][d],d))[:a.depth] for q in ids}
    pending=[(q,d) for q in ids for d in heads[q] if a.parity or q+'\t'+d not in scores]
    if not a.parity and not a.legacy:assert read(ROOT/'artifacts/adaptive_parity.json')['passed']
    started=time.perf_counter();truncated=0;new=0;batches={}
    if pending:
        path=PORTFOLIO/'cache/models/qwen_reranker4';tok=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='left')
        prefix=tok.encode(PREFIX,add_special_tokens=False);suffix=tok.encode(SUFFIX,add_special_tokens=False);budget=2048-len(prefix)-len(suffix)
        examples=[]
        for start in range(0,len(pending),128):
            pairs=pending[start:start+128];raw=tok([f'<Instruct>: {TASK}\n<Query>: {queries[q]}\n<Document>: {docs[d]}' for q,d in pairs],add_special_tokens=False,truncation=False)['input_ids']
            for pair,tokens in zip(pairs,raw):truncated+=len(tokens)>budget;examples.append((prefix+tokens[:budget]+suffix,pair))
        if not a.legacy:examples.sort(key=lambda x:(len(x[0]),x[1]))
        model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),device_map={'':0}).eval();model.config.use_cache=False
        no=tok.convert_tokens_to_ids('no');yes=tok.convert_tokens_to_ids('yes');position=0;last_print=-1
        with torch.inference_mode():
            while position<len(examples):
                count=1
                while count<(2 if a.legacy else 8) and position+count<len(examples) and (count+1)*max(len(t[0]) for t in examples[position:position+count+1])<=4096:count+=1
                batch=examples[position:position+count];encoded=tok.pad({'input_ids':[x[0] for x in batch]},padding=True,return_tensors='pt').to('cuda')
                values=model(**encoded,logits_to_keep=1,use_cache=False).logits[:,-1,[no,yes]].float().softmax(-1)[:,1].cpu().tolist()
                for (_,pair),value in zip(batch,values):assert np.isfinite(value);scores['\t'.join(pair)]=float(value)
                position+=count;new+=count;batches[str(count)]=batches.get(str(count),0)+1
                if position//200!=last_print:
                    last_print=position//200
                    if not a.parity:dump(checkpoint,scores)
                    print('ADAPTIVE',position,'/',len(examples),'seconds',round(time.perf_counter()-started,1),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    if a.parity:
        keys=[q+'\t'+d for q in ids for d in heads[q]];difference=max(abs(scores[k]-previous[k]) for k in keys)
        same=all(sorted(heads[q],key=lambda d:(-scores[q+'\t'+d],d))==sorted(heads[q],key=lambda d:(-previous[q+'\t'+d],d)) for q in ids)
        record=dict(passed=difference<=1e-5 and same,pairs=len(keys),max_abs_difference=difference,ranks_equal=same,batches=batches,seconds=time.perf_counter()-started,code_sha256=sha(__file__),scope='Only this40-pair resource parity probe; not proof of cross-hardware bit equality')
        dump(ROOT/'artifacts/adaptive_parity.json',record);print('PARITY',record,flush=True);return
    dump(checkpoint,scores);run={}
    for q in ids:
        head=sorted(heads[q],key=lambda d:(-scores[q+'\t'+d],d));tail=[d for d in sorted(pool[q],key=lambda d:(-pool[q][d],d)) if d not in head];ordered=head+tail;run[q]={d:float(len(ordered)-i) for i,d in enumerate(ordered)}
    name='adaptive_'+a.pool+'_d'+str(a.depth)+'_'+('full' if a.full else 'screen')
    measure(name,run,rels,dict(depth=a.depth,pool=a.pool,max_length=2048,new_pairs=new,batch_profile=batches,max_padded_tokens=4096,max_batch=2 if a.legacy else 8,truncated_new_pairs=truncated,seconds=time.perf_counter()-started,code_sha256=sha(__file__),baseline_source_sha256=sha(ROOT/'scripts/qwen_probe.py')))
if __name__=='__main__':main()
