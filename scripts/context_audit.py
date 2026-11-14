"""Post-inference token-length audit; no qrels or predictions are changed."""
from common import ROOT,jsonl,read,dump,model_path,sha
from transformers import AutoTokenizer

def main():
    corpus={r['_id']:r['title']+'\n'+r['text'] for r in jsonl(ROOT/'data/processed/corpus.jsonl')};queries={r['_id']:r['text'] for r in jsonl(ROOT/'data/processed/queries.jsonl')};scores=read(ROOT/'cache/pair_scores_bge_reranker_512.json');keys=list(scores);tok=AutoTokenizer.from_pretrained(model_path('bge_reranker'),local_files_only=True);truncated=0;clipped=0
    for start in range(0,len(keys),64):
        pairs=[k.split('\t') for k in keys[start:start+64]];raw=[(queries[q][:8192],corpus[d][:8192]) for q,d in pairs];clipped+=sum(len(queries[q])>8192 or len(corpus[d])>8192 for q,d in pairs)
        ids=tok(raw,truncation=False)['input_ids'];truncated+=sum(len(v)>512 for v in ids)
    result=dict(bge_unique_pairs=len(keys),bge_pair_tokens_over_512=truncated,bge_pair_chars_clipped=clipped,scope='All unique scored query/document pairs, measured after inference without reading qrels',code_sha256=sha(__file__))
    dump(ROOT/'artifacts/context_audit.json',result);runtime=read(ROOT/'artifacts/neural_runtime.json');runtime['bge_reranker512']['truncated_new_pairs']=None;runtime['bge_reranker512']['truncated_all_pairs']=truncated;runtime['bge_reranker512']['length_audit_note']='Original inference did not count BGE truncations; post-hoc label-blind context_audit counts all cached pairs.';dump(ROOT/'artifacts/neural_runtime.json',runtime);print(result,flush=True)
if __name__=='__main__':main()
