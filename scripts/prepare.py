"""Preserve the complete official splits without exposing test relevance."""
import csv,re
from collections import Counter
from common import ROOT,jsonl,dump,write_jsonl,text_hash,sha,development_allowed

def main():
    development_allowed();base=ROOT/'data/raw';corpus=jsonl(base/'rus-nfcorpus/corpus.jsonl');queries=jsonl(base/'rus-nfcorpus/queries.jsonl')
    ids={r['_id'] for r in corpus};assert len(ids)==3633
    query_by_id={r['_id']:r for r in queries};assert len(query_by_id)==3237
    qrels={};splits={}
    for split in ['train','dev','test']:
        entries=list(csv.DictReader((base/'rus-nfcorpus-qrels'/(split+'.tsv')).open(encoding='utf-8'),delimiter='\t'))
        rel={}
        for e in entries:
            q,d,s=e['query-id'],e['corpus-id'],int(e['score']);assert q in query_by_id and d in ids
            assert d not in rel.setdefault(q,{})
            rel[q][d]=s
        splits[split]=list(rel);qrels[split]=rel
    assert [len(splits[s]) for s in ['train','dev','test']]==[2590,324,323]
    assert not (set(splits['train'])&set(splits['dev']) or set(splits['train'])&set(splits['test']) or set(splits['dev'])&set(splits['test']))
    rows=[dict(query_by_id[q],split=s) for s,qs in splits.items() for q in qs]
    write_jsonl(ROOT/'data/processed/corpus.jsonl',corpus);write_jsonl(ROOT/'data/processed/queries.jsonl',rows)
    dump(ROOT/'data/processed/development_qrels.json',{s:qrels[s] for s in ['train','dev']});dump(ROOT/'data/processed/test_qrels.json',qrels['test'])
    norm=lambda x:re.sub(r'\s+',' ',x.lower().strip())
    trtexts={norm(query_by_id[q]['text']) for q in splits['train']}
    audit=dict(corpus=len(corpus),queries=len(queries),splits={s:dict(queries=len(qs),qrels=sum(len(x) for x in qrels[s].values()),grades=dict(Counter(v for r in qrels[s].values() for v in r.values())),duplicate_train_text=sum(norm(query_by_id[q]['text']) in trtexts for q in qs) if s!='train' else None) for s,qs in splits.items()},empty_documents=sum(not(r['title']+r['text']).strip() for r in corpus),max_document_chars=max(len(r['title']+r['text']) for r in corpus))
    dump(ROOT/'artifacts/audit.json',audit);dump(ROOT/'artifacts/splits.json',dict(splits=splits,query_sha256=sha(ROOT/'data/processed/queries.jsonl'),corpus_sha256=sha(ROOT/'data/processed/corpus.jsonl')))
    print(audit,flush=True)
if __name__=='__main__':main()
