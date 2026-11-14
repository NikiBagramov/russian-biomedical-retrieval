from common import ROOT,jsonl,read,dump

def main():
    per=read(ROOT/'artifacts/per_query_metrics.json');queries={r['_id']:r['text'] for r in jsonl(ROOT/'data/processed/queries.jsonl')};corpus={r['_id']:r for r in jsonl(ROOT/'data/processed/corpus.jsonl')};ours=read(ROOT/'cache/run_selected.json');base=read(ROOT/'cache/run_bm25_bge_reranker512.json');qrels=read(ROOT/'data/processed/test_qrels.json')
    ids=list(qrels);delta=lambda q:per['selected'][q]['ndcg_cut_10']-per['bm25_bge_reranker512'][q]['ndcg_cut_10'];selected=[]
    for kind,q in [('largest_gain',max(ids,key=lambda q:(delta(q),q))),('largest_regression',min(ids,key=lambda q:(delta(q),q))),('median_difference',sorted(ids,key=lambda q:(delta(q),q))[len(ids)//2])]:
        selected.append(dict(type=kind,query_id=q,query=queries[q],ndcg_delta=delta(q),rankings={name:[dict(doc_id=d,title=corpus[d]['title'][:180],grade=qrels[q].get(d,0)) for d in sorted(run[q],key=lambda d:(-run[q][d],d))[:3]] for name,run in [('selected',ours),('BM25+BGE',base)]}))
    dump(ROOT/'artifacts/error_analysis.json',dict(selection='Extremal and median per-query nDCG differences after the frozen official evaluation',cases=selected))
if __name__=='__main__':main()
