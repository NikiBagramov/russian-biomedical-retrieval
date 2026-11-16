"""Export query-level ranks/grades for independent dev and test metric checks."""
from study import *
from common import write_jsonl
import csv

rels=read(ROOT/'data/dev_qrels.json');old=read(BASE/'cache/run_selected.json');old={q:old[q] for q in rels}
measure('phase1_selected_dev',old,rels,dict(purpose='Audit export of immutable phase1 baseline ranks; not a new fitted model'))
rows=[]
for path in sorted((ROOT/'artifacts').glob('result_*.json')):
    if 'smoke' in path.name:continue
    item=read(path);name=item['name'];run=read(ROOT/'cache'/('run_'+name+'.json'))
    for q in item['per_query']:
        for rank,d in enumerate(sorted(run[q],key=lambda d:(-run[q][d],d)),1):rows.append(dict(method=name,query_id=q,doc_id=d,rank=rank,relevance=rels[q].get(d,0)))
write_jsonl(ROOT/'artifacts/dev_predictions.jsonl',rows)
dump(ROOT/'artifacts/dev_relevance_summary.json',{q:dict(relevant_documents=sum(v>0 for v in v.values()),ideal_grades=sorted(v.values(),reverse=True)[:10]) for q,v in rels.items()})
result=read(ROOT/'artifacts/final_results.json')
with (ROOT/'artifacts/results.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['method']+list(next(iter(result['metrics'].values()))),lineterminator='\n');writer.writeheader();writer.writerows(dict(method=n,**v) for n,v in result['metrics'].items())
print('Public phase2 rankings exported',len(rows),'development rows',flush=True)
