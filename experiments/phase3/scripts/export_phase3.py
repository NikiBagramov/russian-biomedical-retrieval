"""Publish numerical evidence without redistributing texts or model weights."""
from study import *
from common import write_jsonl

assert (ROOT/'artifacts/final_results.json').exists()
selection=read(ROOT/'artifacts/selection_phase3.json')
ids,_,_,qrels=inputs()
scores=read(ROOT/'cache/pair_scores_qwen4_2048.json')
rows=[]
for name in selection['grid']['pools']:
    pool=read(ROOT/'cache'/('run_'+name+'.json'))
    for q in ids:
        for rank,d in enumerate(sorted(pool[q],key=lambda d:(-pool[q][d],d)),1):
            rows.append(dict(pool=name,query_id=q,doc_id=d,rank=rank,
                retrieval_score=pool[q][d],qwen_probability=scores.get(q+'\t'+d),relevance=qrels[q].get(d,0)))
write_jsonl(ROOT/'artifacts/dev_candidate_evidence.jsonl',rows)
dump(ROOT/'artifacts/dev_relevance_summary.json',{q:dict(relevant_documents=sum(g>0 for g in rel.values()),ideal_grades=sorted(rel.values(),reverse=True)[:10]) for q,rel in qrels.items()})
lock=read(ROOT/'artifacts/evaluation_lock.json');test_pool=read(ROOT/'cache/frozen_phase3_test_candidates.json');test_scores=read(ROOT/'cache/phase3_test_pair_scores.json');old_path=BASE/'runs/stronger_20261116/cache/frozen_test_pair_scores.json'
old_pairs=read(old_path) if old_path.exists() and lock['previous_test_pair_cache']['sha256'] else {}
if old_pairs:assert sha(old_path)==lock['previous_test_pair_cache']['sha256']
test_rows=[]
for q,pool in test_pool.items():
    for rank,d in enumerate(sorted(pool,key=lambda d:(-pool[d],d))[:lock['selection']['depth']],1):
        key=q+'\t'+d;value=test_scores[key]
        if key in old_pairs:assert value==old_pairs[key]
        test_rows.append(dict(query_id=q,doc_id=d,head_rank=rank,qwen_probability=value,source='phase2_reused' if key in old_pairs else 'phase3_new'))
write_jsonl(ROOT/'artifacts/test_pair_evidence.jsonl',test_rows)
reused=sum(r['source']=='phase2_reused' for r in test_rows)
dump(ROOT/'artifacts/test_pair_audit.json',dict(queries=len(test_pool),depth=lock['selection']['depth'],needed_pairs=len(test_rows),reused_in_actual_heads=reused,new_in_actual_heads=len(test_rows)-reused,retained_phase2_cache_pairs=len(old_pairs),last_resume_new_pairs=read(ROOT/'artifacts/test_inference.json')['new_pairs'],scope='Exact pair counts; last resume count is not total new phase3 work'))
dump(ROOT/'artifacts/public_evidence_manifest.json',dict(files={name:sha(ROOT/'artifacts'/name) for name in ['dev_candidate_evidence.jsonl','dev_relevance_summary.json','final_predictions.jsonl','test_relevance_summary.json','test_pair_evidence.jsonl','test_pair_audit.json','selection_phase3.json','final_results.json','evaluation_lock.json','protocol.json','score_controls_trials.json','giga_screen_controls_trials.json']},scope='Numerical ranks, probabilities and judgments; no source documents, queries or weights'))
print('PHASE3 NUMERICAL EVIDENCE EXPORTED',len(rows),flush=True)
