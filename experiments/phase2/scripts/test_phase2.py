"""Predict all official test queries with the already frozen phase-two pipeline."""
from frozen_pipeline import *

lock=read(ROOT/'artifacts/evaluation_lock.json')
for name,digest in lock['source_sha256'].items():assert sha(ROOT/'scripts'/name)==digest,name
assert sha(ROOT/'cache/frozen_candidate_pool.json')==lock['candidate_pool_sha256']
rows=[r for r in jsonl(BASE/'data/processed/queries.jsonl') if r['split']=='test'];ids=[r['_id'] for r in rows];assert len(ids)==323
assert not set(ids)&set(read(ROOT/'artifacts/protocol.json')['full_dev_query_ids'])
queries={r['_id']:r['text'] for r in rows};docs={r['_id']:r['title']+'\n'+r['text'] for r in jsonl(ROOT/'data/corpus.jsonl')};pool=read(ROOT/'cache/frozen_candidate_pool.json')
run,scores,timing=score_pairs(ids,queries,docs,pool,ROOT/'cache/frozen_test_pair_scores.json',depth=lock['depth'])
dump(ROOT/'cache/run_phase2_test.json',run);dump(ROOT/'artifacts/test_inference.json',dict(rows=len(ids),pairs=len(scores),run_sha256=sha(ROOT/'cache/run_phase2_test.json'),lock_sha256=sha(ROOT/'artifacts/evaluation_lock.json'),timing=timing))
print('PHASE2 TEST PREDICTIONS COMPLETE',len(ids),flush=True)
