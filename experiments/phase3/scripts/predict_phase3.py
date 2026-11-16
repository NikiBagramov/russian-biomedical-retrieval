"""All323 test predictions from the frozen choice; never reads relevance."""
import subprocess,sys
from common import ROOT,read,dump,sha
from frozen_candidates import BASE,PORTFOLIO,assert_sources,test_inputs,candidate_pool
from ranking_math import rerank
from pointwise_phase2 import score_pairs

lock_path=ROOT/'artifacts/evaluation_lock.json';lock=read(lock_path);assert_sources(lock)
config=lock['selection'];stage=lock['first_stage'];lock_sha=sha(lock_path)
output=ROOT/'cache/run_phase3_test.json';manifest=ROOT/'artifacts/test_inference.json'
if output.exists() and manifest.exists():
    old=read(manifest);assert old['lock_sha256']==lock_sha and old['run_sha256']==sha(output)
    print('Frozen test predictions already complete',flush=True);raise SystemExit(0)
if stage['kind']=='giga_fusion' and not (ROOT/'cache/embeddings_giga_nf4_l1024_test.npz').exists():
    subprocess.run([sys.executable,str(ROOT/'scripts/giga_encode.py'),'--test'],check=True,cwd=ROOT)
if stage['kind']=='giga_fusion':
    encoding=read(ROOT/'artifacts/encoding_giga_nf4_l1024_test.json')
    assert encoding['code_sha256']==lock['source_sha256']['giga_encode.py'] and encoding['model']==lock['model_provenance']['giga']
ids,queries,docs=test_inputs();pool=candidate_pool(stage,ids,'test');pool_path=ROOT/'cache/frozen_phase3_test_candidates.json';dump(pool_path,pool)
dump(ROOT/'artifacts/candidate_pool_lock.json',dict(lock_sha256=lock_sha,candidate_sha256=sha(pool_path),queries=len(ids)))
cache=ROOT/'cache/phase3_test_pair_scores.json';cache_state=ROOT/'artifacts/test_pair_cache_state.json';reused=0
if cache.exists():
    assert read(cache_state)['lock_sha256']==lock_sha;scores=read(cache)
else:
    original=BASE/'runs/stronger_20261116/cache/frozen_test_pair_scores.json';scores={}
    if original.exists() and lock['previous_test_pair_cache']['sha256']:
        assert sha(original)==lock['previous_test_pair_cache']['sha256'];scores=read(original);reused=len(scores)
        assert all(k.split('\t')[0] in set(ids) for k in scores)
    dump(cache_state,dict(lock_sha256=lock_sha,initial_reused_pairs=reused));dump(cache,scores)
_,scores,timing=score_pairs(ids,queries,docs,pool,cache,depth=config['depth'])
run={q:rerank(pool[q],scores,q,config) for q in ids};dump(output,run)
dump(manifest,dict(rows=len(ids),config=config,**timing,cached_pair_count=len(scores),initial_reused_pairs=read(cache_state)['initial_reused_pairs'],scoring_source_sha256=sha(ROOT/'scripts/pointwise_phase2.py'),run_sha256=sha(output),lock_sha256=lock_sha,candidate_sha256=sha(pool_path)))
print('PHASE3 TEST PREDICTIONS COMPLETE',len(ids),flush=True)
