"""Lock the complete dev choice, models, code and unlabeled inputs before test."""
from common import ROOT,read,dump,sha,now
from frozen_candidates import BASE,stage_config,candidate_pool
from ranking_math import rerank

assert not (ROOT/'artifacts/evaluation_lock.json').exists()
selected=read(ROOT/'artifacts/selection_phase3.json');assert not selected['interim'] and selected['selection_split']=='dev324'
config=selected['selected'];stage=stage_config(config['pool']);ids=read(ROOT/'artifacts/protocol.json')['full_dev_query_ids']
pool=candidate_pool(stage,ids,'dev');stored=read(ROOT/'cache'/('run_'+config['pool']+'.json'))
assert pool==stored,'Frozen candidate builder does not reproduce all324dev pools'
scores=read(ROOT/'cache/pair_scores_qwen4_2048.json');reconstructed={q:rerank(pool[q],scores,q,config) for q in ids}
assert reconstructed==read(ROOT/'cache/run_phase3_selected_dev.json')
files=['common.py','study.py','metrics.py','sparse.py','ranking_math.py','qwen_probe.py','qwen_adaptive.py','giga_encode.py','giga_pool.py','choose_phase3.py','frozen_candidates.py','freeze_phase3.py','predict_phase3.py','evaluate_phase3.py','pointwise_phase2.py']
assert all((ROOT/'scripts'/name).exists() for name in files)
for name,digest in selected['source_sha256'].items():assert sha(ROOT/'scripts'/name)==digest
parity=read(ROOT/'artifacts/inference_parity.json');assert parity['passed'] and parity['source_sha256']==sha(ROOT/'scripts/pointwise_phase2.py')
assert sha(ROOT/'scripts/pointwise_phase2.py')==sha(BASE/'experiments/phase2/scripts/frozen_pipeline.py')
models=dict(qwen4=read(ROOT/'artifacts/model_qwen4.json'),baseline=read(BASE/'artifacts/model_provenance.json'))
encoding=None;giga_inputs={}
if stage['kind']=='giga_fusion':
    models['giga']=read(ROOT/'artifacts/model_giga.json');encoding=read(ROOT/'artifacts/encoding_giga_nf4_l1024.json')
    assert encoding['code_sha256']==sha(ROOT/'scripts/giga_encode.py')
    giga_inputs={n:sha(ROOT/'cache'/n) for n in ['embeddings_giga_nf4_l1024.npz','scores_giga_nf4_l1024_'+stage['prompt']+'.npy']}
assert sha(ROOT/'data/corpus.jsonl')==sha(BASE/'data/processed/corpus.jsonl')
reuse=BASE/'runs/stronger_20261116/cache/frozen_test_pair_scores.json'
record=dict(created=now(),selection=config,first_stage=stage,dev_metrics=selected['metrics'],model_provenance=models,encoding=encoding,
 source_sha256={n:sha(ROOT/'scripts'/n) for n in files},selection_sha256=sha(ROOT/'artifacts/selection_phase3.json'),giga_input_sha256=giga_inputs,
 corpus_sha256=sha(ROOT/'data/corpus.jsonl'),queries_sha256=sha(BASE/'data/processed/queries.jsonl'),index_sha256=sha(BASE/'cache/index.json'),
 base_score_sha256={n:sha(BASE/'cache'/n) for n in ['scores_bm25.npy','scores_e5_large.npy','scores_qwen_embedding.npy']},
 dev_candidate_parity=True,all324_dev_ranking_parity=True,
 previous_test_pair_cache=dict(optional=True,sha256=sha(reuse) if reuse.exists() else None,purpose='Inference-only reuse of identical query/document/model/template pairs from phase2; no relevance labels'),
 inference_parity=parity,baseline_test_ndcg10=read(ROOT/'artifacts/protocol.json')['baseline_test_ndcg10'],target_test_ndcg10=read(ROOT/'artifacts/protocol.json')['targets']['relative_5pct_over_own'],
 test_policy='Original benchmark test was previously evaluated in phases1/2. Phase3 choice uses dev only; configuration and source locked before phase3 test inference/relevance evaluation.')
dump(ROOT/'artifacts/evaluation_lock.json',record);print('PHASE3 FROZEN',config,selected['metrics'],flush=True)
