"""Freeze dev-selected configuration and measured inputs before test inference."""
from study import *
from frozen_pipeline import candidate_pool

assert not (ROOT/'artifacts/evaluation_lock.json').exists()
assert read(ROOT/'artifacts/inference_parity.json')['passed']
selection=read(ROOT/'artifacts/selection_trials_phase2.json')['selected']
assert selection['qwen4_weight']==1. and selection['phase1_weight']==0. and selection['retrieval_weight']==0.
pool=candidate_pool();dump(ROOT/'cache/frozen_candidate_pool.json',pool)
source_files=['common.py','study.py','metrics.py','sparse.py','qwen_probe.py','frozen_pipeline.py','freeze_phase2.py','test_phase2.py','evaluate_phase2.py']
record=dict(selection=selection,retrieval=read(ROOT/'artifacts/result_score_fusion_dev.json')['metadata']['selected'],depth=20,max_length=2048,quantization='NF4 double quantization; BF16 computation',model_provenance=read(ROOT/'artifacts/model_qwen4.json'),source_sha256={f:sha(ROOT/'scripts'/f) for f in source_files},candidate_pool_sha256=sha(ROOT/'cache/frozen_candidate_pool.json'),dev_run_sha256=sha(ROOT/'cache/run_qwen4_fusion_d20_l2048_full.json'),phase1_commit='aab0c59a4a08736dac40117cda2e759a5818732c',test_policy='Original benchmark test was evaluated in phase1. Phase2 development used only official dev; new configuration frozen before phase2 test inference and relevance evaluation.')
dump(ROOT/'artifacts/evaluation_lock.json',record)
print('PHASE2 CONFIGURATION FROZEN',selection,flush=True)
