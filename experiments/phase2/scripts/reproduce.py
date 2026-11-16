"""Create a fresh workspace and replay the actual baseline and phase-two stages."""
import argparse,os,sys,subprocess,shutil,json
from pathlib import Path
from datetime import datetime
SOURCE=Path(__file__).resolve().parents[1];BASE=SOURCE.parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--download',action='store_true');ap.add_argument('--report',action='store_true');ap.add_argument('--plan',action='store_true');args=ap.parse_args()
stages=['fresh baseline run: data/models, sparse/dense/rerank/dev selection','phase2 dev-only input preparation','fixed score-fusion grid','Jina v3.5 screen, Qwen0.6 deeper screen, grounded bilingual expansion screen','Qwen4 NF4 screen and full dev','fixed15 final grid,40-pair parity, freeze','323-query test prediction and one final evaluation','independent public-rank verification']
if args.plan:print(json.dumps(dict(stages=stages,device='BF16 CUDA GPU, measured on6GB RTX3060',models='pinned open weights; no external paid API',expected='hours; no bit-exact cross-hardware guarantee'),ensure_ascii=False,indent=2));raise SystemExit(0)
if not args.download:ap.error('Use --download for a complete fresh isolated run, or --plan to inspect stages')
workspace=BASE/'runs'/('phase2_reproduce_'+datetime.now().strftime('%Y%m%d_%H%M%S'));baseline=workspace/'russian-biomedical-retrieval';baseline.mkdir(parents=True)
for name in ['scripts','third_party']:
    if (BASE/name).exists():shutil.copytree(BASE/name,baseline/name,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['project.json','PROTOCOL.md','requirements.txt']:shutil.copyfile(BASE/name,baseline/name)
(baseline/'artifacts').mkdir()
for name in ['data_provenance.json','model_provenance.json','upstream_code_provenance.json']:
    if (BASE/'artifacts'/name).exists():shutil.copyfile(BASE/'artifacts'/name,baseline/'artifacts'/name)
run=baseline/'experiments/phase2';shutil.copytree(SOURCE/'scripts',run/'scripts',ignore=shutil.ignore_patterns('__pycache__'));(run/'artifacts').mkdir();shutil.copyfile(SOURCE/'artifacts/protocol.json',run/'artifacts/protocol.json')
env=os.environ.copy();env['NLP_SHARED_MODELS']=str(workspace/'cache/models')
def call(path,*extra):subprocess.run([sys.executable,str(path),*extra],cwd=baseline,env=env,check=True)
call(baseline/'scripts/run_pipeline.py','--run','--download')
for repo,rev,alias,manifest in [('Qwen/Qwen3-Reranker-4B','22e683669bc0f0bd69640a1354a6d0aebcfeede5','qwen_reranker4','model_qwen4.json'),('jinaai/jina-reranker-v3.5','e8a93f33f0b22108f8c2364f8484ce3422552fbc','jina_reranker35','model_jina35.json'),('Qwen/Qwen3-1.7B','70d244cc86ccca08cf5af4e1e306ecf908b1ad5e','qwen_generation','model_generation.json')]:call(run/'scripts/download_model.py','--repo',repo,'--revision',rev,'--alias',alias,'--manifest',manifest)
for name,extra in [('prepare_phase2.py',[]),('retrieval_screen.py',[]),('jina_probe.py',['--depth','20']),('qwen_probe.py',['--size','06','--depth','50']),('expand_queries.py',[]),('qwen_probe.py',['--smoke','--depth','20']),('qwen_probe.py',['--depth','20']),('qwen_probe.py',['--full','--depth','20']),('select_phase2.py',[]),('frozen_pipeline.py',[]),('freeze_phase2.py',[]),('test_phase2.py',[]),('evaluate_phase2.py',[])]:call(run/'scripts'/name,*extra)
call(run/'scripts/export_results.py');call(baseline/'scripts/run_pipeline.py','--verify')
if args.report:call(baseline/'scripts/run_pipeline.py','--report')
print('Complete independent reproduction:',workspace)
