"""Replay phase3 in a fresh directory, optionally after a fresh full phase2 run."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone

SOURCE=Path(__file__).resolve().parents[1];ORIGINAL_BASE=SOURCE.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',action='store_true');ap.add_argument('--download',action='store_true');ap.add_argument('--from-phase2',type=Path);ap.add_argument('--report',action='store_true');args=ap.parse_args()
    stages=['Fresh baseline and phase2 reproduction, unless explicit --from-phase2 cache supplied','Fresh dev-only phase3 directory; inherited64query screen and324dev','Old-pool50/100 screens, full50, fixed72 ranking controls and real40pair resource parity','Pinned Giga weights/checksums, reviewed custom-code hashes, NF4 smoke and full corpus/dev encoding','Full54 first-stage dev trials, three Qwen top20 screens, existing score/tie controls','Full20 before50; within-pool depth promotion under recorded7.5%dev heuristic','All-complete324dev selection; stop if final configuration differs from publication','Source/model/input lock,323test predictions and one relevance evaluation','Export numerical evidence; independent arithmetic and source-hash verification']
    if args.plan:
        print(json.dumps(dict(stages=stages,device='One BF16 CUDA inference worker; measured RTX3060Laptop6GB',models='Pinned open weights; no paid APIs',runtime='Hours including prior phases; numerical differences on other hardware can change the dev winner',cache_mode='--from-phase2 reuses an existing completed phase2; omit it for a fresh full replay',scope='Replay of the recorded adaptive dev study; not a claim of fully preregistered architecture choices'),ensure_ascii=False,indent=2));return
    if not args.download:ap.error('Use --download for full execution, or --plan to inspect the stages')
    phase2=args.from_phase2.resolve() if args.from_phase2 else None
    if phase2 is None:
        before=set((ORIGINAL_BASE/'runs').glob('phase2_reproduce_*'))
        subprocess.run([sys.executable,str(ORIGINAL_BASE/'experiments/phase2/scripts/reproduce.py'),'--download'],cwd=ORIGINAL_BASE,check=True)
        created=set((ORIGINAL_BASE/'runs').glob('phase2_reproduce_*'))-before
        assert len(created)==1,'Cannot identify the fresh phase2 workspace'
        phase2=created.pop()/'russian-biomedical-retrieval/experiments/phase2'
    baseline=phase2.parents[1];assert (phase2/'artifacts/final_results.json').exists()
    assert read(phase2/'artifacts/final_results.json')==read(baseline/'experiments/phase2/artifacts/final_results.json'),'Use a phase2 cache matching its baseline public artifact'
    run=baseline/'runs'/('phase3_reproduce_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
    shutil.copytree(SOURCE/'scripts',run/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
    for folder in ['data','cache','artifacts']:(run/folder).mkdir()
    for name in ['corpus.jsonl','queries.jsonl','dev_qrels.json']:shutil.copyfile(phase2/'data'/name,run/'data'/name)
    for name in ['pair_scores_qwen4_2048.json','run_score_fusion_dev.json','run_qwen4_fusion_d20_l2048_full.json']:shutil.copyfile(phase2/'cache'/name,run/'cache'/name)
    for name in ['model_qwen4.json','result_score_fusion_dev.json']:shutil.copyfile(phase2/'artifacts'/name,run/'artifacts'/name)
    shutil.copyfile(SOURCE/'artifacts/giga_source_review_inputs.json',run/'artifacts/giga_source_review_inputs.json')
    protocol=read(SOURCE/'artifacts/protocol.json');own=read(phase2/'artifacts/final_results.json')['metrics']['phase2_selected']['ndcg_cut_10']
    protocol.update(created=datetime.now(timezone.utc).isoformat(),reproduction_of=str(SOURCE),baseline_test_ndcg10=own)
    protocol['targets']['relative_5pct_over_own']=own*1.05;dump(run/'artifacts/protocol.json',protocol)
    dev=read(run/'data/dev_qrels.json');assert set(dev)==set(protocol['full_dev_query_ids'])
    assert all(k.split('\t')[0] in dev for k in read(run/'cache/pair_scores_qwen4_2048.json'))
    env=os.environ.copy();env['NLP_SHARED_MODELS']=str(baseline.parent/'cache/models')
    def call(script,*extra):subprocess.run([sys.executable,'-u',str(run/'scripts'/script),*extra],cwd=run,env=env,check=True)
    for extra in [['--depth','50'],['--depth','100'],['--depth','50','--full']]:call('qwen_probe.py',*extra)
    call('score_controls.py');call('check_ranking_math.py');call('qwen_adaptive.py','--parity')
    dump(run/'artifacts/jobs/continuation.json',dict(state='not_scheduled',reason='Old-pool full100 was explicitly deferred in the recorded study'))
    dump(run/'artifacts/depth_family_disposition.json',dict(action='defer_full100',scope='Old candidate pool only; replay of the resource decision'))
    call('download_model.py','--repo','ai-sage/Giga-Embeddings-instruct','--revision','2cf0fdc97194aaedf10ac0e6bf798834acd31042','--alias','giga_embedding','--manifest','model_giga.json')
    call('giga_encode.py','--smoke');call('giga_encode.py');call('giga_pool.py')
    for i in [1,2,3]:call('qwen_adaptive.py','--pool','giga_pool'+str(i)+'_dev','--depth','20','--legacy')
    call('giga_screen_controls.py')
    control=read(run/'artifacts/giga_screen_controls_trials.json')['best'];pool=control['config']['pool']
    if control['metrics']['ndcg_cut_10']>=.95*read(run/'artifacts/score_controls_trials.json')['best']['ndcg10']:
        order=[pool]+[n for n in ['giga_pool1_dev','giga_pool2_dev','giga_pool3_dev'] if n!=pool]
        for name in order:call('qwen_adaptive.py','--pool',name,'--depth','20','--full','--legacy')
        call('choose_phase3.py','--interim')
        interim=read(run/'artifacts/interim_selection_phase3.json');target=interim['baseline_phase2_dev']['ndcg_cut_10']*1.075
        pool=interim['selected']['pool'];assert pool.startswith('giga_pool')
        if interim['metrics']['ndcg_cut_10']<target:
            call('qwen_adaptive.py','--pool',pool,'--depth','50','--full','--legacy');call('choose_phase3.py','--interim')
            interim=read(run/'artifacts/interim_selection_phase3.json');best={d:max(t['metrics']['ndcg_cut_10'] for t in interim['trials'] if t['config']['pool']==pool and t['config']['depth']==d) for d in [20,50]}
            if best[50]>best[20]+.002 and interim['metrics']['ndcg_cut_10']<target:call('qwen_adaptive.py','--pool',pool,'--depth','100','--full','--legacy')
    dump(run/'artifacts/jobs/giga_pipeline.json',dict(state='complete',pid=os.getpid(),exit_code=0,controller='reproduce.py',scope='All actual replay stages returned0; Qwen always legacy batch2'))
    call('choose_phase3.py')
    actual=read(run/'artifacts/selection_phase3.json')['selected'];expected=read(SOURCE/'artifacts/selection_phase3.json')['selected']
    assert actual==expected,f'Dev winner differs from publication: {actual} versus {expected}; no phase3 test evaluation performed'
    for script in ['pointwise_phase2.py','freeze_phase3.py','predict_phase3.py','evaluate_phase3.py','export_phase3.py','verify_phase3.py']:call(script)
    if args.report:
        call('make_paper_phase3.py','--experiment',str(run))
        subprocess.run([sys.executable,str(baseline/'scripts/build_report.py')],cwd=baseline,env=env,check=True)
    print('Complete independent phase3 reproduction:',run,flush=True)

if __name__=='__main__':main()
