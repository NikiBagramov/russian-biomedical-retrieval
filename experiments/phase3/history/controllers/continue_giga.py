"""Resource-aware continuation: finish live screen, then full20 before full50."""
import os,sys,time,subprocess,psutil
from common import ROOT,read,dump,sha,now

status=ROOT/'artifacts/jobs/giga_pipeline.json'
previous=read(ROOT/'artifacts/jobs/giga_pipeline_before_full20_amendment.json')
record=dict(pid=os.getpid(),state='waiting_for_existing_screen',started=now(),inherited_child_pid=previous['child_pid'],controller_source_sha256=sha(__file__))
dump(status,record)
try:
    existing=psutil.Process(previous['child_pid']) if psutil.pid_exists(previous['child_pid']) else None
    if existing:
        assert any(str(ROOT/'scripts/qwen_adaptive.py').lower()==s.lower() for s in existing.cmdline())
        record['child_pid']=existing.pid;dump(status,record)
        while existing.is_running():time.sleep(10)
    def call(script,*args):
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts'/script),*args],cwd=ROOT)
        record.update(state='running',stage=script,args=args,child_pid=child.pid);dump(status,record)
        code=child.wait()
        if code:raise RuntimeError(script+' failed '+str(code))
    def result(pool,depth,full=False):return ROOT/'artifacts'/('result_adaptive_'+pool+'_d'+str(depth)+'_'+('full' if full else 'screen')+'.json')
    screens=[]
    for i in [1,2,3]:
        pool='giga_pool'+str(i)+'_dev';path=result(pool,20)
        if not path.exists():call('qwen_adaptive.py','--pool',pool,'--depth','20','--legacy')
        item=read(path);assert item['queries']==64 and item['metadata']['code_sha256']==sha(ROOT/'scripts/qwen_adaptive.py')
        screens.append((item['metrics']['ndcg_cut_10'],pool))
    call('giga_screen_controls.py')
    control=read(ROOT/'artifacts/giga_screen_controls_trials.json')['best']
    score=control['metrics']['ndcg_cut_10'];pool=control['config']['pool'];reference=read(ROOT/'artifacts/score_controls_trials.json')['best']['ndcg10']
    record.update(raw_screens=screens,screen_best=dict(score=score,pool=pool,config=control['config']),screen_reference=reference);dump(status,record)
    if score>=.95*reference:
        call('qwen_adaptive.py','--pool',pool,'--depth','20','--full','--legacy')
        call('choose_phase3.py','--interim')
        interim=read(ROOT/'artifacts/interim_selection_phase3.json')
        baseline=interim['baseline_phase2_dev']['ndcg_cut_10'];target=baseline*1.075
        decision=dict(created=now(),pool=pool,full20_available=True,best_dev_after_full20=interim['metrics']['ndcg_cut_10'],dev_heuristic_margin=1.075,dev_heuristic_target=target,policy='Proceed to50 only if all-complete dev selection remains below heuristic7.5% margin; proceed to100 only if within-pool best50 improves best20 by>.002 and overall margin still below7.5%. No test labels used; dev margin is not a test guarantee.')
        dump(ROOT/'artifacts/giga_depth_decision.json',decision)
        if interim['metrics']['ndcg_cut_10']<target:
            call('qwen_adaptive.py','--pool',pool,'--depth','50','--full','--legacy')
            call('choose_phase3.py','--interim');interim=read(ROOT/'artifacts/interim_selection_phase3.json')
            best={d:max(t['metrics']['ndcg_cut_10'] for t in interim['trials'] if t['config']['pool']==pool and t['config']['depth']==d) for d in [20,50]}
            decision.update(best20=best[20],best50=best[50],best_dev_after_full50=interim['metrics']['ndcg_cut_10'])
            if best[50]>best[20]+.002 and interim['metrics']['ndcg_cut_10']<target:
                decision['full100']='promoted';dump(ROOT/'artifacts/giga_depth_decision.json',decision)
                call('qwen_adaptive.py','--pool',pool,'--depth','100','--full','--legacy')
            else:decision['full100']='deferred: insufficient within-pool depth gain or dev margin reached'
        else:decision.update(full50='deferred: dev heuristic margin reached',full100='deferred: dev heuristic margin reached')
        dump(ROOT/'artifacts/giga_depth_decision.json',decision)
    record.update(state='complete',finished=now(),exit_code=0);dump(status,record)
except BaseException as e:
    record.update(state='failed',error=str(e),finished=now());dump(status,record);raise
