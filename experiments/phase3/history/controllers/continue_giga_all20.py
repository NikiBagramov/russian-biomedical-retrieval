"""Complete all three inexpensive full20 pools before considering deeper scoring."""
import os,sys,time,subprocess,psutil
from common import ROOT,read,dump,sha,now

status=ROOT/'artifacts/jobs/giga_pipeline.json';previous=read(ROOT/'artifacts/jobs/giga_pipeline_before_all20.json')
record=dict(pid=os.getpid(),state='waiting_for_existing_full20',started=now(),child_pid=previous['child_pid'],controller_source_sha256=sha(__file__))
dump(status,record)
try:
    if psutil.pid_exists(previous['child_pid']):
        existing=psutil.Process(previous['child_pid']);assert any(str(ROOT/'scripts/qwen_adaptive.py').lower()==s.lower() for s in existing.cmdline())
        while existing.is_running():time.sleep(10)
    def call(script,*args):
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts'/script),*args],cwd=ROOT)
        record.update(state='running',stage=script,args=args,child_pid=child.pid);dump(status,record)
        code=child.wait()
        if code:raise RuntimeError(script+' failed '+str(code))
    assert (ROOT/'artifacts/result_adaptive_giga_pool2_dev_d20_full.json').exists(),'Inherited full20 worker did not complete'
    for i in [1,2,3]:
        pool='giga_pool'+str(i)+'_dev';path=ROOT/'artifacts'/('result_adaptive_'+pool+'_d20_full.json')
        if not path.exists():call('qwen_adaptive.py','--pool',pool,'--depth','20','--full','--legacy')
        item=read(path);assert item['queries']==324 and item['metadata']['code_sha256']==sha(ROOT/'scripts/qwen_adaptive.py')
    call('choose_phase3.py','--interim');interim=read(ROOT/'artifacts/interim_selection_phase3.json')
    baseline=interim['baseline_phase2_dev']['ndcg_cut_10'];target=baseline*1.075;pool=interim['selected']['pool']
    assert pool.startswith('giga_pool')
    decision=dict(created=now(),pool=pool,full20_available=['giga_pool1_dev','giga_pool2_dev','giga_pool3_dev'],best_dev_after_all20=interim['metrics']['ndcg_cut_10'],dev_heuristic_target=target,policy='All3 complete324dev20 pools before deeper scoring.50 if best dev remains below1.075xold dev;100 only if within-pool best50-best20>.002 and overall margin still below1.075. No test labels used; heuristic is not a test guarantee.')
    dump(ROOT/'artifacts/giga_depth_decision.json',decision)
    if interim['metrics']['ndcg_cut_10']<target:
        call('qwen_adaptive.py','--pool',pool,'--depth','50','--full','--legacy');call('choose_phase3.py','--interim')
        interim=read(ROOT/'artifacts/interim_selection_phase3.json');best={d:max(t['metrics']['ndcg_cut_10'] for t in interim['trials'] if t['config']['pool']==pool and t['config']['depth']==d) for d in [20,50]}
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
