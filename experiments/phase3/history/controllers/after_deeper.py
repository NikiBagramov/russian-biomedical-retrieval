"""One GPU at a time: compare the alternative retriever after deeper controls."""
import os,sys,time,subprocess,psutil
from common import ROOT,read,dump,now
status=ROOT/'artifacts/jobs/giga_pipeline.json';record=dict(pid=os.getpid(),state='waiting',started=now());dump(status,record)
try:
    while True:
        p=read(ROOT/'artifacts/jobs/full50.json')
        if p['state']=='complete':assert p['exit_code']==0;break
        if p['state']=='failed':raise RuntimeError('Full50 failed')
        assert any(psutil.pid_exists(p[k]) for k in ['pid','child_pid'] if k in p),'Full50 is not live'
        time.sleep(20)
    # The40-pair parity child is allowed to finish after its waiting controller
    # is reprioritized. Never overlap its GPU allocation with Giga.
    parity_job=read(ROOT/'artifacts/jobs/continuation.json');parity_pid=parity_job.get('child_pid')
    while parity_pid and psutil.pid_exists(parity_pid):time.sleep(10)
    manifest=ROOT/'artifacts/model_giga.json'
    while not manifest.exists():
        active=[]
        for proc in psutil.process_iter(['pid','cmdline']):
            cmd=proc.info['cmdline'] or []
            if any('phase3_20261116' in part and 'download_model.py' in part for part in cmd):active.append(proc.pid)
        if not active:raise RuntimeError('Giga weights not ready and downloader is not live')
        record.update(state='waiting_for_download',download_pids=active);dump(status,record);time.sleep(30)
    def call(script,*args):
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts'/script),*args],cwd=ROOT)
        record.update(state='running',stage=script,args=args,child_pid=child.pid);dump(status,record);code=child.wait()
        if code:raise RuntimeError(script+' failed '+str(code))
    dump(ROOT/'artifacts/giga_promotion_policy.json',dict(policy='Screen three full-dev-selected pools, promote best top20 screen if within5% of best deeper screen; then full50, full100 if full50 improves available full-depth control by>.002',screen_margin=.95,full_margin=.002))
    call('giga_encode.py','--smoke');call('giga_encode.py');call('giga_pool.py')
    parity=ROOT/'artifacts/adaptive_parity.json'
    legacy=[] if parity.exists() and read(parity)['passed'] else ['--legacy']
    record['reranking_runtime']='legacy_batch2' if legacy else 'validated_adaptive';dump(status,record)
    screens=[]
    for i in [1,2,3]:
        pool='giga_pool'+str(i)+'_dev';call('qwen_adaptive.py','--pool',pool,'--depth','20',*legacy)
        item=read(ROOT/'artifacts'/('result_adaptive_'+pool+'_d20_screen.json'));screens.append((item['metrics']['ndcg_cut_10'],pool))
    score,pool=max(screens);reference=read(ROOT/'artifacts/score_controls_trials.json')['best']['ndcg10']
    record.update(screen_best=dict(score=score,pool=pool),screen_reference=reference);dump(status,record)
    if score>=.95*reference:
        call('qwen_adaptive.py','--pool',pool,'--depth','50','--full',*legacy)
        full=read(ROOT/'artifacts'/('result_adaptive_'+pool+'_d50_full.json'))['metrics']['ndcg_cut_10']
        prior=ROOT/'artifacts/result_adaptive_score_fusion_dev_d100_full.json'
        if not prior.exists():prior=ROOT/'artifacts/result_qwen4_fusion_d100_l2048_full.json'
        if not prior.exists():prior=ROOT/'artifacts/result_qwen4_fusion_d50_l2048_full.json'
        record['full_depth_reference']=str(prior.name);dump(status,record)
        if full>read(prior)['metrics']['ndcg_cut_10']+.002:call('qwen_adaptive.py','--pool',pool,'--depth','100','--full',*legacy)
    record.update(state='complete',finished=now(),exit_code=0);dump(status,record)
except BaseException as e:
    record.update(state='failed',error=str(e),finished=now());dump(status,record);raise
