"""Run the single locked test prediction/evaluation and its numerical audit."""
import os,sys,subprocess,psutil
from common import ROOT,read,dump,sha,now
from frozen_candidates import assert_sources

status=ROOT/'artifacts/jobs/final_test.json';record=dict(pid=os.getpid(),state='starting',started=now())
try:
    lock=read(ROOT/'artifacts/evaluation_lock.json');assert_sources(lock)
    record.update(lock_sha256=sha(ROOT/'artifacts/evaluation_lock.json'),selection=lock['selection']);dump(status,record)
    for name in ['giga_pipeline','parity']:
        item=read(ROOT/'artifacts/jobs'/(name+'.json'));assert item['state']=='complete' and item['exit_code']==0
        for key in ['pid','child_pid']:
            if item.get(key) and psutil.pid_exists(item[key]):raise RuntimeError('Earlier worker still alive: '+str(item[key]))
    def call(script):
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts'/script)],cwd=ROOT)
        record.update(state='running',stage=script,child_pid=child.pid);dump(status,record)
        code=child.wait()
        if code:raise RuntimeError(script+' failed '+str(code))
    call('predict_phase3.py')
    if not (ROOT/'artifacts/final_results.json').exists():call('evaluate_phase3.py')
    else:assert read(ROOT/'artifacts/final_results.json')['lock_sha256']==record['lock_sha256']
    call('export_phase3.py');call('verify_phase3.py')
    record.update(state='complete',exit_code=0,finished=now(),final_results_sha256=sha(ROOT/'artifacts/final_results.json'));dump(status,record)
except BaseException as e:
    record.update(state='failed',error=str(e),finished=now());dump(status,record);raise
