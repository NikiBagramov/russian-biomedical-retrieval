"""Sequential GPU continuation with auditable predecessor and parity checks."""
import os,sys,time,subprocess,psutil
from common import ROOT,read,dump,now
status=ROOT/'artifacts/jobs/continuation.json';record=dict(pid=os.getpid(),state='waiting',started=now());dump(status,record)
try:
    while True:
        p=read(ROOT/'artifacts/jobs/full50.json')
        if p['state']=='complete':assert p['exit_code']==0;break
        if p['state']=='failed':raise RuntimeError('Full50 failed')
        assert any(psutil.pid_exists(p[k]) for k in ['pid','child_pid'] if k in p),'Full50 is not live'
        time.sleep(20)
    def call(name,*args):
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts'/name),*args],cwd=ROOT)
        record.update(state='running',stage=name,args=args,child_pid=child.pid);dump(status,record);code=child.wait()
        if code:raise RuntimeError(name+' failed '+str(code))
    call('qwen_adaptive.py','--parity')
    if read(ROOT/'artifacts/adaptive_parity.json')['passed']:call('qwen_adaptive.py','--full','--depth','100')
    else:call('qwen_probe.py','--full','--depth','100')
    record.update(state='complete',finished=now(),exit_code=0);dump(status,record)
except BaseException as e:
    record.update(state='failed',error=str(e),finished=now());dump(status,record);raise
