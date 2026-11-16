"""Continue only after a verified predecessor exits successfully."""
import time,subprocess,sys,os
import psutil
from common import ROOT,read,dump,now

predecessor=ROOT/'artifacts/jobs/full50.json';status=ROOT/'artifacts/jobs/full100.json'
record=dict(job='full100',pid=os.getpid(),started=now(),state='waiting',predecessor='full50');dump(status,record)
while True:
    before=read(predecessor)
    if before['state']=='complete':
        assert before['exit_code']==0 and (ROOT/'artifacts/result_qwen4_fusion_d50_l2048_full.json').exists()
        break
    if before['state']=='failed':raise RuntimeError('Predecessor failed; inspect its log')
    live=any(psutil.pid_exists(before[k]) for k in ['pid','child_pid'] if k in before)
    if not live:raise RuntimeError('Predecessor not live and lacks terminal status; audit before resuming')
    print('WAITING FOR LIVE FULL50',before.get('child_pid'),flush=True);time.sleep(30)
child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/qwen_probe.py'),'--depth','100','--full'],cwd=ROOT)
record.update(state='running',child_pid=child.pid);dump(status,record)
code=child.wait();record.update(state='complete' if code==0 else 'failed',exit_code=code,finished=now());dump(status,record);sys.exit(code)
