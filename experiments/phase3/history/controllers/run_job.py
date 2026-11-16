"""Durable local worker with explicit child process and terminal status."""
import sys,subprocess,os,re,traceback
from common import ROOT,dump,now

name,script,*extra=sys.argv[1:]
assert re.fullmatch(r'[a-zA-Z0-9_-]+',name)
assert re.fullmatch(r'[a-zA-Z0-9_-]+\.py',script)
status=ROOT/'artifacts/jobs'/(name+'.json')
record=dict(job=name,script=script,args=extra,pid=os.getpid(),started=now(),state='starting')
dump(status,record)
try:
    child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts'/script),*extra],cwd=ROOT)
    record.update(child_pid=child.pid,state='running');dump(status,record)
    code=child.wait();record.update(exit_code=code,state='complete' if code==0 else 'failed',finished=now());dump(status,record)
except BaseException as e:
    record.update(state='failed',finished=now(),error=type(e).__name__);dump(status,record);raise
sys.exit(code)
