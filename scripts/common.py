"""Local experiment utilities; standalone when installed from requirements.txt."""
import os,sys,json,hashlib,random
from pathlib import Path
from datetime import datetime,timezone
os.environ.setdefault('USE_TF','0');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,os.environ.get('NLP_SHARED_DEPS',str(ROOT/'.deps')))

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.part');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n');tmp.replace(path)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def jsonl(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
def write_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8',newline='\n')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def text_hash(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def model_path(alias):
    cache=Path(os.environ.get('NLP_SHARED_MODELS',str(ROOT/'cache/models')))
    aliases=read(cache/'aliases.json') if (cache/'aliases.json').exists() else {}
    return Path(aliases.get(alias,str(cache/alias)))
def seed(value):
    if value<7:raise ValueError('Experiment seeds begin at 7')
    import numpy as np
    import torch
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)
    torch.set_num_threads(4)
def now():return datetime.now(timezone.utc).isoformat()
def development_allowed():
    if (ROOT/'artifacts/evaluation_lock.json').exists():raise RuntimeError('Test has been evaluated. Use a new experiment directory for development.')
