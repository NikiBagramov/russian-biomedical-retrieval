"""Capture actual runtime versions independently of the portable requirements."""
from common import ROOT,dump,now
import platform,importlib.metadata,subprocess
import torch
names=['numpy','pandas','scipy','scikit-learn','torch','transformers','accelerate','peft','huggingface-hub','bm25s','pytrec-eval-terrier','ru-errant','spacy','ru-core-news-lg','Levenshtein']
versions={}
for name in names:
    try:versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:pass
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True)
dump(ROOT/'artifacts/environment.json',dict(captured=now(),python=platform.python_version(),platform=platform.platform(),versions=versions,cuda=torch.version.cuda,gpu=gpu.stdout.strip(),threads=4,determinism='Seeded Python/NumPy/PyTorch. GPU kernels and batched BF16 can differ across hardware/library releases; statistical rather than bit-exact reproduction expected.'))
print('Runtime captured')
