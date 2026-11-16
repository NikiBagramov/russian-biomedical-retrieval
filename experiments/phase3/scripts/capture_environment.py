"""Record the actual inference environment without allocating another GPU context."""
import sys,platform,subprocess
from importlib.metadata import version
from common import ROOT,dump,now,sha

packages=['numpy','pandas','scipy','scikit-learn','torch','transformers','accelerate','huggingface-hub','safetensors','bitsandbytes','einops','psutil','pytrec-eval-terrier']
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True)
record=dict(created=now(),python=sys.version,executable=sys.executable,platform=platform.platform(),packages={n:version(n) for n in packages},gpu=gpu.stdout.strip(),source_sha256=sha(__file__),scope='Actual local phase3 inference runtime; stable requirements are a portable approximation, not the exact Torch development wheel')
dump(ROOT/'artifacts/runtime_manifest.json',record);print(record,flush=True)
