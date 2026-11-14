"""Download only pinned competition/benchmark sources and released model weights."""
import argparse,shutil,time
from common import ROOT,read,sha,model_path
from huggingface_hub import hf_hub_download,snapshot_download
import requests

ALLOW=['model.safetensors','model-*.safetensors','model.safetensors.index.json','config.json','generation_config.json','tokenizer*.json','special_tokens_map.json','merges.txt','vocab*.json','vocab.txt','sentencepiece.bpe.model','spiece.model','added_tokens.json','README.md','LICENSE','*.jinja']
def data_path(record):
    raw=ROOT/'data/raw';repo=record['repo']
    if repo.startswith('https://github.com/'):return raw/'rupaws'/record['file']
    if 'rus-nfcorpus' in repo:return raw/repo.split('/')[-1]/record['file']
    return raw/record['file']

def data(check=False):
    for record in read(ROOT/'artifacts/data_provenance.json'):
        target=data_path(record)
        if target.exists() and sha(target)==record['sha256']:continue
        if check:raise RuntimeError('Missing or modified benchmark file: '+str(target))
        target.parent.mkdir(parents=True,exist_ok=True)
        if record['repo'].startswith('https://github.com/'):
            repo=record['repo'].removeprefix('https://github.com/');url=f"https://raw.githubusercontent.com/{repo}/{record['revision']}/{record['file']}"
            response=requests.get(url,timeout=120);response.raise_for_status();target.write_bytes(response.content)
        else:
            cached=hf_hub_download(record['repo'],record['file'],revision=record['revision'],repo_type='dataset')
            shutil.copyfile(cached,target)
        assert sha(target)==record['sha256'],'Benchmark checksum mismatch'
    print('Pinned dataset checksums passed',flush=True)

def models(check=False):
    for alias,record in read(ROOT/'artifacts/model_provenance.json').items():
        path=model_path(alias)
        if not check:snapshot_download(record['repo'],revision=record['revision'],local_dir=path,allow_patterns=ALLOW,max_workers=3)
        for filename,digest in record['weights'].items():assert sha(path/filename)==digest,'Released model checksum mismatch: '+alias
        print('Pinned model passed',alias,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',action='store_true');ap.add_argument('--models',action='store_true');ap.add_argument('--check',action='store_true');a=ap.parse_args()
    if not a.data and not a.models:ap.error('Choose --data and/or --models; --check verifies without network')
    if a.data:data(a.check)
    if a.models:models(a.check)
