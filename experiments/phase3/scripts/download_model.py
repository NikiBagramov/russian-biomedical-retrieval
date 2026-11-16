"""Resume official large weights with bounded HTTP ranges and official SHA256."""
from pathlib import Path
import argparse
import hashlib,json,time,requests,re
from huggingface_hub import HfApi

from study import ROOT,PORTFOLIO
root=PORTFOLIO
ap=argparse.ArgumentParser();ap.add_argument('--repo',default='Qwen/Qwen3-Reranker-4B');ap.add_argument('--revision',default='22e683669bc0f0bd69640a1354a6d0aebcfeede5');ap.add_argument('--alias',default='qwen_reranker4');ap.add_argument('--manifest',default='model_qwen4.json');args=ap.parse_args()
assert re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',args.repo) and re.fullmatch(r'[a-f0-9]{40}',args.revision)
assert re.fullmatch(r'[A-Za-z0-9_-]+',args.alias) and re.fullmatch(r'[A-Za-z0-9_-]+\.json',args.manifest)
repo=args.repo;rev=args.revision
dest=root/'cache/models'/args.alias;dest.mkdir(parents=True,exist_ok=True)
started=time.perf_counter();session=requests.Session()
info=HfApi().model_info(repo,revision=rev,files_metadata=True)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

records={}
for item in info.siblings:
    if not item.lfs or not item.rfilename.endswith('.safetensors'):continue
    target=dest/item.rfilename;expected=item.lfs.sha256;size=item.size
    if target.exists() and target.stat().st_size==size and digest(target)==expected:
        records[item.rfilename]=expected;print(item.rfilename,'already verified',flush=True);continue
    existing=list((dest/'.cache/huggingface/download').rglob('*.'+expected+'.incomplete'))
    partial=max(existing,key=lambda p:p.stat().st_size) if existing else target.with_suffix('.safetensors.part')
    assert partial.resolve().is_relative_to(dest.resolve()) and target.resolve().is_relative_to(dest.resolve())
    failures=0;last_print=-1
    while not partial.exists() or partial.stat().st_size<size:
        offset=partial.stat().st_size if partial.exists() else 0;end=min(size-1,offset+32*1024**2-1)
        try:
            with session.get(f'https://huggingface.co/{repo}/resolve/{rev}/{item.rfilename}?download=true',headers={'Range':f'bytes={offset}-{end}'},stream=True,timeout=(15,30)) as response:
                response.raise_for_status();match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                assert response.status_code==206 and match and tuple(map(int,match.groups()))==(offset,end,size),'Unexpected HTTP range'
                count=0
                with partial.open('ab') as output:
                    for data in response.iter_content(1024**2):output.write(data);count+=len(data)
                assert count==end-offset+1,'Short HTTP range'
            failures=0
            percent=int(100*partial.stat().st_size/size)
            if percent//5!=last_print:last_print=percent//5;print(item.rfilename,percent,'%',partial.stat().st_size,'/',size,flush=True)
        except Exception as e:
            failures+=1;print(item.rfilename,'range retry',failures,type(e).__name__,flush=True)
            session.close();session=requests.Session()
            if failures>=6:raise
            time.sleep(min(10,failures*2))
    assert partial.stat().st_size==size and digest(partial)==expected,'Weight checksum mismatch'
    partial.replace(target);records[item.rfilename]=expected
    print(item.rfilename,'SHA256 verified',flush=True)
for item in info.siblings:
    if not item.rfilename.endswith(('.json','.txt','.jinja','.py','.md')):continue
    target=dest/item.rfilename
    if target.exists():continue
    target.parent.mkdir(parents=True,exist_ok=True)
    response=session.get(f'https://huggingface.co/{repo}/resolve/{rev}/{item.rfilename}',timeout=(15,60));response.raise_for_status();target.write_bytes(response.content)
for filename in ['config.json','tokenizer.json','tokenizer_config.json']:assert (dest/filename).exists()
out=ROOT/'artifacts'/args.manifest
out.write_text(json.dumps(dict(repo=repo,revision=rev,weights_sha256=records,seconds=time.perf_counter()-started,download='bounded32MiB HTTP ranges from official Hugging Face; resumed verified-size prefix'),indent=2)+'\n',encoding='utf-8',newline='\n')
print('MODEL VERIFIED',repo,flush=True)
