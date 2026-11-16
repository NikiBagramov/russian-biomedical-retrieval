"""Recreate dev-only inputs; no test relevance is copied or read."""
from study import *
import shutil

assert not (ROOT/'artifacts/evaluation_lock.json').exists()
(ROOT/'data').mkdir(exist_ok=True);(ROOT/'cache').mkdir(exist_ok=True)
protocol=read(ROOT/'artifacts/protocol.json');rels=read(BASE/'data/processed/development_qrels.json')['dev']
assert set(rels)==set(protocol['full_dev_query_ids'])
dump(ROOT/'data/dev_qrels.json',rels)
rows=[r for r in jsonl(BASE/'data/processed/queries.jsonl') if r['split']=='dev']
from common import write_jsonl
write_jsonl(ROOT/'data/queries.jsonl',rows);shutil.copyfile(BASE/'data/processed/corpus.jsonl',ROOT/'data/corpus.jsonl')
print('Phase2 dev-only inputs prepared',len(rows),flush=True)
