"""Final choice on all324dev; incomplete candidate sets never receive a score."""
import argparse
from itertools import product
from study import *
from ranking_math import order,rerank

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--coverage',action='store_true');ap.add_argument('--interim',action='store_true');a=ap.parse_args()
    assert not (ROOT/'artifacts/evaluation_lock.json').exists()
    ids,queries,docs,rels=inputs();scores=read(ROOT/'cache/pair_scores_qwen4_2048.json')
    names=['score_fusion_dev']+[p.stem[4:] for p in sorted((ROOT/'cache').glob('run_giga_pool*_dev.json'))]
    pools={n:read(ROOT/'cache'/('run_'+n+'.json')) for n in names};coverage=[]
    for name,pool in pools.items():
        for depth in [20,50,100]:
            missing=sum(q+'\t'+d not in scores for q in ids for d in order(pool[q])[:depth]);coverage.append(dict(pool=name,depth=depth,missing=missing))
    if a.coverage:print(coverage,flush=True);return
    conformance=read(ROOT/'artifacts/ranking_math_conformance.json');assert conformance['passed'] and conformance['source_sha256']==sha(ROOT/'scripts/ranking_math.py')
    if not a.interim:
        deeper=read(ROOT/'artifacts/jobs/continuation.json')
        if deeper['state']!='complete':assert read(ROOT/'artifacts/depth_family_disposition.json')['action']=='defer_full100'
        alternative=read(ROOT/'artifacts/jobs/giga_pipeline.json')
        assert alternative['state']=='complete' or (ROOT/'artifacts/alternative_family_disposition.json').exists()
    grid=dict(pools=names,depths=[20,50,100],ties=['doc_id','retrieval'],modes=['rank_rrf','zlogit'],weights=[0.,.05,.1,.25,.5,1.],criterion='maximum full324dev nDCG10; exact ties keep first enumerated config; no test selection')
    prefix='interim_' if a.interim else ''
    dump(ROOT/'artifacts'/(prefix+'final_grid_spec.json'),grid)
    trials=[];best=None
    for name,pool in pools.items():
        configurations=[dict(pool=name,depth=0,mode='retrieval',tie='doc_id',weight=0.)]
        configurations += [dict(pool=name,depth=d,tie=t,mode=m,weight=w) for d,t,m,w in product(grid['depths'],grid['ties'],grid['modes'],grid['weights']) if next(x['missing'] for x in coverage if x['pool']==name and x['depth']==d)==0]
        for config in configurations:
            run={q:rerank(pool[q],scores,q,config) for q in ids};metrics,per=evaluate(rels,run);item=dict(config=config,metrics=metrics);trials.append(item)
            if best is None or metrics['ndcg_cut_10']>best[0]:best=(metrics['ndcg_cut_10'],config,run,metrics,per)
    baseline=evaluate(rels,read(ROOT/'cache/run_qwen4_fusion_d20_l2048_full.json'))[0]
    dump(ROOT/'artifacts'/(prefix+'selection_phase3.json'),dict(selected=best[1],metrics=best[3],per_query=best[4],baseline_phase2_dev=baseline,relative_dev_gain=best[0]/baseline['ndcg_cut_10']-1,grid=grid,trials=trials,coverage=coverage,selection_split='dev324',interim=a.interim,source_sha256={n:sha(ROOT/'scripts'/n) for n in ['choose_phase3.py','ranking_math.py']}))
    dump(ROOT/'cache'/('run_phase3_'+prefix+'selected_dev.json'),best[2]);print('PHASE3 INTERIM' if a.interim else 'PHASE3 SELECTED',best[1],best[3],flush=True)
if __name__=='__main__':main()
