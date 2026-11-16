"""Independent reconstruction of phase-two public ranks, selection and intervals."""
import math,json,hashlib
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'experiments/phase2/artifacts'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def check(path,summary,expected,per,query_sets):
    grouped=defaultdict(list)
    for line in path.read_text(encoding='utf-8').splitlines():
        r=json.loads(line);grouped[(r['method'],r['query_id'])].append(r)
    assert set(grouped)=={(n,q) for n,qs in query_sets.items() for q in qs}
    actual=defaultdict(lambda:defaultdict(list))
    for (n,q),rows in grouped.items():
        rows.sort(key=lambda r:r['rank']);assert len(rows)==len({r['doc_id'] for r in rows})==100
        assert [r['rank'] for r in rows]==list(range(1,101))
        ideal=sum(g/math.log2(i+2) for i,g in enumerate(summary[q]['ideal_grades']));rel=summary[q]['relevant_documents'];hits=0;ap=0.;reciprocal=0.
        for r in rows[:10]:
            if r['relevance']>0:hits+=1;ap+=hits/r['rank'];reciprocal=reciprocal or 1/r['rank']
        values=dict(ndcg_cut_10=sum(r['relevance']/math.log2(r['rank']+1) for r in rows[:10])/ideal if ideal else 0.,map_cut_10=ap/rel,recall_10=hits/rel,recall_100=sum(r['relevance']>0 for r in rows)/rel,P_10=hits/10,mrr_10=reciprocal)
        for k,v in values.items():assert abs(v-per[n][q][k])<1e-10,(n,q,k);actual[n][k].append(v)
    for n,values in actual.items():
        for k,v in values.items():assert abs(sum(v)/len(v)-expected[n][k])<1e-10,(n,k)
    return grouped
def main():
    r=read(ART/'final_results.json');summary=read(ROOT/'artifacts/test_relevance_summary.json')
    check(ART/'final_predictions.jsonl',summary,r['metrics'],r['per_query'],{n:set(summary) for n in r['metrics']})
    expected={};per={};sets={}
    for path in ART.glob('result_*.json'):
        item=read(path);name=item['name'];expected[name]=item['metrics'];per[name]=item['per_query'];sets[name]=set(item['per_query'])
    dev_summary=read(ART/'dev_relevance_summary.json');groups=check(ART/'dev_predictions.jsonl',dev_summary,expected,per,sets)
    # Reconstruct all15 final dev fusion trials, without models or private qrels.
    trials=read(ART/'selection_trials_phase2.json');names=['qwen4_fusion_d20_l2048_full','phase1_selected_dev','score_fusion_dev']
    for trial in trials['trials']:
        values=[]
        for q in dev_summary:
            scores={};grades={}
            for name,w in zip(names,[1.,trial['phase1_weight'],trial['retrieval_weight']]):
                for row in groups[(name,q)]:
                    d=row['doc_id'];scores[d]=scores.get(d,0)+w/(60+row['rank']);grades[d]=row['relevance']
            ordered=sorted(scores,key=lambda d:(-scores[d],d))[:10];ideal=sum(g/math.log2(i+2) for i,g in enumerate(dev_summary[q]['ideal_grades']))
            values.append(sum(grades[d]/math.log2(i+2) for i,d in enumerate(ordered))/ideal if ideal else 0.)
        assert abs(sum(values)/len(values)-trial['dev_ndcg10'])<1e-10
    assert max(trials['trials'],key=lambda t:t['dev_ndcg10'])==trials['selected']
    import numpy as np
    ids=list(summary);rng=np.random.default_rng(10);sample=rng.integers(0,len(ids),size=(2000,len(ids)))
    for name,v in r['bootstrap']['comparisons'].items():
        delta=np.array([r['per_query']['phase2_selected'][q]['ndcg_cut_10']-r['per_query'][name][q]['ndcg_cut_10'] for q in ids]);ci=np.quantile(delta[sample].mean(1),[.025,.975])
        assert abs(float(delta.mean())-v['delta'])<1e-12 and np.max(np.abs(ci-v['ci95']))<1e-10
    value=r['metrics']['phase2_selected']['ndcg_cut_10'];anchor=r['published_reference']['ndcg10']
    assert abs(r['relative_gain_vs_published']-(value/anchor-1))<1e-12
    assert r['target_5pct_met']==(value>=anchor*1.05) and r['target_15pct_met']==(value>=anchor*1.15)
    # Audit exact sources frozen before test prediction.
    lock=read(ART/'evaluation_lock.json')
    for name,digest in lock['source_sha256'].items():assert hashlib.sha256((ART.parent/'scripts'/name).read_bytes()).hexdigest()==digest,name
    assert read(ART/'inference_parity.json')['passed']
    print(f"Phase2: all dev screens,15 selection trials,six full323-query test systems,bootstrap CIs and {100*r['relative_gain_vs_published']:.2f}% gain independently verified")
if __name__=='__main__':main()
