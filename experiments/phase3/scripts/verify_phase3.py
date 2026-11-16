"""Independent arithmetic audit of phase3 public evidence and dev selection."""
import argparse,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def rows(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s]

def metrics(ranking,grades,summary):
    ideal=sum(g/math.log2(i+2) for i,g in enumerate(summary['ideal_grades']))
    relevant=summary['relevant_documents'];assert relevant>0
    top=ranking[:10];hits=0;ap=0.;rr=0.
    for i,d in enumerate(top,1):
        if grades.get(d,0)>0:hits+=1;ap+=hits/i;rr=rr or 1/i
    return dict(ndcg_cut_10=sum(grades.get(d,0)/math.log2(i+2) for i,d in enumerate(top))/ideal if ideal else 0.,
        map_cut_10=ap/relevant,recall_10=hits/relevant,recall_100=sum(grades.get(d,0)>0 for d in ranking[:100])/relevant,P_10=hits/10,mrr_10=rr)

def ranked(pool,prob,config):
    original=sorted(pool,key=lambda d:(-pool[d],d));depth=config['depth']
    if not depth:return original
    head=original[:depth];prior={d:i for i,d in enumerate(original)}
    assert all(prob[d] is not None for d in head)
    tie=lambda d:prior[d] if config['tie']=='retrieval' else d
    scored=sorted(head,key=lambda d:(-prob[d],tie(d)))
    if config['mode']=='rank_rrf':
        values={d:1/(61+i)+config['weight']/(61+prior[d]) for i,d in enumerate(scored)}
    else:
        assert config['mode']=='zlogit'
        p=np.clip(np.array([prob[d] for d in scored],dtype=np.float64),1e-7,1-1e-7)
        x=np.log(p/(1-p));x=(x-x.mean())/max(x.std(),1e-12)
        y=-np.array([prior[d] for d in scored],dtype=float);y=(y-y.mean())/max(y.std(),1e-12)
        values=dict(zip(scored,x+config['weight']*y))
    return sorted(head,key=lambda d:(-values[d],tie(d)))+original[depth:]

def equal(actual,expected):
    assert set(actual)==set(expected)
    for k,v in actual.items():assert abs(v-expected[k])<1e-10,(k,v,expected[k])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--artifacts',type=Path,default=Path(__file__).resolve().parents[1]/'artifacts');a=ap.parse_args();art=a.artifacts
    manifest=read(art/'public_evidence_manifest.json')
    for name,digest in manifest['files'].items():assert hashlib.sha256((art/name).read_bytes()).hexdigest()==digest,name
    selected=read(art/'selection_phase3.json');assert not selected['interim']
    summary=read(art/'dev_relevance_summary.json');assert len(summary)==324
    grouped=defaultdict(list)
    for row in rows(art/'dev_candidate_evidence.jsonl'):grouped[(row['pool'],row['query_id'])].append(row)
    pools={};probs={};grades={}
    for key,items in grouped.items():
        items.sort(key=lambda r:r['rank']);assert len(items)==len({r['doc_id'] for r in items})==100
        assert [r['rank'] for r in items]==list(range(1,101))
        pools[key]={r['doc_id']:r['retrieval_score'] for r in items}
        probs[key]={r['doc_id']:r['qwen_probability'] for r in items}
        grades[key]={r['doc_id']:r['relevance'] for r in items}
    assert set(grouped)=={(p,q) for p in selected['grid']['pools'] for q in summary}
    for trial in selected['trials']:
        config=trial['config'];per={}
        for q in summary:
            key=(config['pool'],q);per[q]=metrics(ranked(pools[key],probs[key],config),grades[key],summary[q])
        average={k:sum(v[k] for v in per.values())/len(per) for k in next(iter(per.values()))};equal(average,trial['metrics'])
        if config==selected['selected']:
            equal(average,selected['metrics'])
            for q in per:equal(per[q],selected['per_query'][q])
    assert max(selected['trials'],key=lambda t:t['metrics']['ndcg_cut_10'])['config']==selected['selected']
    screen=read(art/'protocol.json')['screen_query_ids'];assert len(screen)==64 and set(screen)<=set(summary)
    old_controls=read(art/'score_controls_trials.json')
    for config in old_controls['trials']:
        values=[]
        for q in screen:
            key=('score_fusion_dev',q)
            values.append(metrics(ranked(pools[key],probs[key],config),grades[key],summary[q])['ndcg_cut_10'])
        assert abs(sum(values)/64-config['ndcg10'])<1e-10
    assert max(old_controls['trials'],key=lambda t:t['ndcg10'])==old_controls['best']
    giga_controls=read(art/'giga_screen_controls_trials.json')
    for trial in giga_controls['trials']:
        config=trial['config'];values=[]
        for q in screen:
            key=(config['pool'],q);values.append(metrics(ranked(pools[key],probs[key],config),grades[key],summary[q]))
        equal({k:sum(v[k] for v in values)/64 for k in values[0]},trial['metrics'])
    assert max(giga_controls['trials'],key=lambda t:t['metrics']['ndcg_cut_10'])==giga_controls['best']
    final=read(art/'final_results.json');test_summary=read(art/'test_relevance_summary.json');assert len(test_summary)==323
    prior=read(art.parents[2]/'experiments/phase2/artifacts/final_results.json')
    assert final['baseline_phase2']==prior['metrics']['phase2_selected'] and final['baseline_phase2_per_query']==prior['per_query']['phase2_selected']
    test=defaultdict(list)
    for row in rows(art/'final_predictions.jsonl'):test[(row['method'],row['query_id'])].append(row)
    expected={'phase3_selected':(final['metrics'],final['per_query']),'phase3_retrieval':(final['retrieval_metrics'],final['retrieval_per_query'])}
    assert set(test)=={(n,q) for n in expected for q in test_summary}
    for name,(aggregate,per) in expected.items():
        actual={}
        for q in test_summary:
            items=sorted(test[(name,q)],key=lambda r:r['rank']);assert len(items)==len({r['doc_id'] for r in items})==100
            assert [r['rank'] for r in items]==list(range(1,101))
            actual[q]=metrics([r['doc_id'] for r in items],{r['doc_id']:r['relevance'] for r in items},test_summary[q]);equal(actual[q],per[q])
        equal({k:sum(v[k] for v in actual.values())/323 for k in aggregate},aggregate)
    lock=read(art/'evaluation_lock.json');pair_groups=defaultdict(list)
    for row in rows(art/'test_pair_evidence.jsonl'):pair_groups[row['query_id']].append(row)
    count=read(art/'test_pair_audit.json');depth=lock['selection']['depth']
    assert count['queries']==323 and count['depth']==depth and count['needed_pairs']==323*depth
    assert count['reused_in_actual_heads']+count['new_in_actual_heads']==count['needed_pairs']
    assert sum(r['source']=='phase2_reused' for values in pair_groups.values() for r in values)==count['reused_in_actual_heads']
    for q in test_summary:
        first=sorted(test[('phase3_retrieval',q)],key=lambda r:r['rank']);wanted=sorted(test[('phase3_selected',q)],key=lambda r:r['rank'])
        pairs=sorted(pair_groups[q],key=lambda r:r['head_rank']);assert [r['head_rank'] for r in pairs]==list(range(1,depth+1))
        assert [r['doc_id'] for r in pairs]==[r['doc_id'] for r in first[:depth]]
        pool={r['doc_id']:float(101-r['rank']) for r in first};prob={r['doc_id']:r['qwen_probability'] for r in pairs}
        assert ranked(pool,prob,lock['selection'])==[r['doc_id'] for r in wanted]
    ids=list(test_summary);delta=np.array([final['per_query'][q]['ndcg_cut_10']-final['baseline_phase2_per_query'][q]['ndcg_cut_10'] for q in ids])
    indices=np.random.default_rng(10).integers(0,323,size=(2000,323));ci=np.quantile(delta[indices].mean(1),[.025,.975]);boot=final['paired_bootstrap']
    assert abs(float(delta.mean())-boot['delta'])<1e-12 and np.max(np.abs(ci-boot['ci95']))<1e-10
    value=final['metrics']['ndcg_cut_10'];baseline=final['baseline_phase2']['ndcg_cut_10']
    assert abs(final['relative_gain_vs_own']-(value/baseline-1))<1e-12
    assert final['target_5pct_met']==(value>=baseline*1.05)
    assert hashlib.sha256((art/'evaluation_lock.json').read_bytes()).hexdigest()==final['lock_sha256']
    for name,digest in lock['source_sha256'].items():assert hashlib.sha256((art.parent/'scripts'/name).read_bytes()).hexdigest()==digest,name
    print('Phase3 verified:',len(selected['trials']),'complete dev trials,72old/75Giga screen controls,324dev/323test queries,all metrics,exact test score fusion,paired bootstrap and frozen source hashes')
if __name__=='__main__':main()
