"""Offline IR metric calculation from exported rankings, grades, and ideal gains."""
import math
from collections import defaultdict
from common import ROOT,jsonl,read

def main():
    summary=read(ROOT/'artifacts/test_relevance_summary.json');per=read(ROOT/'artifacts/per_query_metrics.json');expected=read(ROOT/'artifacts/results.json')['metrics'];groups=defaultdict(list)
    for row in jsonl(ROOT/'artifacts/predictions.jsonl'):groups[(row['method'],row['query_id'])].append(row)
    computed=defaultdict(list)
    for (name,q),rows in groups.items():
        rows=sorted(rows,key=lambda r:r['rank']);assert [r['rank'] for r in rows]==list(range(1,len(rows)+1))
        idcg=sum(g/math.log2(i+2) for i,g in enumerate(summary[q]['ideal_grades']));dcg=sum(r['relevance']/math.log2(r['rank']+1) for r in rows[:10]);ndcg=dcg/idcg if idcg else 0
        assert abs(ndcg-per[name][q]['ndcg_cut_10'])<1e-10
        relevant=summary[q]['relevant_documents'];hits=0;ap=0
        for r in rows[:10]:
            if r['relevance']>0:hits+=1;ap+=hits/r['rank']
        # TREC map_cut divides by all relevant documents, not min(R,k).
        assert abs(ap/relevant-per[name][q]['map_cut_10'])<1e-10
        for k in [10,100]:assert abs(sum(r['relevance']>0 for r in rows[:k])/relevant-per[name][q]['recall_'+str(k)])<1e-10
        computed[name].append(ndcg)
    for name,values in computed.items():assert len(values)==323 and abs(sum(values)/len(values)-expected[name]['ndcg_cut_10'])<1e-10
    print('Public grades/rankings reproduce official nDCG, MAP and recall for all 323 test queries')
if __name__=='__main__':main()
